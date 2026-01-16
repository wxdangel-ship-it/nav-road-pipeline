from __future__ import annotations
from pathlib import Path
import argparse
import json
import math
import re
import subprocess
from typing import Iterable

import yaml
from shapely.geometry import shape, mapping, LineString, Point
from shapely.ops import unary_union

from pipeline._io import ensure_dir, new_run_id, load_yaml


def _extract_geom_run_dir(text: str, repo: Path) -> Path | None:
    m = re.search(r"\[GEOM\]\s+DONE\s+->\s+([^\r\n(]+)", text)
    if not m:
        return None
    raw = m.group(1).strip().strip("\"'")
    p = Path(raw)
    if not p.is_absolute():
        p = repo / p
    return p


def _latest_geom_run(repo: Path) -> Path | None:
    runs_dir = repo / "runs"
    if not runs_dir.exists():
        return None
    geom_dirs = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("geom_")]
    if not geom_dirs:
        return None
    geom_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return geom_dirs[0] / "outputs"


def _run_geom_cmd(repo: Path, drive: str, max_frames: int) -> Path:
    cmd = [
        "cmd.exe",
        "/c",
        str(repo / "scripts" / "build_geom.cmd"),
        "--drive",
        drive,
        "--max-frames",
        str(max_frames),
    ]
    proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    run_dir = _extract_geom_run_dir(output, repo)
    if run_dir is None:
        run_dir = _latest_geom_run(repo)
    if proc.returncode != 0 or run_dir is None:
        raise SystemExit(f"ERROR: build_geom.cmd failed (code={proc.returncode}). Output:\n{output}")
    return run_dir


def _read_geojson(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    feats = data.get("features", [])
    geoms = []
    for f in feats:
        geom = f.get("geometry")
        if geom:
            geoms.append(shape(geom))
    return geoms


def _cluster_points(points: list[Point], tol: float) -> tuple[list[Point], list[int], list[float]]:
    centers: list[Point] = []
    assignment: list[int] = []
    dists: list[float] = []
    for p in points:
        best = -1
        best_d = 1e9
        for i, c in enumerate(centers):
            d = p.distance(c)
            if d < best_d:
                best = i
                best_d = d
        if best >= 0 and best_d <= tol:
            assignment.append(best)
            dists.append(best_d)
            # update centroid
            cx = (centers[best].x + p.x) / 2.0
            cy = (centers[best].y + p.y) / 2.0
            centers[best] = Point(cx, cy)
        else:
            centers.append(Point(p.x, p.y))
            assignment.append(len(centers) - 1)
            dists.append(0.0)
    return centers, assignment, dists


def _flatten_lines(geoms: Iterable) -> list[LineString]:
    out = []
    for g in geoms:
        if g.geom_type == "LineString":
            out.append(g)
        elif g.geom_type == "MultiLineString":
            out.extend(list(g.geoms))
    return out


def _write_geojson(path: Path, features: list[dict]) -> None:
    fc = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="2013_05_28_drive_0000_sync", help="drive name")
    ap.add_argument("--max-frames", type=int, default=2000, help="max frames for geom input")
    ap.add_argument("--snap-tol-m", type=float, default=1.5, help="node snap tolerance (m)")
    ap.add_argument("--spur-len-m", type=float, default=8.0, help="spur length threshold (m)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    run_id = new_run_id("topo")
    run_dir = ensure_dir(repo / "runs" / run_id)
    out_dir = ensure_dir(run_dir / "outputs")

    geom_dir = _run_geom_cmd(repo, args.drive, args.max_frames)
    centerlines = _flatten_lines(_read_geojson(geom_dir / "centerlines.geojson"))
    intersections = _read_geojson(geom_dir / "intersections.geojson")
    inter_union = unary_union(intersections) if intersections else None

    endpoints = []
    for line in centerlines:
        coords = list(line.coords)
        endpoints.append(Point(coords[0]))
        endpoints.append(Point(coords[-1]))

    centers, assignment, dists = _cluster_points(endpoints, args.snap_tol_m)
    node_degree = {i: 0 for i in range(len(centers))}

    edges = []
    issues = []
    removed_edges = 0
    dangling_count = 0
    issue_idx = 0

    for i, line in enumerate(centerlines):
        coords = list(line.coords)
        p0 = Point(coords[0])
        p1 = Point(coords[-1])
        a0 = assignment[2 * i]
        a1 = assignment[2 * i + 1]
        dist0 = dists[2 * i]
        dist1 = dists[2 * i + 1]
        edge_id = f"edge_{i:05d}"
        length_m = float(line.length)
        in_inter = False
        if inter_union and not inter_union.is_empty:
            in_inter = bool(p0.within(inter_union) or p1.within(inter_union))

        node_degree[a0] += 1
        node_degree[a1] += 1

        edges.append(
            {
                "edge_id": edge_id,
                "from_node": f"node_{a0:05d}",
                "to_node": f"node_{a1:05d}",
                "length_m": round(length_m, 3),
                "geometry": line,
                "in_intersection": in_inter,
                "dist_to_node_m": max(dist0, dist1),
            }
        )

        if dist0 > args.snap_tol_m or dist1 > args.snap_tol_m:
            issue_idx += 1
            issues.append(
                {
                    "issue_id": f"{run_id}_{issue_idx:04d}",
                    "tile_id": args.drive,
                    "involved_edges": [edge_id],
                    "involved_nodes": [f"node_{a0:05d}", f"node_{a1:05d}"],
                    "rule_failed": "EndpointSnap",
                    "severity": "S2",
                    "description": "Edge endpoint exceeds snap tolerance.",
                    "recommend_actions": ["merge_endpoints"],
                    "evidence_summary": {
                        "traj_support": "unknown",
                        "lidar_support": "unknown",
                        "image_support": "unknown",
                        "prior_conflict": "none",
                        "alignment_residual": None,
                    },
                    "error_code": "E08",
                }
            )

    # Remove short dangling spurs
    kept_edges = []
    for e in edges:
        n0 = int(e["from_node"].split("_")[1])
        n1 = int(e["to_node"].split("_")[1])
        if (node_degree[n0] == 1 or node_degree[n1] == 1) and e["length_m"] < args.spur_len_m:
            dangling_count += 1
            removed_edges += 1
            issue_idx += 1
            issues.append(
                {
                    "issue_id": f"{run_id}_{issue_idx:04d}",
                    "tile_id": args.drive,
                    "involved_edges": [e["edge_id"]],
                    "involved_nodes": [e["from_node"], e["to_node"]],
                    "rule_failed": "DanglingEnd",
                    "severity": "S1",
                    "description": "Short dangling spur removed.",
                    "recommend_actions": ["remove_spur"],
                    "evidence_summary": {
                        "traj_support": "unknown",
                        "lidar_support": "unknown",
                        "image_support": "unknown",
                        "prior_conflict": "none",
                        "alignment_residual": None,
                    },
                    "error_code": "E08",
                }
            )
        else:
            kept_edges.append(e)

    # Output nodes/edges
    node_features = []
    for i, p in enumerate(centers):
        node_features.append(
            {
                "type": "Feature",
                "geometry": mapping(p),
                "properties": {"node_id": f"node_{i:05d}", "degree": int(node_degree[i])},
            }
        )
    edge_features = []
    for e in kept_edges:
        edge_features.append(
            {
                "type": "Feature",
                "geometry": mapping(e["geometry"]),
                "properties": {
                    "edge_id": e["edge_id"],
                    "from_node": e["from_node"],
                    "to_node": e["to_node"],
                    "length_m": e["length_m"],
                    "in_intersection": bool(e["in_intersection"]),
                },
            }
        )

    _write_geojson(out_dir / "graph_nodes.geojson", node_features)
    _write_geojson(out_dir / "graph_edges.geojson", edge_features)

    issues_path = out_dir / "TopoIssues.jsonl"
    with issues_path.open("w", encoding="utf-8") as f:
        for item in issues:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "run_id": run_id,
        "node_count": len(node_features),
        "edge_count": len(edge_features),
        "dangling_removed": removed_edges,
        "dangling_total": dangling_count,
        "issues_count": len(issues),
    }
    (out_dir / "TopoSummary.md").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[TOPO] DONE -> {out_dir}")
    print("outputs:")
    print(f"- {out_dir / 'graph_nodes.geojson'}")
    print(f"- {out_dir / 'graph_edges.geojson'}")
    print(f"- {out_dir / 'TopoIssues.jsonl'}")
    print(f"- {out_dir / 'TopoSummary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

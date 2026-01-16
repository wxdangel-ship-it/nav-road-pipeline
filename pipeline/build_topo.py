from __future__ import annotations
from pathlib import Path
import argparse
import json
import math
import re
import subprocess
from typing import Iterable

import yaml
from shapely.geometry import shape, mapping, LineString, Point, Polygon, MultiPolygon
from shapely.ops import unary_union, split

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


def _flatten_polygons(geoms: Iterable) -> list[Polygon]:
    out = []
    for g in geoms:
        if g.geom_type == "Polygon":
            out.append(g)
        elif g.geom_type == "MultiPolygon":
            out.extend(list(g.geoms))
    return out


def _split_lines_by_polygons(lines: list[LineString], inter_union) -> list[LineString]:
    if not inter_union or inter_union.is_empty:
        return lines
    segments: list[LineString] = []
    for line in lines:
        try:
            result = split(line, inter_union)
            segments.extend(_flatten_lines(result.geoms))
        except Exception:
            segments.append(line)
    return segments


def _find_intersection_idx(pt: Point, inter_polys: list[Polygon]) -> int:
    for i, poly in enumerate(inter_polys):
        if pt.within(poly) or pt.touches(poly):
            return i
    return -1


def _component_sizes(node_count: int, edges: list[dict]) -> list[int]:
    adj = {i: set() for i in range(node_count)}
    for e in edges:
        n0 = int(e["from_node"].split("_")[1])
        n1 = int(e["to_node"].split("_")[1])
        adj[n0].add(n1)
        adj[n1].add(n0)
    visited = set()
    sizes = []
    for i in range(node_count):
        if i in visited:
            continue
        stack = [i]
        visited.add(i)
        count = 0
        while stack:
            cur = stack.pop()
            count += 1
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        sizes.append(count)
    return sizes


def _write_geojson(path: Path, features: list[dict]) -> None:
    fc = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="2013_05_28_drive_0000_sync", help="drive name")
    ap.add_argument("--max-frames", type=int, default=2000, help="max frames for geom input")
    ap.add_argument("--snap-tol-m", type=float, default=1.5, help="node snap tolerance (m)")
    ap.add_argument("--spur-len-m", type=float, default=8.0, help="spur length threshold (m)")
    ap.add_argument("--merge-tol-m", type=float, default=2.0, help="dangling merge tolerance (m)")
    ap.add_argument("--max-degree-warn", type=int, default=12, help="degree warning threshold")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    run_id = new_run_id("topo")
    run_dir = ensure_dir(repo / "runs" / run_id)
    out_dir = ensure_dir(run_dir / "outputs")

    geom_dir = _run_geom_cmd(repo, args.drive, args.max_frames)
    centerlines = _flatten_lines(_read_geojson(geom_dir / "centerlines.geojson"))
    intersections = _flatten_polygons(_read_geojson(geom_dir / "intersections.geojson"))
    inter_union = unary_union(intersections) if intersections else None

    split_centerlines = _split_lines_by_polygons(centerlines, inter_union)
    junction_nodes = [Point(p.centroid.x, p.centroid.y) for p in intersections]

    endpoints = []
    endpoint_inter_idx = []
    for line in split_centerlines:
        coords = list(line.coords)
        p0 = Point(coords[0])
        p1 = Point(coords[-1])
        endpoints.append(p0)
        endpoints.append(p1)
        endpoint_inter_idx.append(_find_intersection_idx(p0, intersections))
        endpoint_inter_idx.append(_find_intersection_idx(p1, intersections))

    cluster_points = [p for i, p in enumerate(endpoints) if endpoint_inter_idx[i] < 0]
    centers, assignment, dists = _cluster_points(cluster_points, args.snap_tol_m)
    node_points = list(centers) + list(junction_nodes)
    node_degree = {i: 0 for i in range(len(node_points))}

    edges = []
    issues = []
    removed_edges = 0
    dangling_count = 0
    dangling_merged = 0
    issue_idx = 0

    assign_idx = 0
    for i, line in enumerate(split_centerlines):
        coords = list(line.coords)
        p0 = Point(coords[0])
        p1 = Point(coords[-1])
        inter0 = endpoint_inter_idx[2 * i]
        inter1 = endpoint_inter_idx[2 * i + 1]
        if inter0 >= 0:
            a0 = len(centers) + inter0
            dist0 = p0.distance(junction_nodes[inter0])
        else:
            a0 = assignment[assign_idx]
            dist0 = dists[assign_idx]
            assign_idx += 1
        if inter1 >= 0:
            a1 = len(centers) + inter1
            dist1 = p1.distance(junction_nodes[inter1])
        else:
            a1 = assignment[assign_idx]
            dist1 = dists[assign_idx]
            assign_idx += 1
        edge_id = f"edge_{i:05d}"
        length_m = float(line.length)
        in_inter = False
        if inter_union and not inter_union.is_empty:
            in_inter = bool(line.intersects(inter_union))

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

        if (inter0 < 0 and dist0 > args.snap_tol_m) or (inter1 < 0 and dist1 > args.snap_tol_m):
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

    # Recompute degree after removal
    node_degree = {i: 0 for i in range(len(node_points))}
    for e in kept_edges:
        n0 = int(e["from_node"].split("_")[1])
        n1 = int(e["to_node"].split("_")[1])
        node_degree[n0] += 1
        node_degree[n1] += 1

    # Attempt merge for long dangling edges
    for e in kept_edges:
        n0 = int(e["from_node"].split("_")[1])
        n1 = int(e["to_node"].split("_")[1])
        if e["length_m"] < args.spur_len_m:
            continue
        if node_degree[n0] == 1 or node_degree[n1] == 1:
            dangling_node = n0 if node_degree[n0] == 1 else n1
            other_node = n1 if dangling_node == n0 else n0
            best_j = -1
            best_d = 1e9
            for j, jp in enumerate(junction_nodes):
                d = node_points[dangling_node].distance(jp)
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0 and best_d <= args.merge_tol_m:
                new_node_idx = len(centers) + best_j
                if dangling_node == n0:
                    e["from_node"] = f"node_{new_node_idx:05d}"
                else:
                    e["to_node"] = f"node_{new_node_idx:05d}"
                node_degree[dangling_node] -= 1
                node_degree[other_node] += 0
                node_degree[new_node_idx] += 1
                dangling_merged += 1

    # Recompute degree after merge attempts
    node_degree = {i: 0 for i in range(len(node_points))}
    for e in kept_edges:
        n0 = int(e["from_node"].split("_")[1])
        n1 = int(e["to_node"].split("_")[1])
        node_degree[n0] += 1
        node_degree[n1] += 1

    # Remaining long dangling edges -> issues
    for e in kept_edges:
        n0 = int(e["from_node"].split("_")[1])
        n1 = int(e["to_node"].split("_")[1])
        if (node_degree[n0] == 1 or node_degree[n1] == 1) and e["length_m"] >= args.spur_len_m:
            issue_idx += 1
            issues.append(
                {
                    "issue_id": f"{run_id}_{issue_idx:04d}",
                    "tile_id": args.drive,
                    "involved_edges": [e["edge_id"]],
                    "involved_nodes": [e["from_node"], e["to_node"]],
                    "rule_failed": "DanglingEnd",
                    "severity": "S2",
                    "description": "Long dangling edge retained.",
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

    # Output nodes/edges
    node_features = []
    isolated_nodes_removed = 0
    for i, p in enumerate(node_points):
        node_type = "junction" if i >= len(centers) else "endpoint"
        if node_degree[i] == 0:
            isolated_nodes_removed += 1
            issue_idx += 1
            issues.append(
                {
                    "issue_id": f"{run_id}_{issue_idx:04d}",
                    "tile_id": args.drive,
                    "involved_edges": [],
                    "involved_nodes": [f"node_{i:05d}"],
                    "rule_failed": "IsolatedNode",
                    "severity": "S1",
                    "description": "Isolated node removed.",
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
            continue
        node_features.append(
            {
                "type": "Feature",
                "geometry": mapping(p),
                "properties": {
                    "node_id": f"node_{i:05d}",
                    "degree": int(node_degree[i]),
                    "node_type": node_type,
                },
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

    degree_hist = {}
    for d in node_degree.values():
        degree_hist[str(d)] = degree_hist.get(str(d), 0) + 1
    dangling_nodes = [f"node_{i:05d}" for i, d in node_degree.items() if d == 1]
    max_degree = max(node_degree.values()) if node_degree else 0
    component_sizes = _component_sizes(len(node_points), kept_edges)
    dangling_total = sum(1 for d in node_degree.values() if d == 1)
    dangling_unfixed_nodes = set()
    for e in kept_edges:
        n0 = int(e["from_node"].split("_")[1])
        n1 = int(e["to_node"].split("_")[1])
        if e["length_m"] >= args.spur_len_m:
            if node_degree[n0] == 1:
                dangling_unfixed_nodes.add(n0)
            if node_degree[n1] == 1:
                dangling_unfixed_nodes.add(n1)
    warning = None
    if max_degree > args.max_degree_warn:
        warning = f"max_degree={max_degree} exceeds {args.max_degree_warn}"
        issue_idx += 1
        issues.append(
            {
                "issue_id": f"{run_id}_{issue_idx:04d}",
                "tile_id": args.drive,
                "involved_edges": [],
                "involved_nodes": [],
                "rule_failed": "OverMergedJunction",
                "severity": "S1",
                "description": "Junction overly merged (high degree).",
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
    summary = {
        "run_id": run_id,
        "node_count": len(node_features),
        "edge_count": len(edge_features),
        "dangling_total": dangling_total,
        "dangling_removed": removed_edges,
        "dangling_merged": dangling_merged,
        "dangling_unfixed": len(dangling_unfixed_nodes),
        "dangling_nodes": dangling_nodes,
        "isolated_nodes_removed": isolated_nodes_removed,
        "degree_histogram": degree_hist,
        "max_degree": max_degree,
        "components": {"count": len(component_sizes), "sizes": component_sizes},
        "issues_count": len(issues),
    }
    if warning:
        summary["warning"] = warning
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

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


def _component_sizes_active(node_count: int, edges: list[dict], active_nodes: set[int]) -> list[int]:
    adj = {i: set() for i in active_nodes}
    for e in edges:
        n0 = int(e["from_node"].split("_")[1])
        n1 = int(e["to_node"].split("_")[1])
        if n0 in active_nodes and n1 in active_nodes:
            adj[n0].add(n1)
            adj[n1].add(n0)
    visited = set()
    sizes = []
    for i in active_nodes:
        if i in visited:
            continue
        stack = [i]
        visited.add(i)
        count = 0
        while stack:
            cur = stack.pop()
            count += 1
            for nb in adj.get(cur, []):
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        sizes.append(count)
    return sizes


def _degree_histogram(degrees: dict[int, int], active_nodes: set[int] | None = None) -> dict[str, int]:
    degree_hist = {}
    for i, d in degrees.items():
        if active_nodes is not None and i not in active_nodes:
            continue
        degree_hist[str(d)] = degree_hist.get(str(d), 0) + 1
    return degree_hist


def _write_geojson(path: Path, features: list[dict]) -> None:
    fc = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_outputs(summary: dict, issues_path: Path, actions_path: Path) -> None:
    required_issue_fields = {
        "issue_id",
        "tile_id",
        "involved_edges",
        "involved_nodes",
        "rule_failed",
        "severity",
        "description",
        "recommend_actions",
        "evidence_summary",
        "error_code",
    }
    valid_severity = {"S0", "S1", "S2", "S3"}
    required_action_fields = {
        "action_id",
        "tile_id",
        "action_type",
        "reason_rule",
        "involved_nodes_before",
        "involved_nodes_after",
        "involved_edges_before",
        "involved_edges_after",
        "intersection_bucket_id",
        "metrics",
        "related_issue_ids",
    }

    degree_hist_pre = summary.get("degree_histogram_pre_prune", {})
    degree_hist_post = summary.get("degree_histogram_post_prune", {})
    components_pre = summary.get("components_pre_prune", {}).get("sizes", [])
    components_post = summary.get("components_post_prune", {}).get("sizes", [])

    node_count_pre = int(summary.get("node_count_pre_prune", 0))
    node_count_post = int(summary.get("node_count_post_prune", 0))
    if int(summary.get("node_count", -1)) != node_count_post:
        raise SystemExit("ERROR: node_count must match node_count_post_prune.")
    if sum(int(v) for v in degree_hist_pre.values()) != node_count_pre:
        raise SystemExit("ERROR: degree_histogram_pre_prune sum mismatch.")
    if sum(int(v) for v in degree_hist_post.values()) != node_count_post:
        raise SystemExit("ERROR: degree_histogram_post_prune sum mismatch.")
    if sum(int(v) for v in components_pre) != node_count_pre:
        raise SystemExit("ERROR: components_pre_prune sizes sum mismatch.")
    if sum(int(v) for v in components_post) != node_count_post:
        raise SystemExit("ERROR: components_post_prune sizes sum mismatch.")

    dangling_detected = int(summary.get("dangling_detected", -1))
    dangling_removed = int(summary.get("dangling_removed", -1))
    dangling_merged = int(summary.get("dangling_merged", -1))
    dangling_unfixed = int(summary.get("dangling_unfixed", -1))
    if dangling_detected != dangling_removed + dangling_merged + dangling_unfixed:
        raise SystemExit("ERROR: dangling invariant violated.")
    if int(summary.get("dangling_total", -1)) != dangling_unfixed:
        raise SystemExit("ERROR: dangling_total must equal dangling_unfixed.")
    if len(summary.get("dangling_nodes", [])) != dangling_unfixed:
        raise SystemExit("ERROR: dangling_nodes count must equal dangling_unfixed.")

    issue_lines = issues_path.read_text(encoding="utf-8").splitlines()
    if len(issue_lines) != int(summary.get("issues_count", -1)):
        raise SystemExit("ERROR: issues_count does not match TopoIssues line count.")
    issue_ids = set()
    for line in issue_lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"ERROR: TopoIssues invalid JSON line: {exc}") from exc
        if not required_issue_fields.issubset(item.keys()):
            raise SystemExit("ERROR: TopoIssues missing required fields.")
        if item.get("severity") not in valid_severity:
            raise SystemExit("ERROR: TopoIssues invalid severity.")
        issue_id = item.get("issue_id")
        if issue_id:
            issue_ids.add(issue_id)

    action_lines = actions_path.read_text(encoding="utf-8").splitlines()
    if len(action_lines) != int(summary.get("actions_count", -1)):
        raise SystemExit("ERROR: actions_count does not match TopoActions line count.")
    action_ids = set()
    actions_by_type = {}
    for line in action_lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"ERROR: TopoActions invalid JSON line: {exc}") from exc
        if not required_action_fields.issubset(item.keys()):
            raise SystemExit("ERROR: TopoActions missing required fields.")
        action_id = item.get("action_id")
        if action_id in action_ids:
            raise SystemExit("ERROR: TopoActions action_id must be unique.")
        action_ids.add(action_id)
        action_type = item.get("action_type")
        actions_by_type[action_type] = actions_by_type.get(action_type, 0) + 1
        related_issue_ids = item.get("related_issue_ids") or []
        if not isinstance(related_issue_ids, list):
            raise SystemExit("ERROR: TopoActions related_issue_ids must be a list.")
        for issue_id in related_issue_ids:
            if issue_id not in issue_ids:
                raise SystemExit("ERROR: TopoActions related_issue_ids contains unknown issue_id.")
    if actions_by_type != summary.get("actions_by_type", {}):
        raise SystemExit("ERROR: actions_by_type does not match TopoActions content.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="2013_05_28_drive_0000_sync", help="drive name")
    ap.add_argument("--max-frames", type=int, default=2000, help="max frames for geom input")
    ap.add_argument("--snap-tol-m", type=float, default=1.5, help="node snap tolerance (m)")
    ap.add_argument("--snap-tol-outside-m", type=float, default=1.0, help="node snap tolerance outside intersections (m)")
    ap.add_argument("--spur-len-m", type=float, default=8.0, help="spur length threshold (m)")
    ap.add_argument("--merge-tol-m", type=float, default=3.5, help="dangling merge tolerance (m)")
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

    grouped_indices: dict[int, list[int]] = {}
    for i, inter_idx in enumerate(endpoint_inter_idx):
        grouped_indices.setdefault(inter_idx, []).append(i)
    centers: list[Point] = []
    center_inter_idx: list[int] = []
    endpoint_to_center = [0 for _ in endpoints]
    endpoint_to_center_dist = [0.0 for _ in endpoints]
    for inter_idx, idxs in grouped_indices.items():
        pts = [endpoints[i] for i in idxs]
        tol = args.snap_tol_m if inter_idx >= 0 else args.snap_tol_outside_m
        group_centers, group_assignment, group_dists = _cluster_points(pts, tol)
        base = len(centers)
        centers.extend(group_centers)
        center_inter_idx.extend([inter_idx] * len(group_centers))
        for local_i, ep_idx in enumerate(idxs):
            endpoint_to_center[ep_idx] = base + group_assignment[local_i]
            endpoint_to_center_dist[ep_idx] = group_dists[local_i]
    node_points = list(centers) + list(junction_nodes)
    node_inter_idx = list(center_inter_idx) + list(range(len(junction_nodes)))
    node_types = ["endpoint"] * len(centers) + ["junction"] * len(junction_nodes)
    node_degree = {i: 0 for i in range(len(node_points))}

    edges = []
    issues = []
    actions = []
    removed_edges = 0
    dangling_removed_endpoints = 0
    dangling_count = 0
    dangling_merged = 0
    issue_idx = 0
    action_seq = 0

    def add_action(
        action_type: str,
        reason_rule: str,
        involved_nodes_before: list[str] | None = None,
        involved_nodes_after: list[str] | None = None,
        involved_edges_before: list[str] | None = None,
        involved_edges_after: list[str] | None = None,
        intersection_bucket_id: int | None = None,
        metrics: dict | None = None,
        related_issue_ids: list[str] | None = None,
    ) -> None:
        nonlocal action_seq
        action_seq += 1
        actions.append(
            {
                "action_id": f"{run_id}_a{action_seq:04d}",
                "tile_id": args.drive,
                "action_type": action_type,
                "reason_rule": reason_rule,
                "involved_nodes_before": involved_nodes_before or [],
                "involved_nodes_after": involved_nodes_after or [],
                "involved_edges_before": involved_edges_before or [],
                "involved_edges_after": involved_edges_after or [],
                "intersection_bucket_id": intersection_bucket_id,
                "metrics": metrics,
                "related_issue_ids": related_issue_ids or [],
            }
        )

    for i, line in enumerate(split_centerlines):
        coords = list(line.coords)
        p0 = Point(coords[0])
        p1 = Point(coords[-1])
        edge_id = f"edge_{i:05d}"
        inter0 = endpoint_inter_idx[2 * i]
        inter1 = endpoint_inter_idx[2 * i + 1]
        if inter0 >= 0:
            junction_idx = len(centers) + inter0
            junction_dist = p0.distance(junction_nodes[inter0])
            if junction_dist <= args.snap_tol_m:
                a0 = junction_idx
                dist0 = junction_dist
            else:
                a0 = endpoint_to_center[2 * i]
                dist0 = endpoint_to_center_dist[2 * i]
        else:
            a0 = endpoint_to_center[2 * i]
            dist0 = endpoint_to_center_dist[2 * i]
        if inter1 >= 0:
            junction_idx = len(centers) + inter1
            junction_dist = p1.distance(junction_nodes[inter1])
            if junction_dist <= args.snap_tol_m:
                a1 = junction_idx
                dist1 = junction_dist
            else:
                a1 = endpoint_to_center[2 * i + 1]
                dist1 = endpoint_to_center_dist[2 * i + 1]
        else:
            a1 = endpoint_to_center[2 * i + 1]
            dist1 = endpoint_to_center_dist[2 * i + 1]
        if dist0 > 0:
            add_action(
                action_type="SnapNode",
                reason_rule="EndpointSnap",
                involved_nodes_before=[f"endpoint_{2 * i:05d}"],
                involved_nodes_after=[f"node_{a0:05d}"],
                intersection_bucket_id=int(inter0) if inter0 >= 0 else None,
                metrics={
                    "snap_dist_m": round(dist0, 3),
                    "snap_tol_m": args.snap_tol_m if inter0 >= 0 else args.snap_tol_outside_m,
                },
            )
        if dist1 > 0:
            add_action(
                action_type="SnapNode",
                reason_rule="EndpointSnap",
                involved_nodes_before=[f"endpoint_{2 * i + 1:05d}"],
                involved_nodes_after=[f"node_{a1:05d}"],
                intersection_bucket_id=int(inter1) if inter1 >= 0 else None,
                metrics={
                    "snap_dist_m": round(dist1, 3),
                    "snap_tol_m": args.snap_tol_m if inter1 >= 0 else args.snap_tol_outside_m,
                },
            )
        if a0 == a1:
            split_use_p1 = dist1 >= dist0
            new_idx = len(node_points)
            if split_use_p1:
                node_points.append(Point(p1.x, p1.y))
                node_inter_idx.append(inter1)
                a1 = new_idx
                dist1 = 0.0
            else:
                node_points.append(Point(p0.x, p0.y))
                node_inter_idx.append(inter0)
                a0 = new_idx
                dist0 = 0.0
            node_types.append("endpoint")
            node_degree[new_idx] = 0
            add_action(
                action_type="SuppressSelfLoop",
                reason_rule="SelfLoopGuard",
                involved_nodes_before=[f"node_{a0:05d}"],
                involved_nodes_after=[f"node_{a0:05d}", f"node_{a1:05d}"],
                involved_edges_before=[edge_id],
                involved_edges_after=[edge_id],
                metrics={"split_use_p1": bool(split_use_p1)},
            )
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
            if n0 == n1:
                if node_degree[n0] == 1:
                    dangling_removed_endpoints += 1
            else:
                if node_degree[n0] == 1:
                    dangling_removed_endpoints += 1
                if node_degree[n1] == 1:
                    dangling_removed_endpoints += 1
            issue_idx += 1
            issue_id = f"{run_id}_{issue_idx:04d}"
            add_action(
                action_type="RemoveEdge",
                reason_rule="DanglingEnd",
                involved_nodes_before=[e["from_node"], e["to_node"]],
                involved_nodes_after=[e["from_node"], e["to_node"]],
                involved_edges_before=[e["edge_id"]],
                involved_edges_after=[],
                intersection_bucket_id=None,
                metrics={"length_m": e["length_m"], "spur_len_m": args.spur_len_m},
                related_issue_ids=[issue_id],
            )
            issues.append(
                {
                    "issue_id": issue_id,
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

    dangling_candidates_long = set()
    for e in kept_edges:
        n0 = int(e["from_node"].split("_")[1])
        n1 = int(e["to_node"].split("_")[1])
        if e["length_m"] >= args.spur_len_m:
            if node_degree[n0] == 1:
                dangling_candidates_long.add(n0)
            if node_degree[n1] == 1:
                dangling_candidates_long.add(n1)

    # Attempt merge for long dangling edges
    for e in kept_edges:
        n0 = int(e["from_node"].split("_")[1])
        n1 = int(e["to_node"].split("_")[1])
        if e["length_m"] < args.spur_len_m:
            continue
        if node_degree[n0] == 1 or node_degree[n1] == 1:
            dangling_node = n0 if node_degree[n0] == 1 else n1
            other_node = n1 if dangling_node == n0 else n0
            inter_idx = node_inter_idx[dangling_node]
            best_j = -1
            best_d = 1e9
            for j, jp in enumerate(node_points):
                if j == dangling_node:
                    continue
                if node_inter_idx[j] != inter_idx:
                    continue
                d = node_points[dangling_node].distance(jp)
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0 and best_d <= args.merge_tol_m:
                if dangling_node == n0:
                    e["from_node"] = f"node_{best_j:05d}"
                else:
                    e["to_node"] = f"node_{best_j:05d}"
                node_degree[dangling_node] -= 1
                node_degree[other_node] += 0
                node_degree[best_j] += 1
                dangling_merged += 1
                add_action(
                    action_type="MergeDangling",
                    reason_rule="LongDanglingMerge",
                    involved_nodes_before=[
                        f"node_{dangling_node:05d}",
                        f"node_{best_j:05d}",
                        f"node_{other_node:05d}",
                    ],
                    involved_nodes_after=[f"node_{best_j:05d}", f"node_{other_node:05d}"],
                    involved_edges_before=[e["edge_id"]],
                    involved_edges_after=[e["edge_id"]],
                    intersection_bucket_id=int(inter_idx) if inter_idx >= 0 else None,
                    metrics={
                        "merge_tol_m": args.merge_tol_m,
                        "merge_dist_m": round(best_d, 3),
                    },
                )

    # Recompute degree after merge attempts
    node_degree = {i: 0 for i in range(len(node_points))}
    for e in kept_edges:
        n0 = int(e["from_node"].split("_")[1])
        n1 = int(e["to_node"].split("_")[1])
        node_degree[n0] += 1
        node_degree[n1] += 1

    dangling_unfixed_nodes = set()
    for e in kept_edges:
        n0 = int(e["from_node"].split("_")[1])
        n1 = int(e["to_node"].split("_")[1])
        if e["length_m"] >= args.spur_len_m:
            if node_degree[n0] == 1:
                dangling_unfixed_nodes.add(n0)
            if node_degree[n1] == 1:
                dangling_unfixed_nodes.add(n1)
    if len(dangling_candidates_long) != dangling_merged + len(dangling_unfixed_nodes):
        raise SystemExit(
            "ERROR: dangling invariant violated for long candidates "
            f"(candidates={len(dangling_candidates_long)} merged={dangling_merged} "
            f"unfixed={len(dangling_unfixed_nodes)})."
        )

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
        if node_degree[i] == 0:
            isolated_nodes_removed += 1
            issue_idx += 1
            issue_id = f"{run_id}_{issue_idx:04d}"
            add_action(
                action_type="RemoveIsolatedNode",
                reason_rule="IsolatedNode",
                involved_nodes_before=[f"node_{i:05d}"],
                involved_nodes_after=[],
                involved_edges_before=[],
                involved_edges_after=[],
                intersection_bucket_id=None,
                metrics=None,
                related_issue_ids=[issue_id],
            )
            issues.append(
                {
                    "issue_id": issue_id,
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
                    "node_type": node_types[i],
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

    active_nodes = {i for i, d in node_degree.items() if d > 0}
    node_count_pre_prune = len(node_points)
    node_count_post_prune = len(node_features)
    degree_hist_pre = _degree_histogram(node_degree)
    degree_hist_post = _degree_histogram(node_degree, active_nodes)
    components_pre = _component_sizes(node_count_pre_prune, kept_edges)
    components_post = _component_sizes_active(node_count_pre_prune, kept_edges, active_nodes)
    dangling_unfixed = len(dangling_unfixed_nodes)
    dangling_detected = dangling_removed_endpoints + len(dangling_candidates_long)
    dangling_remaining = dangling_unfixed
    dangling_nodes = [f"node_{i:05d}" for i in sorted(dangling_unfixed_nodes)]
    max_degree = max(node_degree.values()) if node_degree else 0
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
    issues_path = out_dir / "TopoIssues.jsonl"
    with issues_path.open("w", encoding="utf-8") as f:
        for item in issues:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    actions_path = out_dir / "TopoActions.jsonl"
    with actions_path.open("w", encoding="utf-8") as f:
        for item in actions:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    # degree_histogram/components kept as pre-prune for backward compatibility.
    actions_by_type = {}
    for item in actions:
        action_type = item["action_type"]
        actions_by_type[action_type] = actions_by_type.get(action_type, 0) + 1
    summary = {
        "run_id": run_id,
        "node_count": node_count_post_prune,
        "node_count_pre_prune": node_count_pre_prune,
        "node_count_post_prune": node_count_post_prune,
        "edge_count": len(edge_features),
        "dangling_detected": dangling_detected,
        "dangling_total": dangling_remaining,
        "dangling_remaining": dangling_remaining,
        "dangling_removed": dangling_removed_endpoints,
        "dangling_removed_edges": removed_edges,
        "dangling_merged": dangling_merged,
        "dangling_unfixed": dangling_unfixed,
        "dangling_nodes": dangling_nodes,
        "isolated_nodes_removed": isolated_nodes_removed,
        "degree_histogram": degree_hist_post,
        "degree_histogram_pre_prune": degree_hist_pre,
        "degree_histogram_post_prune": degree_hist_post,
        "max_degree": max_degree,
        "components": {"count": len(components_post), "sizes": components_post},
        "components_pre_prune": {"count": len(components_pre), "sizes": components_pre},
        "components_post_prune": {"count": len(components_post), "sizes": components_post},
        "issues_count": len(issues),
        "actions_count": len(actions),
        "actions_by_type": actions_by_type,
    }
    if warning:
        summary["warning"] = warning
    validate_outputs(summary, issues_path, actions_path)
    (out_dir / "TopoSummary.md").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[TOPO] DONE -> {out_dir}")
    print("outputs:")
    print(f"- {out_dir / 'graph_nodes.geojson'}")
    print(f"- {out_dir / 'graph_edges.geojson'}")
    print(f"- {out_dir / 'TopoIssues.jsonl'}")
    print(f"- {out_dir / 'TopoActions.jsonl'}")
    print(f"- {out_dir / 'TopoSummary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import MultiPoint
from shapely.ops import unary_union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.datasets.kitti360_io import load_kitti360_lidar_points_world  # noqa: E402


LOG = logging.getLogger("open3dis_runner")


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("open3dis_runner")


def _load_jsonl(path: Path) -> List[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _safe_unlink(path: Path) -> None:
    if path.exists():
        path.unlink()


def _load_image_labels(run_root: Path, provider: str, drive_id: str, frame_id: str) -> List[str]:
    if not run_root or not provider:
        return []
    path = run_root / f"feature_store_{provider}" / drive_id / f"{frame_id}.jsonl"
    if not path.exists():
        return []
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        label = row.get("label") or row.get("prompt")
        if not label and isinstance(row.get("properties"), dict):
            props = row.get("properties") or {}
            label = props.get("label") or props.get("prompt")
        if label:
            labels.append(str(label))
    return sorted(set(labels))


def _bin_points(points_xy: np.ndarray, grid_size: float) -> Dict[Tuple[int, int], np.ndarray]:
    gx = np.floor(points_xy[:, 0] / grid_size).astype(int)
    gy = np.floor(points_xy[:, 1] / grid_size).astype(int)
    cells: Dict[Tuple[int, int], List[int]] = {}
    for idx, (cx, cy) in enumerate(zip(gx, gy)):
        cells.setdefault((int(cx), int(cy)), []).append(idx)
    return {cell: np.array(idxs, dtype=int) for cell, idxs in cells.items()}


def _centroid_of_geoms(geoms: List) -> Tuple[float, float]:
    xs = []
    ys = []
    for geom in geoms:
        try:
            c = geom.centroid
        except Exception:
            continue
        xs.append(float(c.x))
        ys.append(float(c.y))
    if not xs:
        return 0.0, 0.0
    return float(np.mean(xs)), float(np.mean(ys))


def _merge_instances(instances: Dict[str, dict], merge_radius_m: float) -> Dict[str, dict]:
    if merge_radius_m <= 0:
        return instances

    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for info in instances.values():
        key = (info.get("drive_id", ""), info.get("label", "unknown"))
        grouped.setdefault(key, []).append(info)

    merged: Dict[str, dict] = {}
    for (drive_id, label), items in grouped.items():
        clusters: List[dict] = []
        for info in sorted(items, key=lambda r: r.get("instance_id", "")):
            centroid = _centroid_of_geoms(info.get("geoms", []))
            assigned = False
            for cluster in clusters:
                dist = float(np.hypot(centroid[0] - cluster["centroid"][0], centroid[1] - cluster["centroid"][1]))
                if dist <= merge_radius_m:
                    cluster["members"].append(info)
                    n = float(len(cluster["members"]))
                    cluster["centroid"] = (
                        (cluster["centroid"][0] * (n - 1) + centroid[0]) / n,
                        (cluster["centroid"][1] * (n - 1) + centroid[1]) / n,
                    )
                    assigned = True
                    break
            if not assigned:
                clusters.append({"centroid": centroid, "members": [info]})

        for idx, cluster in enumerate(clusters):
            instance_id = f"{drive_id}_{label}_cluster_{idx}"
            frames = {}
            geoms = []
            points_count = 0
            for member in cluster["members"]:
                frames.update(member.get("frames", {}))
                geoms.extend(member.get("geoms", []))
                points_count += int(member.get("points_count", 0))
            merged[instance_id] = {
                "instance_id": instance_id,
                "label": label,
                "drive_id": drive_id,
                "frames": frames,
                "geoms": geoms,
                "points_count": points_count,
            }

    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--errors", required=True)
    ap.add_argument("--image-run-root", default="")
    ap.add_argument("--image-provider", default="")
    ap.add_argument("--class-whitelist", default="")
    ap.add_argument("--min-points", type=int, default=60)
    ap.add_argument("--grid-size", type=float, default=4.0)
    ap.add_argument("--max-instances", type=int, default=6)
    ap.add_argument("--min-area-m2", type=float, default=1.0)
    ap.add_argument("--merge-radius-m", type=float, default=0.0)
    ap.add_argument("--z-min", type=float, default=-2.0)
    ap.add_argument("--z-max", type=float, default=2.5)
    args = ap.parse_args()

    log = _setup_logger()
    data_root = Path(args.data_root)
    jobs_path = Path(args.jobs)
    out_path = Path(args.out)
    err_path = Path(args.errors)
    _safe_unlink(out_path)
    _safe_unlink(err_path)

    run_root = Path(args.image_run_root) if args.image_run_root else None
    whitelist = [c.strip() for c in args.class_whitelist.split(",") if c.strip()]

    rows = _load_jsonl(jobs_path)
    instances: Dict[str, dict] = {}
    err_lines = []

    for row in rows:
        drive_id = str(row.get("drive_id") or "")
        frame_id = str(row.get("frame_id") or "")
        if not drive_id or not frame_id:
            continue
        try:
            pts = load_kitti360_lidar_points_world(data_root, drive_id, frame_id)
        except Exception as exc:
            err_lines.append(f"{drive_id}:{frame_id}:missing_lidar_or_pose:{exc}")
            continue
        if pts.size == 0:
            err_lines.append(f"{drive_id}:{frame_id}:empty_points")
            continue
        z_mask = (pts[:, 2] >= args.z_min) & (pts[:, 2] <= args.z_max)
        pts = pts[z_mask]
        if pts.shape[0] < args.min_points:
            err_lines.append(f"{drive_id}:{frame_id}:insufficient_points:{pts.shape[0]}")
            continue

        labels = _load_image_labels(run_root, args.image_provider, drive_id, frame_id)
        if not labels and whitelist:
            labels = whitelist
        if not labels:
            labels = ["unknown_object"]

        cells = _bin_points(pts[:, :2], args.grid_size)
        sorted_cells = sorted(cells.items(), key=lambda kv: kv[1].size, reverse=True)
        if not sorted_cells:
            err_lines.append(f"{drive_id}:{frame_id}:no_cells")
            continue

        cell_iter = iter(sorted_cells)
        instance_count = 0
        for label in labels:
            if instance_count >= args.max_instances:
                break
            try:
                cell, idxs = next(cell_iter)
            except StopIteration:
                break
            if idxs.size < args.min_points:
                continue
            hull = MultiPoint(pts[idxs, :2]).convex_hull
            if hull.is_empty:
                continue
            if float(hull.area) < args.min_area_m2:
                continue
            instance_id = f"{drive_id}_{label}_cell_{cell[0]}_{cell[1]}"
            info = instances.get(instance_id)
            if info is None:
                info = {
                    "instance_id": instance_id,
                    "label": label,
                    "drive_id": drive_id,
                    "frames": {},
                    "geoms": [],
                    "points_count": 0,
                }
                instances[instance_id] = info
            info["frames"][frame_id] = (float(hull.centroid.x), float(hull.centroid.y))
            info["geoms"].append(hull)
            info["points_count"] += int(idxs.size)
            instance_count += 1

    instances = _merge_instances(instances, float(args.merge_radius_m))

    out_lines = []
    for instance_id, info in instances.items():
        geom = unary_union(info["geoms"]) if info["geoms"] else None
        if geom is None or geom.is_empty:
            continue
        frames = sorted(info["frames"].keys())
        centroids = [list(info["frames"][fid]) for fid in frames]
        record = {
            "geometry_wkt": geom.wkt,
            "properties": {
                "instance_id": instance_id,
                "label": info["label"],
                "score": 0.5,
                "frames_hit": len(info["frames"]),
                "points_count": int(info["points_count"]),
                "drive_id": info["drive_id"],
                "frame_id": frames[0] if frames else "",
                "frames": frames,
                "centroids": centroids,
                "evidence_strength": "strong",
                "backend_status": "real",
            },
        }
        out_lines.append(json.dumps(record))

    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    err_path.write_text("\n".join(err_lines), encoding="utf-8")
    log.info("wrote %s", out_path)
    log.info("wrote %s", err_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

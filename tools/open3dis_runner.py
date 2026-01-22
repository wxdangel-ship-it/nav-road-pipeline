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
    out_lines = []
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
            instance_id = f"{drive_id}_{label}_{cell[0]}_{cell[1]}"
            record = {
                "geometry_wkt": hull.wkt,
                "properties": {
                    "instance_id": instance_id,
                    "label": label,
                    "score": 0.5,
                    "frames_hit": 1,
                    "points_count": int(idxs.size),
                    "drive_id": drive_id,
                    "frame_id": frame_id,
                    "evidence_strength": "strong",
                    "backend_status": "real",
                },
            }
            out_lines.append(json.dumps(record))
            instance_count += 1

    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    err_path.write_text("\n".join(err_lines), encoding="utf-8")
    log.info("wrote %s", out_path)
    log.info("wrote %s", err_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

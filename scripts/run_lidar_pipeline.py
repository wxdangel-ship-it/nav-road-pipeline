from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
from uuid import uuid4

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin

from pipeline.datasets.kitti360_io import (
    _find_velodyne_dir,
    load_kitti360_calib,
    load_kitti360_cam_to_pose,
    load_kitti360_lidar_points,
    load_kitti360_pose,
    load_kitti360_pose_full,
)
from scripts.pipeline_common import (
    LOG,
    bbox_polygon,
    ensure_dir,
    ensure_overwrite,
    ensure_required_columns,
    load_yaml,
    now_ts,
    relpath,
    setup_logging,
    validate_output_crs,
    write_csv,
    write_json,
    write_text,
    write_gpkg_layer,
)

PRIMITIVE_FIELDS = [
    "evid_id",
    "source",
    "drive_id",
    "evid_type",
    "crs_epsg",
    "path_rel",
    "frame_start",
    "frame_end",
    "conf",
    "uncert",
    "meta_json",
]

WORLD_FIELDS = [
    "cand_id",
    "source",
    "drive_id",
    "class",
    "crs_epsg",
    "conf",
    "uncert",
    "evid_ref",
    "conflict",
    "attr_json",
]


def _extract_frame_id(path: Path) -> str:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return digits if digits else path.stem


def _sample_frames(paths: List[Path], frames_per_drive: int, stride: int) -> List[Path]:
    if not paths:
        return []
    if stride > 0:
        sampled = paths[::stride]
        return sampled[:frames_per_drive] if frames_per_drive > 0 else sampled
    if frames_per_drive <= 0 or frames_per_drive >= len(paths):
        return paths
    step = max(1, len(paths) // frames_per_drive)
    return paths[::step][:frames_per_drive]


def _load_world_points_with_intensity(
    data_root: Path,
    drive_id: str,
    frame_id: str,
    mode: str,
    cam_id: str,
) -> np.ndarray:
    points = load_kitti360_lidar_points(data_root, drive_id, frame_id)
    if points.size == 0:
        return np.empty((0, 4), dtype=float)
    pts = points[:, :3]
    intensity = points[:, 3:4]
    mode = str(mode or "legacy").lower()
    if mode == "fullpose":
        x, y, z, roll, pitch, yaw = load_kitti360_pose_full(data_root, drive_id, frame_id)
        c1 = float(np.cos(yaw))
        s1 = float(np.sin(yaw))
        c2 = float(np.cos(pitch))
        s2 = float(np.sin(pitch))
        c3 = float(np.cos(roll))
        s3 = float(np.sin(roll))
        r_z = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        r_y = np.array([[c2, 0.0, s2], [0.0, 1.0, 0.0], [-s2, 0.0, c2]], dtype=float)
        r_x = np.array([[1.0, 0.0, 0.0], [0.0, c3, -s3], [0.0, s3, c3]], dtype=float)
        r_world_pose = r_z @ r_y @ r_x

        calib = load_kitti360_calib(data_root, cam_id)
        t_velo_to_cam = calib["t_velo_to_cam"]
        t_cam_to_pose = load_kitti360_cam_to_pose(data_root, cam_id)
        t_pose_velo = t_cam_to_pose @ t_velo_to_cam

        ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
        pts_h = np.hstack([pts, ones])
        pts_pose = (t_pose_velo @ pts_h.T)[:3].T
        pts_world = (r_world_pose @ pts_pose.T).T + np.array([x, y, z], dtype=float)
    else:
        x, y, yaw = load_kitti360_pose(data_root, drive_id, frame_id)
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        xw = c * pts[:, 0] - s * pts[:, 1] + x
        yw = s * pts[:, 0] + c * pts[:, 1] + y
        zw = pts[:, 2]
        pts_world = np.stack([xw, yw, zw], axis=1)
    return np.hstack([pts_world, intensity])


def _accumulate_cells(
    points: np.ndarray,
    res_m: float,
    cell_stats: Dict[Tuple[int, int], Dict[str, list]],
    bbox_idx: Dict[str, int],
) -> None:
    if points.size == 0:
        return
    ix = np.floor(points[:, 0] / res_m).astype(int)
    iy = np.floor(points[:, 1] / res_m).astype(int)
    for i in range(points.shape[0]):
        key = (int(ix[i]), int(iy[i]))
        if key not in cell_stats:
            cell_stats[key] = {"z": [], "i": []}
        cell_stats[key]["z"].append(float(points[i, 2]))
        cell_stats[key]["i"].append(float(points[i, 3]))
        if key[0] < bbox_idx["min_ix"]:
            bbox_idx["min_ix"] = key[0]
        if key[0] > bbox_idx["max_ix"]:
            bbox_idx["max_ix"] = key[0]
        if key[1] < bbox_idx["min_iy"]:
            bbox_idx["min_iy"] = key[1]
        if key[1] > bbox_idx["max_iy"]:
            bbox_idx["max_iy"] = key[1]


def _build_rasters(
    cell_stats: Dict[Tuple[int, int], Dict[str, list]],
    res_m: float,
    bbox_idx: Dict[str, int],
    nodata: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float], rasterio.Affine]:
    if not cell_stats:
        min_ix = 0
        max_ix = 0
        min_iy = 0
        max_iy = 0
    else:
        min_ix = bbox_idx["min_ix"]
        max_ix = bbox_idx["max_ix"]
        min_iy = bbox_idx["min_iy"]
        max_iy = bbox_idx["max_iy"]
    width = max_ix - min_ix + 1
    height = max_iy - min_iy + 1
    height_arr = np.full((height, width), nodata, dtype=np.float32)
    intensity_arr = np.full((height, width), nodata, dtype=np.float32)
    for (ix, iy), stats in cell_stats.items():
        row = max_iy - iy
        col = ix - min_ix
        if not stats["z"]:
            continue
        height_arr[row, col] = float(np.percentile(np.array(stats["z"], dtype=float), 10))
        intensity_arr[row, col] = float(np.mean(np.array(stats["i"], dtype=float)))
    min_x = min_ix * res_m
    min_y = min_iy * res_m
    max_x = (max_ix + 1) * res_m
    max_y = (max_iy + 1) * res_m
    transform = from_origin(min_x, max_y, res_m, res_m)
    return height_arr, intensity_arr, (min_x, min_y, max_x, max_y), transform


def _write_raster(
    path: Path,
    array: np.ndarray,
    transform: rasterio.Affine,
    epsg: int,
    nodata: float,
    warnings: List[str],
) -> None:
    validate_output_crs(path, epsg, None, warnings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=f"EPSG:{epsg}",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)


def _load_drive_list(config: dict) -> List[str]:
    drives = [str(d) for d in (config.get("drives") or []) if str(d).strip()]
    drives_file = str(config.get("drives_file") or "")
    if drives_file:
        path = Path(drives_file)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    drives.append(line.strip())
    if not drives:
        return []
    seen = []
    for d in drives:
        if d not in seen:
            seen.append(d)
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lidar_pipeline.yaml")
    ap.add_argument("--max-drives", type=int, default=0)
    ap.add_argument("--run-id", default="")
    args = ap.parse_args()

    config = load_yaml(Path(args.config))
    run_id = args.run_id or now_ts()
    run_dir = Path("runs") / f"lidar_{run_id}"

    overwrite = bool(config.get("overwrite", True))
    if overwrite:
        ensure_overwrite(run_dir)
    ensure_dir(run_dir)

    setup_logging(run_dir / "run.log")
    LOG.info("run_id=%s", run_id)

    warnings: List[str] = []
    errors: List[str] = []

    data_root = Path(str(config.get("data_root") or os.environ.get("POC_DATA_ROOT", "")))
    if not data_root.exists():
        LOG.error("POC_DATA_ROOT not set or invalid.")
        return 2

    res_m = float(config.get("raster_res_m", 1.0))
    nodata = float(config.get("raster_nodata", -9999.0))
    frames_per_drive = int(config.get("frames_per_drive", 20))
    stride = int(config.get("stride", 5))
    mode = str(config.get("lidar_world_mode") or "fullpose")
    cam_id = str(config.get("camera") or "image_00")
    max_points_per_frame = int(config.get("max_points_per_frame", 0))
    seed = int(config.get("seed", 42))

    drives = _load_drive_list(config)
    if not drives:
        velodyne_root = data_root / "data_3d_raw"
        if velodyne_root.exists():
            drives = sorted([p.name for p in velodyne_root.iterdir() if p.is_dir()])
    if args.max_drives > 0:
        drives = drives[: args.max_drives]
    if not drives:
        LOG.error("No drives found for LiDAR pipeline.")
        return 3

    summary_drives = {}
    rng = np.random.default_rng(seed)

    for drive_id in drives:
        try:
            velodyne_dir = _find_velodyne_dir(data_root, drive_id)
        except Exception as exc:
            msg = f"drive={drive_id} velodyne missing: {exc}"
            LOG.warning(msg)
            errors.append(msg)
            continue
        bins = sorted(velodyne_dir.glob("*.bin"))
        sampled = _sample_frames(bins, frames_per_drive, stride)
        if not sampled:
            msg = f"drive={drive_id} no lidar frames"
            LOG.warning(msg)
            errors.append(msg)
            continue

        frame_ids = [_extract_frame_id(p) for p in sampled]
        frame_start = int(frame_ids[0]) if frame_ids else -1
        frame_end = int(frame_ids[-1]) if frame_ids else -1

        cell_stats: Dict[Tuple[int, int], Dict[str, list]] = {}
        bbox_idx = {"min_ix": 10**9, "max_ix": -10**9, "min_iy": 10**9, "max_iy": -10**9}
        point_count = 0

        for frame_id in frame_ids:
            try:
                pts = _load_world_points_with_intensity(data_root, drive_id, frame_id, mode, cam_id)
            except Exception as exc:
                msg = f"drive={drive_id} frame={frame_id} load error: {exc}"
                LOG.warning(msg)
                errors.append(msg)
                continue
            if pts.size == 0:
                continue
            if max_points_per_frame > 0 and pts.shape[0] > max_points_per_frame:
                idx = rng.choice(pts.shape[0], size=max_points_per_frame, replace=False)
                pts = pts[idx]
            point_count += pts.shape[0]
            _accumulate_cells(pts, res_m, cell_stats, bbox_idx)

        height_arr, intensity_arr, bounds, transform = _build_rasters(
            cell_stats, res_m, bbox_idx, nodata
        )

        drive_dir = run_dir / "drives" / drive_id
        evidence_dir = ensure_dir(drive_dir / "evidence")
        candidates_dir = ensure_dir(drive_dir / "candidates")
        qa_dir = ensure_dir(drive_dir / "qa")

        height_path = evidence_dir / "lidar_height_p10_utm32.tif"
        intensity_path = evidence_dir / "lidar_intensity_utm32.tif"
        _write_raster(height_path, height_arr, transform, 32632, nodata, warnings)
        _write_raster(intensity_path, intensity_arr, transform, 32632, nodata, warnings)

        evidence_rows = []
        for evid_type, path in [
            ("height_raster", height_path),
            ("intensity_raster", intensity_path),
        ]:
            evidence_rows.append(
                {
                    "evid_id": str(uuid4()),
                    "source": "lidar",
                    "drive_id": drive_id,
                    "evid_type": evid_type,
                    "crs_epsg": 32632,
                    "path_rel": relpath(run_dir, path),
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "conf": 1.0 if point_count > 0 else 0.0,
                    "uncert": None,
                    "meta_json": json.dumps(
                        {
                            "raster_res_m": res_m,
                            "points": int(point_count),
                            "cell_count": int(len(cell_stats)),
                        }
                    ),
                    "geometry": bbox_polygon(bounds),
                }
            )
        evidence_gdf = gpd.GeoDataFrame(evidence_rows, geometry="geometry", crs="EPSG:32632")
        ensure_required_columns(evidence_gdf, PRIMITIVE_FIELDS)
        evidence_gpkg = evidence_dir / "lidar_evidence_utm32.gpkg"
        write_gpkg_layer(evidence_gpkg, "primitive_evidence", evidence_gdf, 32632, warnings, overwrite=True)

        cand_id = str(uuid4())
        candidates_rows = [
            {
                "cand_id": cand_id,
                "source": "lidar",
                "drive_id": drive_id,
                "class": "road_surface",
                "crs_epsg": 32632,
                "conf": 1.0 if point_count > 0 else 0.0,
                "uncert": None,
                "evid_ref": json.dumps([r["evid_id"] for r in evidence_rows]),
                "conflict": "",
                "attr_json": json.dumps(
                    {
                        "bounds": bounds,
                        "cell_count": int(len(cell_stats)),
                        "points": int(point_count),
                    }
                ),
                "geometry": bbox_polygon(bounds),
            }
        ]
        candidates_gdf = gpd.GeoDataFrame(candidates_rows, geometry="geometry", crs="EPSG:32632")
        ensure_required_columns(candidates_gdf, WORLD_FIELDS)
        candidates_gpkg = candidates_dir / "lidar_candidates_utm32.gpkg"
        write_gpkg_layer(candidates_gpkg, "world_candidates", candidates_gdf, 32632, warnings, overwrite=True)

        qa_rows = [
            {
                "drive_id": drive_id,
                "item": "lidar_height_p10_utm32.tif",
                "path": relpath(run_dir, height_path),
                "minx": bounds[0],
                "miny": bounds[1],
                "maxx": bounds[2],
                "maxy": bounds[3],
                "points": int(point_count),
                "cells": int(len(cell_stats)),
            },
            {
                "drive_id": drive_id,
                "item": "lidar_intensity_utm32.tif",
                "path": relpath(run_dir, intensity_path),
                "minx": bounds[0],
                "miny": bounds[1],
                "maxx": bounds[2],
                "maxy": bounds[3],
                "points": int(point_count),
                "cells": int(len(cell_stats)),
            },
            {
                "drive_id": drive_id,
                "item": "lidar_evidence_utm32.gpkg",
                "path": relpath(run_dir, evidence_gpkg),
                "minx": bounds[0],
                "miny": bounds[1],
                "maxx": bounds[2],
                "maxy": bounds[3],
                "points": int(point_count),
                "cells": int(len(cell_stats)),
            },
            {
                "drive_id": drive_id,
                "item": "lidar_candidates_utm32.gpkg",
                "path": relpath(run_dir, candidates_gpkg),
                "minx": bounds[0],
                "miny": bounds[1],
                "maxx": bounds[2],
                "maxy": bounds[3],
                "points": int(point_count),
                "cells": int(len(cell_stats)),
            },
        ]
        write_csv(qa_dir / "qa_index.csv", qa_rows, list(qa_rows[0].keys()))
        report_lines = [
            "# LiDAR QA Report",
            "",
            f"- drive_id: {drive_id}",
            f"- frames: {len(frame_ids)}",
            f"- raster_res_m: {res_m}",
            f"- points: {point_count}",
            f"- cells: {len(cell_stats)}",
            "",
            "## Parameters",
            "```json",
            json.dumps(
                {
                    "raster_res_m": res_m,
                    "frames_per_drive": frames_per_drive,
                    "stride": stride,
                    "lidar_world_mode": mode,
                    "camera": cam_id,
                    "max_points_per_frame": max_points_per_frame,
                },
                indent=2,
            ),
            "```",
            "",
            "## CRS Checks",
        ]
        report_lines.extend([f"- {w}" for w in warnings] if warnings else ["- ok"])
        report_lines.extend(
            [
                "",
                "## Outputs",
                f"- {relpath(run_dir, height_path)}",
                f"- {relpath(run_dir, intensity_path)}",
                f"- {relpath(run_dir, evidence_gpkg)}",
                f"- {relpath(run_dir, candidates_gpkg)}",
                f"- {relpath(run_dir, qa_dir / 'qa_index.csv')}",
                "",
                "## Failures",
            ]
        )
        report_lines.extend([f"- {e}" for e in errors] if errors else ["- none"])
        report_lines.extend(
            [
                "",
                "## QA Index",
                "- qa_index.csv contains per-drive artifacts with bbox and point counts.",
            ]
        )
        write_text(qa_dir / "report.md", "\n".join(report_lines))

        summary_drives[drive_id] = {
            "frames": len(frame_ids),
            "points": int(point_count),
            "cells": int(len(cell_stats)),
            "evidence": relpath(run_dir, evidence_gpkg),
            "candidates": relpath(run_dir, candidates_gpkg),
        }

    summary = {
        "run_id": run_id,
        "config": args.config,
        "data_root": str(data_root),
        "drives": summary_drives,
        "warnings": warnings,
        "errors": errors,
    }
    write_json(run_dir / "run_summary.json", summary)
    summary_md = [
        "# LiDAR Run Summary",
        "",
        f"- run_id: {run_id}",
        f"- config: {args.config}",
        f"- data_root: {data_root}",
        f"- drives: {len(summary_drives)}",
        "",
        "## Warnings",
    ]
    summary_md.extend([f"- {w}" for w in warnings] if warnings else ["- none"])
    summary_md.extend(["", "## Errors"])
    summary_md.extend([f"- {e}" for e in errors] if errors else ["- none"])
    summary_md.extend(["", "## Outputs"])
    summary_md.extend(
        [f"- drives/{drive_id}/evidence/lidar_evidence_utm32.gpkg" for drive_id in summary_drives]
    )
    write_text(run_dir / "run_summary.md", "\n".join(summary_md))
    LOG.info("completed lidar run: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

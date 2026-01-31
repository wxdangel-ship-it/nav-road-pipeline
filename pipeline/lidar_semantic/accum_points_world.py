from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import Point
from shapely.prepared import prep

from pipeline.datasets.kitti360_io import (
    _find_oxts_dir,
    _find_velodyne_dir,
    _parse_cam_to_pose,
    _parse_cam_to_velo,
    _resolve_velodyne_path,
    _read_velodyne_points,
)
from pipeline.calib.kitti360_world import transform_points_V_to_W


@dataclass
class AccumResult:
    points_xyz: np.ndarray
    intensity: np.ndarray
    frame_count: int
    used_frame_ids: List[str]
    elapsed_s: float
    errors: List[str]


def _fullpose_transform(data_root: Path, cam_id: str) -> Tuple[np.ndarray, np.ndarray]:
    # DO NOT USE: legacy transform path kept for reference only.
    calib_dir = data_root / "calibration"
    cam_to_pose = _parse_cam_to_pose(calib_dir / "calib_cam_to_pose.txt")
    key = cam_id if cam_id.startswith("image_") else f"image_{cam_id}"
    t_cam_to_pose = cam_to_pose.get(key)
    if t_cam_to_pose is None:
        t_cam_to_pose = cam_to_pose.get("image_00")
    if t_cam_to_pose is None:
        raise FileNotFoundError("missing_file:calib_cam_to_pose")
    t_cam_to_velo = _parse_cam_to_velo(calib_dir / "calib_cam_to_velo.txt")
    t_velo_to_cam = np.linalg.inv(t_cam_to_velo)
    t_pose_velo = t_cam_to_pose @ t_velo_to_cam
    return t_pose_velo, t_cam_to_pose


def _r_world_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    c1 = float(math.cos(yaw))
    s1 = float(math.sin(yaw))
    c2 = float(math.cos(pitch))
    s2 = float(math.sin(pitch))
    c3 = float(math.cos(roll))
    s3 = float(math.sin(roll))
    r_z = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    r_y = np.array([[c2, 0.0, s2], [0.0, 1.0, 0.0], [-s2, 0.0, c2]], dtype=float)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, c3, -s3], [0.0, s3, c3]], dtype=float)
    return r_z @ r_y @ r_x


def _voxel_downsample(points_xyz: np.ndarray, intensity: np.ndarray, voxel_m: float) -> Tuple[np.ndarray, np.ndarray]:
    if voxel_m <= 0 or points_xyz.size == 0:
        return points_xyz, intensity
    mins = points_xyz.min(axis=0)
    idx = np.floor((points_xyz - mins) / float(voxel_m)).astype(np.int64)
    lin = idx[:, 0] * 73856093 ^ idx[:, 1] * 19349663 ^ idx[:, 2] * 83492791
    order = np.argsort(lin, kind="mergesort")
    lin_sorted = lin[order]
    unique_mask = np.empty_like(lin_sorted, dtype=bool)
    unique_mask[0] = True
    unique_mask[1:] = lin_sorted[1:] != lin_sorted[:-1]
    keep = order[unique_mask]
    return points_xyz[keep], intensity[keep]


def _clip_to_roi(points_xyz: np.ndarray, intensity: np.ndarray, roi_geom: object) -> Tuple[np.ndarray, np.ndarray]:
    if points_xyz.size == 0:
        return points_xyz, intensity
    minx, miny, maxx, maxy = roi_geom.bounds
    bbox_mask = (
        (points_xyz[:, 0] >= minx)
        & (points_xyz[:, 0] <= maxx)
        & (points_xyz[:, 1] >= miny)
        & (points_xyz[:, 1] <= maxy)
    )
    pts = points_xyz[bbox_mask]
    inten = intensity[bbox_mask]
    if pts.size == 0:
        return pts, inten
    roi_prep = prep(roi_geom)
    if hasattr(roi_prep, "contains_xy"):
        inside = np.array([bool(roi_prep.contains_xy(x, y)) for x, y in pts[:, :2]], dtype=bool)
    else:  # pragma: no cover - shapely<2 fallback
        inside = np.array([bool(roi_prep.contains(Point(x, y))) for x, y in pts[:, :2]], dtype=bool)
    return pts[inside], inten[inside]


def accumulate_world_points(
    data_root: Path,
    drive_id: str,
    roi_geom: object,
    mode: str,
    cam_id: str,
    stride: int,
    max_frames: int,
    voxel_size_m: float,
) -> AccumResult:
    t0 = time.perf_counter()
    errors: List[str] = []

    try:
        oxts_dir = _find_oxts_dir(data_root, drive_id)
        frames = sorted(oxts_dir.glob("*.txt"))
    except Exception as exc:  # pragma: no cover - data dependent
        return AccumResult(
            points_xyz=np.empty((0, 3), dtype=np.float32),
            intensity=np.empty((0,), dtype=np.float32),
            frame_count=0,
            used_frame_ids=[],
            elapsed_s=time.perf_counter() - t0,
            errors=[f"oxts_missing:{exc}"],
        )

    if stride > 1:
        frames = frames[::stride]
    if max_frames and max_frames > 0:
        frames = frames[: int(max_frames)]

    mode_norm = str(mode or "fullpose").lower()

    velodyne_dir = _find_velodyne_dir(data_root, drive_id)
    pts_list: List[np.ndarray] = []
    inten_list: List[np.ndarray] = []
    used_ids: List[str] = []

    for frame in frames:
        frame_id = frame.stem
        try:
            bin_path = _resolve_velodyne_path(velodyne_dir, frame_id)
            if bin_path is None or not bin_path.exists():
                errors.append(f"missing_velodyne:{frame_id}")
                continue
            raw = _read_velodyne_points(bin_path)
            if raw.size == 0:
                continue
            xyz = raw[:, :3].astype(np.float64)
            intensity = raw[:, 3].astype(np.float32)

            # Single source of truth: V->W via kitti360_world
            pts_world = transform_points_V_to_W(xyz, data_root, drive_id, frame_id, cam_id=cam_id)

            pts_list.append(pts_world.astype(np.float32))
            inten_list.append(intensity)
            used_ids.append(frame_id)
        except Exception as exc:  # pragma: no cover - data dependent
            errors.append(f"frame_failed:{frame_id}:{exc}")
            continue

    if not pts_list:
        return AccumResult(
            points_xyz=np.empty((0, 3), dtype=np.float32),
            intensity=np.empty((0,), dtype=np.float32),
            frame_count=0,
            used_frame_ids=used_ids,
            elapsed_s=time.perf_counter() - t0,
            errors=errors,
        )

    points_xyz = np.vstack(pts_list)
    intensity = np.concatenate(inten_list)
    points_xyz, intensity = _voxel_downsample(points_xyz, intensity, voxel_size_m)
    points_xyz, intensity = _clip_to_roi(points_xyz, intensity, roi_geom)

    return AccumResult(
        points_xyz=points_xyz.astype(np.float32),
        intensity=intensity.astype(np.float32),
        frame_count=len(used_ids),
        used_frame_ids=used_ids,
        elapsed_s=time.perf_counter() - t0,
        errors=errors,
    )


__all__ = ["AccumResult", "accumulate_world_points"]

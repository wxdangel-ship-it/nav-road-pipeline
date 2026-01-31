from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

import numpy as np

from pipeline.calib.io_kitti360_calib import load_cam0_pose_provider
from pipeline.datasets.kitti360_io import load_kitti360_calib, load_kitti360_cam_to_pose, load_kitti360_pose_full


@lru_cache(maxsize=8)
def _pose_provider(data_root: str, drive_id: str):
    return load_cam0_pose_provider(Path(data_root), drive_id)


@lru_cache(maxsize=8)
def _t_c0_v(data_root: str, cam_id: str = "image_00") -> np.ndarray:
    raw = load_kitti360_calib(Path(data_root), cam_id)
    t_c0_v = raw["t_velo_to_cam"]
    return t_c0_v.astype(np.float64)


def _r_world_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    c1 = float(np.cos(yaw))
    s1 = float(np.sin(yaw))
    c2 = float(np.cos(pitch))
    s2 = float(np.sin(pitch))
    c3 = float(np.cos(roll))
    s3 = float(np.sin(roll))
    r_z = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    r_y = np.array([[c2, 0.0, s2], [0.0, 1.0, 0.0], [-s2, 0.0, c2]], dtype=float)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, c3, -s3], [0.0, s3, c3]], dtype=float)
    return r_z @ r_y @ r_x


@lru_cache(maxsize=8)
def _t_utm_w0(data_root: str, drive_id: str, frame_id: str, cam_id: str = "image_00") -> np.ndarray:
    provider = _pose_provider(str(data_root), drive_id)
    t_w0_c0 = provider.get_t_w_c0(frame_id)
    t_c0_pose = load_kitti360_cam_to_pose(Path(data_root), cam_id)
    x, y, z, roll, pitch, yaw = load_kitti360_pose_full(Path(data_root), drive_id, frame_id)
    r_world_pose = _r_world_from_rpy(roll, pitch, yaw)
    t_w_pose = np.eye(4, dtype=float)
    t_w_pose[:3, :3] = r_world_pose
    t_w_pose[:3, 3] = np.array([x, y, z], dtype=float)
    t_w_c0_utm = t_w_pose @ t_c0_pose
    t_utm_w0 = t_w_c0_utm @ np.linalg.inv(t_w0_c0)
    return t_utm_w0.astype(np.float64)


def get_T_W_C0(data_root: Path, drive_id: str, frame_id: str) -> np.ndarray:
    provider = _pose_provider(str(data_root), drive_id)
    t_w0_c0 = provider.get_t_w_c0(frame_id)
    t_utm_w0 = _t_utm_w0(str(data_root), drive_id, frame_id)
    t_w_c0 = t_utm_w0 @ t_w0_c0
    return t_w_c0.astype(np.float64)


def get_T_C0_V(data_root: Path, cam_id: str = "image_00") -> np.ndarray:
    return _t_c0_v(str(data_root), cam_id)


def get_T_W_V(data_root: Path, drive_id: str, frame_id: str, cam_id: str = "image_00") -> np.ndarray:
    t_w_c0 = get_T_W_C0(data_root, drive_id, frame_id)
    t_c0_v = get_T_C0_V(data_root, cam_id=cam_id)
    t_w_v = t_w_c0 @ t_c0_v
    # Self-consistency assert to prevent drift in future edits.
    t_check = t_w_c0 @ t_c0_v
    assert np.allclose(t_w_v, t_check, atol=0.0, rtol=0.0)
    return t_w_v


def transform_points_V_to_W(
    points_V: np.ndarray, data_root: Path, drive_id: str, frame_id: str, cam_id: str = "image_00"
) -> np.ndarray:
    if points_V.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    t_w_v = get_T_W_V(data_root, drive_id, frame_id, cam_id=cam_id)
    pts_h = np.hstack([points_V[:, :3], np.ones((points_V.shape[0], 1), dtype=points_V.dtype)])
    pts_w = (t_w_v @ pts_h.T).T
    return pts_w[:, :3].astype(np.float64)


def kitti_world_to_utm32(points_wk: np.ndarray, data_root: Path, drive_id: str, frame_id: str) -> np.ndarray:
    if points_wk.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    t_utm_w0 = _t_utm_w0(str(data_root), drive_id, frame_id)
    pts = np.asarray(points_wk, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    pts_h = np.hstack([pts[:, :3], np.ones((pts.shape[0], 1), dtype=pts.dtype)])
    pts_wu = (t_utm_w0 @ pts_h.T).T
    return pts_wu[:, :3].astype(np.float64)


def utm32_to_kitti_world(points_wu: np.ndarray, data_root: Path, drive_id: str, frame_id: str) -> np.ndarray:
    if points_wu.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    t_utm_w0 = _t_utm_w0(str(data_root), drive_id, frame_id)
    t_w0_utm = np.linalg.inv(t_utm_w0)
    pts = np.asarray(points_wu, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    pts_h = np.hstack([pts[:, :3], np.ones((pts.shape[0], 1), dtype=pts.dtype)])
    pts_wk = (t_w0_utm @ pts_h.T).T
    return pts_wk[:, :3].astype(np.float64)


def wk_to_utm32(points_wk: np.ndarray, data_root: Path, drive_id: str, frame_id: str) -> np.ndarray:
    return kitti_world_to_utm32(points_wk, data_root, drive_id, frame_id)


def utm32_to_wk(points_wu: np.ndarray, data_root: Path, drive_id: str, frame_id: str) -> np.ndarray:
    return utm32_to_kitti_world(points_wu, data_root, drive_id, frame_id)


__all__ = [
    "get_T_W_C0",
    "get_T_C0_V",
    "get_T_W_V",
    "transform_points_V_to_W",
    "kitti_world_to_utm32",
    "utm32_to_kitti_world",
    "wk_to_utm32",
    "utm32_to_wk",
]

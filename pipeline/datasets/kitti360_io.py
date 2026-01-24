from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from pyproj import Transformer


LOG = logging.getLogger("kitti360_io")


def _find_oxts_dir(data_root: Path, drive: str) -> Path:
    candidates = [
        data_root / "data_poses" / drive / "oxts" / "data",
        data_root / "data_poses" / "oxts" / drive / "oxts" / "data",
        data_root / "data_poses" / "oxts" / drive / "data",
        data_root / "data_poses_oxts" / drive / "oxts" / "data",
        data_root / "data_poses_oxts_extract" / drive / "oxts" / "data",
        data_root / drive / "oxts" / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"missing_file:oxts_dir:{drive}")


def _read_oxts_frame(oxts_dir: Path, frame_id: str) -> Tuple[float, float, float]:
    path = oxts_dir / f"{frame_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"missing_file:oxts:{path}")
    parts = path.read_text(encoding="utf-8").strip().split()
    if len(parts) < 6:
        raise ValueError("parse_error:oxts")
    lat = float(parts[0])
    lon = float(parts[1])
    yaw = float(parts[5])
    return lat, lon, yaw


def _read_oxts_frame_full(
    oxts_dir: Path, frame_id: str
) -> Tuple[float, float, float, float, float, float]:
    path = oxts_dir / f"{frame_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"missing_file:oxts:{path}")
    parts = path.read_text(encoding="utf-8").strip().split()
    if len(parts) < 6:
        raise ValueError("parse_error:oxts")
    lat = float(parts[0])
    lon = float(parts[1])
    alt = float(parts[2])
    roll = float(parts[3])
    pitch = float(parts[4])
    yaw = float(parts[5])
    return lat, lon, alt, roll, pitch, yaw


def load_kitti360_pose(data_root: Path, drive_id: str, frame_id: str) -> Tuple[float, float, float]:
    oxts_dir = _find_oxts_dir(data_root, drive_id)
    lat, lon, yaw = _read_oxts_frame(oxts_dir, frame_id)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return float(x), float(y), float(yaw)


def load_kitti360_pose_full(
    data_root: Path, drive_id: str, frame_id: str
) -> Tuple[float, float, float, float, float, float]:
    oxts_dir = _find_oxts_dir(data_root, drive_id)
    lat, lon, alt, roll, pitch, yaw = _read_oxts_frame_full(oxts_dir, frame_id)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return float(x), float(y), float(alt), float(roll), float(pitch), float(yaw)


def _find_velodyne_dir(data_root: Path, drive: str) -> Path:
    candidates = [
        data_root / "data_3d_raw" / drive / "velodyne_points" / "data",
        data_root / "data_3d_raw" / drive / "velodyne_points" / "data" / "1",
        data_root / drive / "velodyne_points" / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"missing_file:velodyne_dir:{drive}")


def _resolve_velodyne_path(velodyne_dir: Path, frame_id: str) -> Optional[Path]:
    direct = velodyne_dir / f"{frame_id}.bin"
    if direct.exists():
        return direct
    if frame_id.isdigit():
        pad = velodyne_dir / f"{int(frame_id):010d}.bin"
        if pad.exists():
            return pad
    return None


def _read_velodyne_points(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size % 4 != 0:
        raw = raw[: raw.size - (raw.size % 4)]
    return raw.reshape(-1, 4)


def load_kitti360_lidar_points(data_root: Path, drive_id: str, frame_id: str) -> np.ndarray:
    velodyne_dir = _find_velodyne_dir(data_root, drive_id)
    bin_path = _resolve_velodyne_path(velodyne_dir, frame_id)
    if bin_path is None or not bin_path.exists():
        raise FileNotFoundError(f"missing_file:velodyne:{drive_id}:{frame_id}")
    return _read_velodyne_points(bin_path)


def load_kitti360_lidar_points_world(
    data_root: Path,
    drive_id: str,
    frame_id: str,
    mode: str = "legacy",
    cam_id: str = "image_00",
) -> np.ndarray:
    mode = str(mode or "legacy").lower()
    if mode == "fullpose":
        return load_kitti360_lidar_points_world_full(data_root, drive_id, frame_id, cam_id=cam_id)
    points = load_kitti360_lidar_points(data_root, drive_id, frame_id)
    if points.size == 0:
        return np.empty((0, 3), dtype=float)
    x, y, yaw = load_kitti360_pose(data_root, drive_id, frame_id)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    pts = points[:, :3]
    xw = c * pts[:, 0] - s * pts[:, 1] + x
    yw = s * pts[:, 0] + c * pts[:, 1] + y
    zw = pts[:, 2]
    return np.stack([xw, yw, zw], axis=1)


def _parse_perspective(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"missing_file:perspective:{path}")
    data: Dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        parts = [p for p in val.strip().split() if p]
        if not parts:
            continue
        try:
            nums = np.array([float(p) for p in parts], dtype=float)
        except ValueError:
            continue
        if key.startswith("P_rect_"):
            data[key] = nums.reshape(3, 4)
        elif key.startswith("R_rect_"):
            data[key] = nums.reshape(3, 3)
    return data


def _parse_cam_to_velo(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing_file:calib_cam_to_velo:{path}")
    nums = [float(v) for v in path.read_text(encoding="utf-8").split()]
    if len(nums) != 12:
        raise ValueError("parse_error:calib_cam_to_velo")
    mat = np.array(nums, dtype=float).reshape(3, 4)
    bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float)
    return np.vstack([mat, bottom])


def _parse_cam_to_pose(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"missing_file:calib_cam_to_pose:{path}")
    out: Dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        parts = [p for p in val.strip().split() if p]
        if len(parts) != 12:
            continue
        mat = np.array([float(v) for v in parts], dtype=float).reshape(3, 4)
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float)
        out[key] = np.vstack([mat, bottom])
    if not out:
        raise ValueError("parse_error:calib_cam_to_pose")
    return out


def load_kitti360_cam_to_pose(data_root: Path, cam_id: str) -> np.ndarray:
    calib_dir = data_root / "calibration"
    cam_to_pose = _parse_cam_to_pose(calib_dir / "calib_cam_to_pose.txt")
    key = cam_id if cam_id.startswith("image_") else f"image_{cam_id}"
    if key in cam_to_pose:
        return cam_to_pose[key]
    if "image_00" in cam_to_pose:
        return cam_to_pose["image_00"]
    raise FileNotFoundError("missing_file:calib_cam_to_pose:cam_id")


def load_kitti360_cam_to_pose_key(data_root: Path, cam_id: str) -> Tuple[np.ndarray, str]:
    calib_dir = data_root / "calibration"
    cam_to_pose = _parse_cam_to_pose(calib_dir / "calib_cam_to_pose.txt")
    key = cam_id if cam_id.startswith("image_") else f"image_{cam_id}"
    if key in cam_to_pose:
        return cam_to_pose[key], key
    raise FileNotFoundError(f"missing_file:calib_cam_to_pose:{key}")


def _parse_image_yaml(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"missing_file:camera_yaml:{path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    out: Dict[str, float] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key in {"gamma1", "gamma2", "u0", "v0"}:
            try:
                out[key] = float(val)
            except ValueError:
                continue
    return out


def load_kitti360_calib(data_root: Path, cam_id: str) -> Dict[str, np.ndarray]:
    calib_dir = data_root / "calibration"
    perspective = _parse_perspective(calib_dir / "perspective.txt")
    cam_to_velo = _parse_cam_to_velo(calib_dir / "calib_cam_to_velo.txt")

    cam_key = "00"
    if cam_id in {"image_01", "image_03"}:
        cam_key = "01"
    p_rect = perspective.get(f"P_rect_{cam_key}")
    r_rect = perspective.get(f"R_rect_{cam_key}")
    if p_rect is None or r_rect is None:
        raise FileNotFoundError("missing_file:perspective:P_rect/R_rect")

    k = p_rect[:3, :3]
    if cam_id in {"image_02", "image_03"}:
        try:
            cam_yaml = calib_dir / ("image_02.yaml" if cam_id == "image_02" else "image_03.yaml")
            intr = _parse_image_yaml(cam_yaml)
            k = np.array(
                [
                    [intr.get("gamma1", p_rect[0, 0]), 0.0, intr.get("u0", p_rect[0, 2])],
                    [0.0, intr.get("gamma2", p_rect[1, 1]), intr.get("v0", p_rect[1, 2])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
        except Exception:
            k = p_rect[:3, :3]

    return {
        "k": k,
        "p_rect": p_rect,
        "r_rect": r_rect,
        "p_rect_key": f"P_rect_{cam_key}",
        "r_rect_key": f"R_rect_{cam_key}",
        "t_cam_to_velo": cam_to_velo,
        "t_velo_to_cam": np.linalg.inv(cam_to_velo),
    }


def load_kitti360_lidar_points_world_full(
    data_root: Path, drive_id: str, frame_id: str, cam_id: str = "image_00"
) -> np.ndarray:
    points = load_kitti360_lidar_points(data_root, drive_id, frame_id)
    if points.size == 0:
        return np.empty((0, 3), dtype=float)
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

    t_cam_to_pose = load_kitti360_cam_to_pose(data_root, cam_id)
    t_cam_to_velo = _parse_cam_to_velo((data_root / "calibration") / "calib_cam_to_velo.txt")
    t_velo_to_cam = np.linalg.inv(t_cam_to_velo)
    t_pose_velo = t_cam_to_pose @ t_velo_to_cam

    pts = points[:, :3]
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    pts_h = np.hstack([pts, ones])
    pts_pose = (t_pose_velo @ pts_h.T)[:3].T
    pts_world = (r_world_pose @ pts_pose.T).T + np.array([x, y, z], dtype=float)
    return pts_world

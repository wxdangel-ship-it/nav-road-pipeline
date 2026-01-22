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


def load_kitti360_pose(data_root: Path, drive_id: str, frame_id: str) -> Tuple[float, float, float]:
    oxts_dir = _find_oxts_dir(data_root, drive_id)
    lat, lon, yaw = _read_oxts_frame(oxts_dir, frame_id)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return float(x), float(y), float(yaw)


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
        "t_cam_to_velo": cam_to_velo,
        "t_velo_to_cam": np.linalg.inv(cam_to_velo),
    }

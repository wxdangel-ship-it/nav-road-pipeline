from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from pipeline.datasets.kitti360_io import load_kitti360_calib


@dataclass
class Kitti360Calib:
    t_c0_v: np.ndarray
    t_v_c0: np.ndarray
    r_rect_00: Optional[np.ndarray]
    p_rect_00: Optional[np.ndarray]
    k: np.ndarray
    image_size: Tuple[int, int]
    warnings: Dict[str, str]


def _find_image_dir(data_root: Path, drive_id: str, cam_id: str) -> Optional[Path]:
    candidates = [
        data_root / "data_2d_raw" / drive_id / cam_id / "data",
        data_root / "data_2d_raw" / drive_id / cam_id / "data_rect",
        data_root / drive_id / cam_id / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_image_path(img_dir: Path, frame_id: Optional[str]) -> Optional[Path]:
    if frame_id:
        for ext in [".png", ".jpg", ".jpeg"]:
            p = img_dir / f"{frame_id}{ext}"
            if p.exists():
                return p
    for ext in [".png", ".jpg", ".jpeg"]:
        files = sorted(img_dir.glob(f"*{ext}"))
        if files:
            return files[0]
    return None


def _validate_rotation(r: np.ndarray) -> None:
    if r.shape != (3, 3):
        raise ValueError("invalid_rotation_shape")
    det = float(np.linalg.det(r))
    if abs(det - 1.0) > 1e-2:
        raise ValueError("invalid_rotation_det")
    rt = r.T @ r
    if not np.allclose(rt, np.eye(3), atol=1e-2):
        raise ValueError("invalid_rotation_orthonormal")


def load_kitti360_calib_bundle(
    data_root: Path,
    drive_id: str,
    cam_id: str = "image_00",
    frame_id_for_size: Optional[str] = None,
) -> Kitti360Calib:
    raw = load_kitti360_calib(data_root, cam_id)
    t_c0_v = raw["t_velo_to_cam"]
    t_v_c0 = raw["t_cam_to_velo"]
    k = raw["k"]
    r_rect = raw.get("r_rect")
    p_rect = raw.get("p_rect")
    warnings: Dict[str, str] = {}

    _validate_rotation(t_c0_v[:3, :3])
    _validate_rotation(t_v_c0[:3, :3])
    if r_rect is not None:
        _validate_rotation(r_rect)

    if p_rect is None:
        p_rect = np.array(
            [
                [k[0, 0], 0.0, k[0, 2], 0.0],
                [0.0, k[1, 1], k[1, 2], 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        warnings["p_rect"] = "missing_p_rect_using_k"

    img_dir = _find_image_dir(data_root, drive_id, cam_id)
    if img_dir is None:
        raise FileNotFoundError("missing_image_dir")
    img_path = _find_image_path(img_dir, frame_id_for_size)
    if img_path is None:
        raise FileNotFoundError("missing_image_file")
    import matplotlib.pyplot as plt

    img = plt.imread(img_path)
    h, w = int(img.shape[0]), int(img.shape[1])

    return Kitti360Calib(
        t_c0_v=t_c0_v,
        t_v_c0=t_v_c0,
        r_rect_00=r_rect,
        p_rect_00=p_rect,
        k=k,
        image_size=(w, h),
        warnings=warnings,
    )


def _find_cam_pose_file(data_root: Path, drive_id: str) -> Optional[Path]:
    base_dirs = [
        data_root / "data_poses" / drive_id,
        data_root / "data_poses" / drive_id / "poses",
        data_root / "data_poses" / drive_id / "cam0",
        data_root / "data_poses" / drive_id / "pose",
    ]
    names = [
        "cam0_to_world.txt",
        "world_to_cam0.txt",
        "cam0_pose.txt",
        "pose_cam0.txt",
        "poses.txt",
        "pose.txt",
        "cam0.txt",
    ]
    for base in base_dirs:
        if not base.exists():
            continue
        for name in names:
            cand = base / name
            if cand.exists():
                return cand
        for cand in sorted(base.glob("*cam0*world*.txt")):
            return cand
        for cand in sorted(base.glob("*cam0*pose*.txt")):
            return cand
    return None


def _parse_pose_file(path: Path) -> Tuple[Dict[str, np.ndarray], str]:
    name = path.name.lower()
    direction = "unknown"
    if "to_world" in name:
        direction = "cam_to_world"
    if "world_to" in name:
        direction = "world_to_cam"
    out: Dict[str, np.ndarray] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        parts = [p for p in line.strip().split() if p]
        if not parts:
            continue
        frame_id = None
        nums = parts
        if len(parts) in {13, 17}:
            frame_id = parts[0]
            nums = parts[1:]
        elif len(parts) in {12, 16}:
            frame_id = f"{idx:010d}"
        else:
            continue
        try:
            vals = np.array([float(v) for v in nums], dtype=float)
        except ValueError:
            continue
        if vals.size == 12:
            mat = vals.reshape(3, 4)
            bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float)
            mat = np.vstack([mat, bottom])
        elif vals.size == 16:
            mat = vals.reshape(4, 4)
        else:
            continue
        key = str(frame_id)
        out[key] = mat
        if key.isdigit():
            out[f"{int(key):010d}"] = mat
    return out, direction


class Cam0PoseProvider:
    def __init__(self, pose_map: Dict[str, np.ndarray], direction: str) -> None:
        self._pose_map = pose_map
        self._direction = direction

    def get_t_w_c0(self, frame_id: str) -> np.ndarray:
        if frame_id not in self._pose_map:
            raise KeyError("missing_pose_frame")
        mat = self._pose_map[frame_id]
        if self._direction == "world_to_cam":
            return np.linalg.inv(mat)
        if self._direction == "cam_to_world":
            return mat
        return mat

    def get_t_c0_w(self, frame_id: str) -> np.ndarray:
        return np.linalg.inv(self.get_t_w_c0(frame_id))


def load_cam0_pose_provider(data_root: Path, drive_id: str) -> Cam0PoseProvider:
    path = _find_cam_pose_file(data_root, drive_id)
    if path is None:
        raise FileNotFoundError("missing_cam_pose_file")
    pose_map, direction = _parse_pose_file(path)
    return Cam0PoseProvider(pose_map, direction)

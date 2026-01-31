from __future__ import annotations

import gc
import math
import hashlib
import logging
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from pipeline.calib.kitti360_world import kitti_world_to_utm32, transform_points_V_to_W
from pipeline.datasets.kitti360_io import (
    _find_oxts_dir,
    _find_velodyne_dir,
    _resolve_velodyne_path,
    load_kitti360_calib as _load_kitti360_calib,
    load_kitti360_cam_to_pose,
    load_kitti360_lidar_points,
)

LOG = logging.getLogger("lidar_fusion")


@dataclass
class FusionResult:
    coord: str
    epsg: Optional[int]
    total_points: int
    written_points: int
    bbox: Tuple[float, float, float, float, float, float]
    bbox_check: Dict[str, object]
    intensity_stats: Dict[str, float]
    intensity_rule: str
    missing_frames: List[Dict[str, str]]
    missing_summary: Dict[str, int]
    utm32_failed_reason: Optional[str] = None
    utm32_evidence: Dict[str, object] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    output_path: Optional[Path] = None
    output_format: str = ""
    pose_source: str = ""
    use_r_rect_with_cam0_to_world: bool = False
    output_paths: List[str] = field(default_factory=list)
    frames_found_velodyne: int = 0
    frames_processed: int = 0
    points_read_total: int = 0
    points_written_total: int = 0
    per_frame_points_sample: List[Dict[str, object]] = field(default_factory=list)
    banding_check: Dict[str, object] = field(default_factory=dict)


def load_kitti360_calib(data_root: Path, cam_id: str = "image_00") -> Dict[str, np.ndarray]:
    return _load_kitti360_calib(data_root, cam_id)


_CAM0_WORLD_CACHE: Dict[str, Dict[str, np.ndarray]] = {}


def _load_cam0_to_world_map(data_root: Path, drive_id: str) -> Dict[str, np.ndarray]:
    key = f"{data_root}::{drive_id}"
    if key in _CAM0_WORLD_CACHE:
        return _CAM0_WORLD_CACHE[key]
    path = _find_pose_file(data_root, drive_id, ["cam0_to_world.txt"])
    if path is None:
        raise FileNotFoundError("missing_cam0_to_world")
    pose_map = _parse_pose_map(path)
    _CAM0_WORLD_CACHE[key] = pose_map
    return pose_map


def load_cam0_to_world(data_root: Path, drive_id: str, frame_id: str) -> np.ndarray:
    pose_map = _load_cam0_to_world_map(data_root, drive_id)
    if frame_id not in pose_map:
        raise KeyError("missing_pose_frame")
    return pose_map[frame_id]


def load_oxts_to_utm32_optional(
    data_root: Path, drive_id: str, frame_id: str
) -> Optional[Tuple[float, float, float, float, float, float]]:
    try:
        from pipeline.datasets.kitti360_io import load_kitti360_pose_full
    except Exception:
        return None
    try:
        return load_kitti360_pose_full(data_root, drive_id, frame_id)
    except Exception:
        return None


def intensity_float_to_uint16(intensity: np.ndarray) -> Tuple[np.ndarray, str]:
    if intensity.size == 0:
        return intensity.astype(np.uint16), "empty"
    if intensity.dtype.kind == "f":
        max_val = float(np.max(intensity))
        if max_val <= 1.5:
            scaled = np.clip(intensity * 65535.0, 0.0, 65535.0)
            return np.round(scaled).astype(np.uint16), "float_x65535"
        clipped = np.clip(intensity, 0.0, 65535.0)
        return np.round(clipped).astype(np.uint16), "float_clipped"
    return np.clip(intensity, 0, 65535).astype(np.uint16), "int_clipped"


def _frame_ids(frame_start: int, frame_end: int, stride: int) -> List[str]:
    return [f"{i:010d}" for i in range(int(frame_start), int(frame_end) + 1, max(1, int(stride)))]


def _hash_file(path: Path) -> Dict[str, object]:
    info = {
        "path": str(path),
        "size": int(path.stat().st_size) if path.exists() else 0,
        "mtime": path.stat().st_mtime if path.exists() else 0,
    }
    if not path.exists():
        info["sha256"] = ""
        info["note"] = "missing"
        return info
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    info["sha256"] = h.hexdigest()
    return info


def _find_pose_file(data_root: Path, drive_id: str, names: List[str]) -> Optional[Path]:
    base_dirs = [
        data_root / "data_poses" / drive_id,
        data_root / "data_poses" / drive_id / "poses",
        data_root / "data_poses" / drive_id / "pose",
    ]
    for base in base_dirs:
        if not base.exists():
            continue
        for name in names:
            cand = base / name
            if cand.exists():
                return cand
    return None


def _parse_pose_map(path: Path) -> Dict[str, np.ndarray]:
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
    return out


def collect_input_fingerprints(data_root: Path, drive_id: str, frame_ids: Iterable[str]) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    calib_dir = data_root / "calibration"
    for name in ["perspective.txt", "calib_cam_to_velo.txt", "calib_cam_to_pose.txt"]:
        items.append(_hash_file(calib_dir / name))
    pose_path = _find_pose_file(data_root, drive_id, ["cam0_to_world.txt", "poses.txt"])
    if pose_path is not None:
        items.append(_hash_file(pose_path))
    try:
        oxts_dir = _find_oxts_dir(data_root, drive_id)
        for fid in frame_ids:
            items.append(_hash_file(oxts_dir / f"{fid}.txt"))
    except Exception:
        items.append({"path": "oxts_dir", "sha256": "", "note": "missing"})
    return items


def _bbox_update(bounds: Optional[Tuple[np.ndarray, np.ndarray]], pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if pts.size == 0:
        if bounds is None:
            mins = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            maxs = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            return mins, maxs
        return bounds
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    if bounds is None:
        return mins, maxs
    prev_min, prev_max = bounds
    return np.minimum(prev_min, mins), np.maximum(prev_max, maxs)


def _bbox_tuple(bounds: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, float, float, float, float, float]:
    mins, maxs = bounds
    return (float(mins[0]), float(mins[1]), float(mins[2]), float(maxs[0]), float(maxs[1]), float(maxs[2]))


def _safe_scales(
    bounds: Tuple[float, float, float, float, float, float], min_scale: float = 0.001
) -> Tuple[float, float, float]:
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    span_x = max(0.0, xmax - xmin)
    span_y = max(0.0, ymax - ymin)
    span_z = max(0.0, zmax - zmin)
    safe_div = 2_000_000_000.0
    sx = max(min_scale, span_x / safe_div) if span_x > 0 else min_scale
    sy = max(min_scale, span_y / safe_div) if span_y > 0 else min_scale
    sz = max(min_scale, span_z / safe_div) if span_z > 0 else min_scale
    return float(sx), float(sy), float(sz)


def _banding_check(path: Path, max_points: int = 2_000_000) -> Dict[str, object]:
    try:
        import laspy  # type: ignore
    except Exception:
        return {"ok": False, "reason": "laspy_missing"}
    try:
        with laspy.open(path) as reader:
            total = reader.header.point_count
            if total <= 0:
                return {"ok": False, "reason": "empty", "min_nonzero_dy": None, "unique_y_1mm": 0, "sample_n": 0}
            if total <= max_points:
                points = reader.read()
                y = np.asarray(points.y, dtype=np.float64)
            else:
                step = max(1, total // max_points)
                ys: List[float] = []
                count = 0
                for chunk in reader.chunk_iterator(1_000_000):
                    y = np.asarray(chunk.y, dtype=np.float64)
                    for j in range(0, y.size, step):
                        ys.append(float(y[j]))
                        count += 1
                        if count >= max_points:
                            break
                    if count >= max_points:
                        break
                y = np.asarray(ys, dtype=np.float64)
    except Exception as exc:
        return {"ok": False, "reason": f"read_failed:{exc}"}
    if y.size == 0:
        return {"ok": False, "reason": "empty", "min_nonzero_dy": None, "unique_y_1mm": 0, "sample_n": 0}
    y_round = np.round(y, 3)
    y_unique = np.unique(y_round)
    dy = np.diff(np.sort(y_unique))
    nonzero = dy[dy > 0]
    min_nonzero = float(np.min(nonzero)) if nonzero.size else None
    return {
        "ok": bool(min_nonzero is not None and min_nonzero <= 0.01),
        "min_nonzero_dy": min_nonzero,
        "unique_y_1mm": int(y_unique.size),
        "sample_n": int(y.size),
    }


def _intensity_stats(min_val: float, max_val: float, sum_val: float, nonzero: int, total: int) -> Dict[str, float]:
    if total <= 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "nonzero_ratio": 0.0}
    mean = sum_val / float(total)
    return {
        "min": float(min_val),
        "mean": float(mean),
        "max": float(max_val),
        "nonzero_ratio": float(nonzero) / float(total),
    }


def _apply_world_to_utm32_transform(points_w: np.ndarray, transform: Dict[str, float]) -> np.ndarray:
    if points_w.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if points_w.dtype != np.float64:
        raise RuntimeError("utm32_input_not_float64")
    dx = np.float64(transform.get("dx", 0.0))
    dy = np.float64(transform.get("dy", 0.0))
    dz = np.float64(transform.get("dz", 0.0))
    yaw_deg = np.float64(transform.get("yaw_deg", 0.0))
    scale = np.float64(transform.get("scale", 1.0))
    if any(x.dtype != np.float64 for x in [dx, dy, dz, yaw_deg, scale]):
        raise RuntimeError("utm32_transform_not_float64")
    yaw = np.deg2rad(yaw_deg)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    pts = np.asarray(points_w, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    x2 = scale * (c * x - s * y) + dx
    y2 = scale * (s * x + c * y) + dy
    z2 = z + dz
    return np.stack([x2, y2, z2], axis=1)


def _bbox_ok(bounds: Tuple[float, float, float, float, float, float], coord: str) -> Dict[str, object]:
    minx, miny, minz, maxx, maxy, maxz = bounds
    if minx >= maxx or miny >= maxy:
        return {"ok": False, "reason": "bbox_invalid_order"}
    if all(abs(v) < 1e-9 for v in bounds):
        return {"ok": False, "reason": "bbox_all_zero"}
    if coord == "utm32":
        if minx < 100000.0 or maxx > 900000.0:
            return {"ok": False, "reason": "bbox_easting_out_of_range"}
        if miny < 0.0 or maxy > 10000000.0:
            return {"ok": False, "reason": "bbox_northing_out_of_range"}
    if minz == maxz == 0.0:
        return {"ok": False, "reason": "bbox_z_all_zero"}
    return {"ok": True, "reason": "ok"}


def _missing_summary(missing_map: Dict[str, str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for reason in missing_map.values():
        out[reason] = out.get(reason, 0) + 1
    return out


def _compute_utm32_evidence(
    data_root: Path,
    drive_id: str,
    frame_ids: List[str],
    cam_id: str,
    max_frames: int = 20,
    sample_points: int = 200,
) -> Tuple[Dict[str, object], Optional[str]]:
    if not frame_ids:
        return {"common_frames_used": 0}, "no_frames"
    used = 0
    offsets: List[np.ndarray] = []
    residuals_xy: List[float] = []
    for fid in frame_ids[:max_frames]:
        try:
            raw = load_kitti360_lidar_points(data_root, drive_id, fid)
            if raw.size == 0:
                continue
            pts_v = raw[:, :3].astype(np.float64)
            pts_w = transform_points_V_to_W(pts_v, data_root, drive_id, fid, cam_id=cam_id)
            pts_u = kitti_world_to_utm32(pts_w, data_root, drive_id, fid)
            if pts_w.shape[0] == 0:
                continue
            rng = np.random.default_rng(0)
            if pts_w.shape[0] > sample_points:
                idx = rng.choice(pts_w.shape[0], size=sample_points, replace=False)
                pts_w = pts_w[idx]
                pts_u = pts_u[idx]
            delta = pts_u - pts_w
            offsets.append(delta)
            used += 1
        except Exception as exc:
            return {"common_frames_used": used}, f"utm32_evidence_failed:{exc}"
    if not offsets:
        return {"common_frames_used": used}, "no_offsets"
    all_offsets = np.vstack(offsets)
    median = np.median(all_offsets, axis=0)
    for delta in offsets:
        resid = delta[:, :2] - median[:2]
        rms = float(np.sqrt(np.mean(resid[:, 0] ** 2 + resid[:, 1] ** 2)))
        residuals_xy.append(rms)
    rms_xy = float(np.mean(residuals_xy)) if residuals_xy else 0.0
    return (
        {
            "world_to_utm32_offset_median": {"dx": float(median[0]), "dy": float(median[1]), "dz": float(median[2])},
            "utm32_xy_residual_rms_m": rms_xy,
            "common_frames_used": used,
        },
        None,
    )


def _swap_coord_suffix(path: Path, coord: str) -> Path:
    name = path.name
    if "utm32" in name or "world" in name:
        if coord == "utm32":
            name = name.replace("world", "utm32")
        else:
            name = name.replace("utm32", "world")
    else:
        stem = path.stem
        suffix = path.suffix
        name = f"{stem}_{coord}{suffix}"
    return path.with_name(name)


def _write_npz_stream(
    out_path: Path,
    frame_ids: List[str],
    loader_fn,
    intensity_fn,
    tmp_dir: Path,
) -> Tuple[int, List[str]]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    xyz_path = tmp_dir / "xyz.npy"
    intensity_path = tmp_dir / "intensity.npy"
    frame_path = tmp_dir / "frame_id.npy"

    errors: List[str] = []
    total_points = 0
    for fid in frame_ids:
        pts, inten = loader_fn(fid)
        if pts is None:
            continue
        total_points += int(pts.shape[0])

    xyz_mm = np.lib.format.open_memmap(xyz_path, mode="w+", dtype=np.float32, shape=(total_points, 3))
    intensity_mm = np.lib.format.open_memmap(intensity_path, mode="w+", dtype=np.uint16, shape=(total_points,))
    frame_mm = np.lib.format.open_memmap(frame_path, mode="w+", dtype=np.int32, shape=(total_points,))

    cursor = 0
    for fid in frame_ids:
        pts, inten = loader_fn(fid)
        if pts is None or pts.size == 0:
            continue
        n = int(pts.shape[0])
        xyz_mm[cursor : cursor + n] = pts.astype(np.float32)
        intensity_mm[cursor : cursor + n] = intensity_fn(inten)[0]
        frame_mm[cursor : cursor + n] = int(fid)
        cursor += n

    xyz_mm.flush()
    intensity_mm.flush()
    frame_mm.flush()

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(xyz_path, arcname="xyz.npy")
        zf.write(intensity_path, arcname="intensity.npy")
        zf.write(frame_path, arcname="frame_id.npy")

    try:
        gc.collect()
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass
    if tmp_dir.exists():
        errors.append(f"tmp_cleanup_failed:{tmp_dir}")
    return total_points, errors


def fuse_frames_to_las(
    data_root: Path,
    drive_id: str,
    frame_start: int,
    frame_end: int,
    stride: int,
    out_path: Path,
    coord: str,
    cam_id: str = "image_00",
    overwrite: bool = False,
    output_format: str = "laz",
    require_laz: bool = False,
    require_cam0_to_world: bool = True,
    allow_poses_fallback: bool = False,
    use_r_rect_with_cam0_to_world: bool = True,
    world_to_utm32_transform: Optional[Dict[str, float]] = None,
    enable_chunking: bool = False,
    target_laz_mb_per_part: float = 1200.0,
    max_parts: int = 8,
    per_frame_sample_stride: int = 50,
) -> FusionResult:
    frame_ids = _frame_ids(frame_start, frame_end, stride)
    missing_map: Dict[str, str] = {}
    warnings: List[str] = []
    errors: List[str] = []

    velodyne_dir = _find_velodyne_dir(data_root, drive_id)
    tmp_dir = out_path.parent.parent / "tmp"
    enable_chunking = bool(enable_chunking)

    fmt = str(output_format or "laz").lower()
    if require_laz and fmt != "laz":
        raise ValueError("require_laz=True but output_format is not laz")
    if fmt not in {"laz", "las", "npz"}:
        raise ValueError(f"unsupported output_format:{fmt}")
    if coord == "utm32" and world_to_utm32_transform is None:
        raise ValueError("utm32_transform_required")

    cam0_to_world_path = _find_pose_file(data_root, drive_id, ["cam0_to_world.txt"])
    poses_path = _find_pose_file(data_root, drive_id, ["poses.txt"])
    pose_source = "cam0_to_world"
    cam0_to_world_map: Optional[Dict[str, np.ndarray]] = None
    poses_map: Optional[Dict[str, np.ndarray]] = None
    t_cam0_to_pose: Optional[np.ndarray] = None
    t_rect: Optional[np.ndarray] = None
    t_rect_inv: Optional[np.ndarray] = None

    if cam0_to_world_path is None:
        if require_cam0_to_world and not allow_poses_fallback:
            raise FileNotFoundError(
                "missing_cam0_to_world (set allow_poses_fallback=True to enable poses.txt fallback)"
            )
        if not allow_poses_fallback:
            raise FileNotFoundError("missing_cam0_to_world")
        if poses_path is None:
            raise FileNotFoundError("missing_poses_txt_for_fallback")
        pose_source = "poses_fallback"
        poses_map = _parse_pose_map(poses_path)
        t_cam0_to_pose = load_kitti360_cam_to_pose(data_root, cam_id)
        calib = _load_kitti360_calib(data_root, cam_id)
        r_rect = calib.get("r_rect")
        if r_rect is None:
            raise FileNotFoundError("missing_r_rect_for_fallback")
        t_rect = np.eye(4, dtype=float)
        t_rect[:3, :3] = r_rect
    else:
        cam0_to_world_map = _parse_pose_map(cam0_to_world_path)
        calib = _load_kitti360_calib(data_root, cam_id)
        r_rect = calib.get("r_rect")
        if r_rect is None:
            raise FileNotFoundError("missing_r_rect_for_cam0_to_world")
        t_rect = np.eye(4, dtype=float)
        t_rect[:3, :3] = r_rect
        t_rect_inv = np.linalg.inv(t_rect)

    def _coord_transform(pts: np.ndarray, fid: str, coord_mode: str) -> np.ndarray:
        if pose_source == "cam0_to_world":
            if cam0_to_world_map is not None:
                if fid not in cam0_to_world_map:
                    raise KeyError("missing_pose_frame")
                t_w_c0 = cam0_to_world_map[fid]
                calib = _load_kitti360_calib(data_root, cam_id)
                t_v_c0 = calib["t_cam_to_velo"]
                t_c0_v = np.linalg.inv(t_v_c0)
                if use_r_rect_with_cam0_to_world:
                    if t_rect is None:
                        raise RuntimeError("missing_r_rect_for_cam0_to_world")
                    t_chain = t_w_c0 @ t_rect @ t_c0_v
                else:
                    t_chain = t_w_c0 @ t_c0_v
                pts_h = np.hstack([pts[:, :3], np.ones((pts.shape[0], 1), dtype=pts.dtype)])
                pts_w = (t_chain @ pts_h.T).T
                pts_w = pts_w[:, :3].astype(np.float64)
            else:
                pts_w = transform_points_V_to_W(pts, data_root, drive_id, fid, cam_id=cam_id)
        else:
            if poses_map is None or t_cam0_to_pose is None or t_rect is None:
                raise RuntimeError("poses_fallback_not_ready")
            if fid not in poses_map:
                raise KeyError("missing_pose_frame")
            t_imu_w = poses_map[fid]
            t_rectcam0_to_world = t_imu_w @ t_cam0_to_pose
            calib = _load_kitti360_calib(data_root, cam_id)
            t_v_c0 = calib["t_cam_to_velo"]
            t_c0_v = np.linalg.inv(t_v_c0)
            pts_h = np.hstack([pts[:, :3], np.ones((pts.shape[0], 1), dtype=pts.dtype)])
            pts_w = (t_rectcam0_to_world @ (t_c0_v @ pts_h.T)).T
            pts_w = pts_w[:, :3].astype(np.float64)
        if coord_mode == "utm32":
            if world_to_utm32_transform is not None:
                return _apply_world_to_utm32_transform(pts_w, world_to_utm32_transform)
            return kitti_world_to_utm32(pts_w, data_root, drive_id, fid)
        return pts_w

    def _load_points(
        fid: str, coord_mode: str, track_stats: bool, stats: Dict[str, int]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            bin_path = _resolve_velodyne_path(velodyne_dir, fid)
            if bin_path is None or not bin_path.exists():
                missing_map.setdefault(fid, "missing_velodyne")
                return None
            if track_stats:
                stats["frames_found_velodyne"] += 1
            raw = load_kitti360_lidar_points(data_root, drive_id, fid)
            if raw.size == 0:
                missing_map.setdefault(fid, "empty_points")
                return None
            pts = raw[:, :3].astype(np.float32)
            inten = raw[:, 3].astype(np.float32)
            pts = _coord_transform(pts, fid, coord_mode)
            if coord_mode == "utm32":
                if pts.dtype != np.float64:
                    raise RuntimeError("utm32_points_not_float64")
                return pts.astype(np.float64), inten
            return pts.astype(np.float32), inten
        except Exception as exc:
            missing_map.setdefault(fid, f"frame_failed:{exc}")
            return None

    def _run(coord_mode: str) -> FusionResult:
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
        total_points = 0
        intensity_min = 0.0
        intensity_max = 0.0
        intensity_sum = 0.0
        intensity_nonzero = 0
        intensity_rule = "unknown"
        missing_map.clear()
        utm32_evidence: Dict[str, object] = {}
        utm32_evidence_reason: Optional[str] = None
        stats = {
            "frames_found_velodyne": 0,
            "frames_processed": 0,
            "points_read_total": 0,
        }
        per_frame_points: List[Dict[str, object]] = []

        if coord_mode == "utm32":
            utm32_evidence, utm32_evidence_reason = _compute_utm32_evidence(
                data_root, drive_id, frame_ids, cam_id
            )
            if utm32_evidence_reason:
                utm32_evidence["reason"] = utm32_evidence_reason

        for idx_f, fid in enumerate(frame_ids):
            loaded = _load_points(fid, coord_mode, track_stats=True, stats=stats)
            if loaded is None:
                if (idx_f + 1) % 10 == 0:
                    LOG.info(
                        "frame_progress: %s/%s total_points=%s missing=%s",
                        idx_f + 1,
                        len(frame_ids),
                        total_points,
                        len(missing_map),
                    )
                continue
            pts, inten = loaded
            if pts.size == 0:
                continue
            stats["frames_processed"] += 1
            mapped, rule = intensity_float_to_uint16(inten)
            if intensity_rule == "unknown":
                intensity_rule = rule
            elif intensity_rule != rule:
                intensity_rule = "mixed"
            bounds = _bbox_update(bounds, pts)
            total_points += int(pts.shape[0])
            stats["points_read_total"] += int(pts.shape[0])
            if mapped.size:
                cur_min = float(np.min(mapped))
                cur_max = float(np.max(mapped))
                if total_points == int(pts.shape[0]):
                    intensity_min = cur_min
                    intensity_max = cur_max
                else:
                    intensity_min = min(intensity_min, cur_min)
                    intensity_max = max(intensity_max, cur_max)
                intensity_sum += float(np.sum(mapped))
                intensity_nonzero += int(np.sum(mapped > 0))
            if per_frame_sample_stride > 0 and (idx_f % per_frame_sample_stride) == 0:
                per_frame_points.append(
                    {
                        "frame": fid,
                        "points_in": int(pts.shape[0]),
                        "points_written": int(pts.shape[0]),
                    }
                )
            if (idx_f + 1) % 10 == 0:
                LOG.info(
                    "frame_progress: %s/%s total_points=%s missing=%s",
                    idx_f + 1,
                    len(frame_ids),
                    total_points,
                    len(missing_map),
                )

        missing_frames = [{"frame_id": k, "reason": v} for k, v in sorted(missing_map.items())]
        missing_summary = _missing_summary(missing_map)

        if total_points <= 0 or bounds is None:
            bbox = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            bbox_check = _bbox_ok(bbox, coord_mode)
            return FusionResult(
                coord=coord_mode,
                epsg=None,
                total_points=0,
                written_points=0,
                bbox=bbox,
                bbox_check=bbox_check,
                intensity_stats=_intensity_stats(0.0, 0.0, 0.0, 0, 0),
                intensity_rule=intensity_rule,
                missing_frames=missing_frames,
                missing_summary=missing_summary,
                warnings=warnings,
                errors=errors + ["no_points_accumulated"],
                utm32_evidence=utm32_evidence,
                utm32_failed_reason=utm32_evidence_reason,
                output_path=None,
                output_format="",
                pose_source=pose_source,
                use_r_rect_with_cam0_to_world=use_r_rect_with_cam0_to_world,
                output_paths=[],
                frames_found_velodyne=stats["frames_found_velodyne"],
                frames_processed=stats["frames_processed"],
                points_read_total=stats["points_read_total"],
                points_written_total=0,
                per_frame_points_sample=per_frame_points,
                banding_check={},
            )

        bounds_tuple = _bbox_tuple(bounds)
        bbox_check = _bbox_ok(bounds_tuple, coord_mode)
        if coord_mode == "utm32" and not bbox_check.get("ok"):
            errors.append(f"utm32_bbox_check_failed:{bbox_check.get('reason')}")
            return FusionResult(
                coord=coord_mode,
                epsg=None,
                total_points=total_points,
                written_points=0,
                bbox=bounds_tuple,
                bbox_check=bbox_check,
                intensity_stats=_intensity_stats(intensity_min, intensity_max, intensity_sum, intensity_nonzero, total_points),
                intensity_rule=intensity_rule,
                missing_frames=missing_frames,
                missing_summary=missing_summary,
                warnings=warnings,
                errors=errors,
                utm32_evidence=utm32_evidence,
                utm32_failed_reason=utm32_evidence_reason,
                output_path=None,
                output_format="",
                pose_source=pose_source,
                use_r_rect_with_cam0_to_world=use_r_rect_with_cam0_to_world,
                output_paths=[],
                frames_found_velodyne=stats["frames_found_velodyne"],
                frames_processed=stats["frames_processed"],
                points_read_total=stats["points_read_total"],
                points_written_total=0,
                per_frame_points_sample=per_frame_points,
                banding_check={},
            )

        output_path = _swap_coord_suffix(out_path, coord_mode)
        if overwrite and output_path.exists():
            output_path.unlink()

        written_points = 0
        if fmt == "npz":
            output_path = output_path.with_suffix(".npz")

            def _loader(fid: str):
                loaded = _load_points(fid, coord_mode)
                if loaded is None:
                    return None, None
                pts, inten = loaded
                return pts, inten

            total_points_npz, npz_errors = _write_npz_stream(
                output_path, frame_ids, _loader, intensity_float_to_uint16, tmp_dir
            )
            for err in npz_errors:
                warnings.append(err)
            written_points = total_points_npz
            return FusionResult(
                coord=coord_mode,
                epsg=None,
                total_points=total_points_npz,
                written_points=written_points,
                bbox=bounds_tuple,
                bbox_check=bbox_check,
                intensity_stats=_intensity_stats(intensity_min, intensity_max, intensity_sum, intensity_nonzero, total_points),
                intensity_rule=intensity_rule,
                missing_frames=missing_frames,
                missing_summary=missing_summary,
                warnings=warnings,
                errors=errors,
                utm32_evidence=utm32_evidence,
                utm32_failed_reason=utm32_evidence_reason,
                output_path=output_path,
                output_format="npz",
                pose_source=pose_source,
                use_r_rect_with_cam0_to_world=use_r_rect_with_cam0_to_world,
                output_paths=[str(output_path)],
                frames_found_velodyne=stats["frames_found_velodyne"],
                frames_processed=stats["frames_processed"],
                points_read_total=stats["points_read_total"],
                points_written_total=written_points,
                per_frame_points_sample=per_frame_points,
                banding_check={},
            )

        try:
            import laspy  # type: ignore
        except Exception as exc:
            if require_laz:
                raise RuntimeError(f"laspy_missing:{exc}") from exc
            warnings.append(f"laspy_missing:{exc}")
            raise

        if fmt == "laz":
            output_path = output_path.with_suffix(".laz")
        elif fmt == "las":
            output_path = output_path.with_suffix(".las")

        header = laspy.LasHeader(point_format=0, version="1.2")
        if coord_mode == "utm32":
            header.scales = (0.001, 0.001, 0.001)
        else:
            header.scales = _safe_scales(bounds_tuple)
        header.offsets = (bounds_tuple[0], bounds_tuple[1], bounds_tuple[2])
        allow_epsg = coord_mode == "utm32" and bool(bbox_check.get("ok"))
        try:
            if allow_epsg:
                header.add_crs(32632)
        except Exception as exc:
            warnings.append(f"las_crs_failed:{exc}")

        do_compress = fmt == "laz"
        if do_compress:
            try:
                import lazrs  # type: ignore
            except Exception as exc:
                if require_laz:
                    raise RuntimeError(f"lazrs_missing:{exc}") from exc
                warnings.append(f"lazrs_missing:{exc}")
                raise

        def _write_records(writer_obj, fid_list: List[str]) -> int:
            count = 0
            for fid in fid_list:
                loaded = _load_points(fid, coord_mode, track_stats=False, stats=stats)
                if loaded is None:
                    continue
                pts, inten = loaded
                if pts.size == 0:
                    continue
                mapped, _ = intensity_float_to_uint16(inten)
                record = laspy.ScaleAwarePointRecord.zeros(pts.shape[0], header=header)
                if coord_mode == "utm32":
                    scale = np.array(header.scales, dtype=np.float64)
                    offset = np.array(header.offsets, dtype=np.float64)
                    coords = pts.astype(np.float64)
                    xyz_int = np.round((coords - offset) / scale).astype(np.int64)
                    record.X = xyz_int[:, 0].astype(np.int32)
                    record.Y = xyz_int[:, 1].astype(np.int32)
                    record.Z = xyz_int[:, 2].astype(np.int32)
                else:
                    record.x = pts[:, 0]
                    record.y = pts[:, 1]
                    record.z = pts[:, 2]
                record.intensity = mapped
                writer_obj.write_points(record)
                count += int(pts.shape[0])
            return count

        output_paths: List[str] = []
        chunking = enable_chunking
        if chunking and fmt == "laz":
            total_frames = len(frame_ids)
            total_points_f = max(1, int(total_points))
            parts_est = int(math.ceil((total_points_f * 16) / (target_laz_mb_per_part * 1024 * 1024)))
            parts = min(max(1, parts_est), max_parts)
            if parts <= 1:
                chunking = False
            else:
                per_part = int(math.ceil(total_frames / parts))
                part_ranges: List[Tuple[int, int]] = []
                for i in range(parts):
                    start_idx = i * per_part
                    end_idx = min(total_frames - 1, (i + 1) * per_part - 1)
                    if start_idx >= total_frames:
                        break
                    part_ranges.append((start_idx, end_idx))
                for start_idx, end_idx in part_ranges:
                    start_f = frame_ids[start_idx]
                    end_f = frame_ids[end_idx]
                    part_name = f"{output_path.stem}_part_{start_f}_{end_f}{output_path.suffix}"
                    part_path = output_path.with_name(part_name)
                    try:
                        writer = laspy.open(part_path, mode="w", header=header, do_compress=do_compress)
                    except Exception as exc:
                        raise RuntimeError(f"laz_write_failed:{exc}") from exc
                    with writer:
                        written_points += _write_records(writer, frame_ids[start_idx : end_idx + 1])
                    output_paths.append(str(part_path))
        if not chunking:
            try:
                writer = laspy.open(output_path, mode="w", header=header, do_compress=do_compress)
            except Exception as exc:
                if require_laz or do_compress:
                    raise RuntimeError(f"laz_write_failed:{exc}") from exc
                raise
            with writer:
                written_points += _write_records(writer, frame_ids)
            output_paths.append(str(output_path))

        banding_check: Dict[str, object] = {}
        if coord_mode == "utm32" and output_paths:
            banding_check = _banding_check(Path(output_paths[0]))
            if not banding_check.get("ok"):
                raise RuntimeError(f"banding_check_failed:{banding_check}")

        LOG.info(
            "run_summary: coord=%s missing=%s bbox_ok=%s intensity_max=%.2f nonzero=%.4f",
            coord_mode,
            len(missing_frames),
            bbox_check.get("ok"),
            float(intensity_max),
            float(intensity_nonzero) / float(total_points) if total_points else 0.0,
        )

        primary_path = Path(output_paths[0]) if output_paths else None
        return FusionResult(
            coord=coord_mode,
            epsg=32632 if allow_epsg else None,
            total_points=total_points,
            written_points=written_points,
            bbox=bounds_tuple,
            bbox_check=bbox_check,
            intensity_stats=_intensity_stats(intensity_min, intensity_max, intensity_sum, intensity_nonzero, total_points),
            intensity_rule=intensity_rule,
            missing_frames=missing_frames,
            missing_summary=missing_summary,
            warnings=warnings,
            errors=errors,
            utm32_evidence=utm32_evidence,
            utm32_failed_reason=utm32_evidence_reason,
            output_path=primary_path,
            output_format=fmt,
            pose_source=pose_source,
            use_r_rect_with_cam0_to_world=use_r_rect_with_cam0_to_world,
            output_paths=output_paths,
            frames_found_velodyne=stats["frames_found_velodyne"],
            frames_processed=stats["frames_processed"],
            points_read_total=stats["points_read_total"],
            points_written_total=written_points,
            per_frame_points_sample=per_frame_points,
            banding_check=banding_check,
        )

    def _cleanup_tmp(result: FusionResult) -> FusionResult:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if tmp_dir.exists():
            result.warnings.append(f"tmp_cleanup_failed:{tmp_dir}")
        return result

    if coord == "auto":
        utm_result = _run("utm32")
        evidence = utm_result.utm32_evidence or {}
        if not utm_result.bbox_check.get("ok"):
            utm_result.utm32_failed_reason = str(utm_result.bbox_check.get("reason", "bbox_failed"))
            world_result = _run("world")
            world_result.utm32_failed_reason = utm_result.utm32_failed_reason
            world_result.utm32_evidence = evidence
            world_result.warnings.append(f"utm32_failed:{utm_result.utm32_failed_reason}")
            return _cleanup_tmp(world_result)
        utm_result.utm32_failed_reason = None
        return _cleanup_tmp(utm_result)

    result = _run(coord)
    return _cleanup_tmp(result)


__all__ = [
    "FusionResult",
    "fuse_frames_to_las",
    "intensity_float_to_uint16",
    "load_cam0_to_world",
    "load_kitti360_calib",
    "load_oxts_to_utm32_optional",
    "collect_input_fingerprints",
]

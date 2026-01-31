from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from pipeline.datasets.kitti360_io import (
    _find_oxts_dir,
    _find_velodyne_dir,
    load_kitti360_calib,
    load_kitti360_cam_to_pose,
    load_kitti360_cam_to_pose_key,
    load_kitti360_lidar_points,
    load_kitti360_lidar_points_world_full,
    load_kitti360_pose_full,
)
from pipeline.projection.projector import project_points_cam0_to_image, world_points_to_image
from scripts.pipeline_common import (
    ensure_dir,
    ensure_overwrite,
    now_ts,
    relpath,
    setup_logging,
    validate_output_crs,
    write_csv,
    write_gpkg_layer,
    write_json,
    write_text,
)


REQUIRED_KEYS = [
    "FRAME_START",
    "FRAME_END",
    "DRIVE_MATCH",
    "SAMPLE_FRAMES_FOR_OVERLAY",
    "LIDAR_MAX_POINTS_FOR_OVERLAY",
    "OUTPUT_IMAGE_CAM",
    "OVERWRITE",
    "TARGET_EPSG",
    "KITTI_ROOT",
    "GROUND_ONLY_ENABLE",
    "GROUND_BAND_METHOD",
    "GROUND_PCTL",
    "GROUND_BAND_DZ_M",
    "GROUND_XYZ_FRAME",
    "GROUND_SAMPLE_MAX_POINTS",
    "GROUND_ONLY_VERSION",
    "GROUND_MODE_BIN_M",
    "GROUND_MODE_MIN_SHARE",
    "GROUND_PCTL_FALLBACK",
    "GROUND_DZ_INIT_M",
    "PROJ_DIST_MIN_M",
    "PROJ_DIST_MAX_M",
    "PROJ_MARGIN_RATIO",
    "GROUND_V3_ENABLE",
    "GROUND_V3_V_MIN_RATIO",
    "GROUND_V3_V_MIN_RELAX",
    "GROUND_V3_U_MIN_RATIO",
    "GROUND_V3_U_MAX_RATIO",
    "GROUND_V3_SEED_MIN",
    "GROUND_V3_RANSAC_ITERS",
    "GROUND_V3_INLIER_D_M",
    "GROUND_V3_NORMAL_MIN_Y",
    "GROUND_V3_DZ_PLANE_INIT",
]


def _load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    import yaml

    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _normalize(cfg: Dict[str, object]) -> Dict[str, object]:
    def _norm(v):
        if isinstance(v, dict):
            return {k: _norm(v[k]) for k in sorted(v.keys())}
        if isinstance(v, list):
            return [_norm(x) for x in v]
        return v

    return _norm(cfg)


def _hash_cfg(cfg: Dict[str, object]) -> str:
    import hashlib

    raw = json.dumps(_normalize(cfg), sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _write_resolved(run_dir: Path, cfg: Dict[str, object]) -> str:
    import yaml

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    params_hash = _hash_cfg(cfg)
    (run_dir / "params_hash.txt").write_text(params_hash + "\n", encoding="utf-8")
    return params_hash


def _resolve_config(base: Dict[str, object], run_dir: Path) -> Tuple[Dict[str, object], str]:
    cfg = dict(base)
    defaults = {
        "FRAME_START": 250,
        "FRAME_END": 500,
        "DRIVE_MATCH": "_0010_",
        "SAMPLE_FRAMES_FOR_OVERLAY": 12,
        "LIDAR_MAX_POINTS_FOR_OVERLAY": 120000,
        "OUTPUT_IMAGE_CAM": "image_00",
        "OVERWRITE": True,
        "TARGET_EPSG": 32632,
        "KITTI_ROOT": "",
        "GROUND_ONLY_ENABLE": True,
        "GROUND_BAND_METHOD": "percentile",
        "GROUND_PCTL": 2.0,
        "GROUND_BAND_DZ_M": 0.15,
        "GROUND_XYZ_FRAME": "velo",
        "GROUND_SAMPLE_MAX_POINTS": 80000,
        "OVERLAY_GROUND_COLOR": [255, 255, 0],
        "GROUND_ONLY_VERSION": "v2",
        "GROUND_MODE_BIN_M": 0.05,
        "GROUND_MODE_MIN_SHARE": 0.08,
        "GROUND_PCTL_FALLBACK": 10,
        "GROUND_DZ_INIT_M": 0.20,
        "PROJ_DIST_MIN_M": 3.0,
        "PROJ_DIST_MAX_M": 45.0,
        "PROJ_MARGIN_RATIO": 0.2,
        "GROUND_V3_ENABLE": True,
        "GROUND_V3_V_MIN_RATIO": 0.65,
        "GROUND_V3_V_MIN_RELAX": 0.55,
        "GROUND_V3_U_MIN_RATIO": 0.2,
        "GROUND_V3_U_MAX_RATIO": 0.8,
        "GROUND_V3_SEED_MIN": 2000,
        "GROUND_V3_RANSAC_ITERS": 200,
        "GROUND_V3_INLIER_D_M": 0.12,
        "GROUND_V3_NORMAL_MIN_Y": 0.7,
        "GROUND_V3_DZ_PLANE_INIT": 0.12,
    }
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys: {missing}")
    params_hash = _write_resolved(run_dir, cfg)
    return cfg, params_hash


def _auto_find_kitti_root(cfg: Dict[str, object], scans: List[str]) -> Optional[Path]:
    cfg_root = str(cfg.get("KITTI_ROOT") or "").strip()
    if cfg_root:
        scans.append(cfg_root)
        p = Path(cfg_root)
        if p.exists():
            return p
    env_root = os.environ.get("POC_DATA_ROOT", "").strip()
    if env_root:
        scans.append(env_root)
        p = Path(env_root)
        if p.exists():
            return p
    for cand in [r"E:\KITTI360\KITTI-360", r"D:\KITTI360\KITTI-360", r"C:\KITTI360\KITTI-360"]:
        scans.append(cand)
        p = Path(cand)
        if p.exists():
            return p
    repo = Path(".").resolve()
    for base in [repo / "data", repo / "datasets"]:
        if not base.exists():
            continue
        for child in base.iterdir():
            scans.append(str(child))
            if child.is_dir() and ("KITTI-360" in child.name or "KITTI360" in child.name):
                return child
    return None


def _select_drive(data_root: Path, match: str) -> str:
    drives_file = Path("configs/golden_drives.txt")
    if drives_file.exists():
        drives = [ln.strip() for ln in drives_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        raw_root = data_root / "data_3d_raw"
        drives = sorted([p.name for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("2013_05_28_drive_")])
    for d in drives:
        if match in d:
            return d
    raise RuntimeError("no_drive_match")


def _find_image_dir(data_root: Path, drive_id: str, cam: str) -> Optional[Path]:
    candidates = [
        data_root / "data_2d_raw" / drive_id / cam / "data",
        data_root / "data_2d_raw" / drive_id / cam / "data_rect",
        data_root / drive_id / cam / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_image_path(img_dir: Path, frame_id: str) -> Optional[Path]:
    for ext in [".png", ".jpg", ".jpeg"]:
        p = img_dir / f"{frame_id}{ext}"
        if p.exists():
            return p
    return None


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
        out[str(frame_id)] = mat
    return out, direction


def _parse_imu_to_velo(calib_dir: Path) -> Tuple[Optional[np.ndarray], str]:
    cand_imu_to_velo = calib_dir / "calib_imu_to_velo.txt"
    cand_velo_to_imu = calib_dir / "calib_velo_to_imu.txt"
    if cand_imu_to_velo.exists():
        nums = [float(v) for v in cand_imu_to_velo.read_text(encoding="utf-8").split()]
        if len(nums) == 12:
            mat = np.array(nums, dtype=float).reshape(3, 4)
            bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float)
            return np.vstack([mat, bottom]), "imu_to_velo"
    if cand_velo_to_imu.exists():
        nums = [float(v) for v in cand_velo_to_imu.read_text(encoding="utf-8").split()]
        if len(nums) == 12:
            mat = np.array(nums, dtype=float).reshape(3, 4)
            bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float)
            return np.linalg.inv(np.vstack([mat, bottom])), "velo_to_imu_inverted"
    return None, "missing"


def _pose_matrix(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    c1 = float(np.cos(yaw))
    s1 = float(np.sin(yaw))
    c2 = float(np.cos(pitch))
    s2 = float(np.sin(pitch))
    c3 = float(np.cos(roll))
    s3 = float(np.sin(roll))
    r_z = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    r_y = np.array([[c2, 0.0, s2], [0.0, 1.0, 0.0], [-s2, 0.0, c2]], dtype=float)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, c3, -s3], [0.0, s3, c3]], dtype=float)
    r = r_z @ r_y @ r_x
    t = np.array([x, y, z], dtype=float)
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = r
    mat[:3, 3] = t
    return mat


def _delta_pose(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    inv_a = np.linalg.inv(a)
    d = inv_a @ b
    dt = float(np.linalg.norm(d[:3, 3]))
    r = d[:3, :3]
    trace = float(np.trace(r))
    val = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    ang = float(np.degrees(np.arccos(val)))
    return dt, ang


def _sample_frames(start: int, end: int, count: int) -> List[str]:
    if count <= 1:
        return [f"{start:010d}"]
    idx = np.linspace(start, end, count)
    frames = sorted({int(round(v)) for v in idx})
    return [f"{i:010d}" for i in frames]


def _project_stats(
    points_world: np.ndarray, pose: Tuple[float, float, float, float, float, float], calib: Dict[str, np.ndarray], img: np.ndarray
) -> Tuple[int, int, float]:
    h, w = img.shape[0], img.shape[1]
    u, v, valid, in_img = world_points_to_image(points_world, pose, calib, (h, w), use_rect=True, y_flip=True)
    n_total = int(points_world.shape[0])
    n_in = int(np.sum(in_img))
    ratio = float(n_in) / max(1, n_total)
    return n_total, n_in, ratio


def _write_overlay(
    img_path: Path,
    points_world: np.ndarray,
    pose: Tuple[float, float, float, float, float, float],
    calib: Dict[str, np.ndarray],
    out_path: Path,
    params_hash: str,
    frame_id: str,
    max_points: int,
    color: str = "lime",
    title_suffix: str = "",
) -> float:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = plt.imread(img_path)
    h, w = img.shape[0], img.shape[1]
    pts = points_world
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    u, v, valid, in_img = world_points_to_image(pts, pose, calib, (h, w), use_rect=True, y_flip=True)
    ratio = float(np.sum(in_img)) / max(1, pts.shape[0])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.imshow(img)
    if np.any(in_img):
        ax.scatter(u[in_img], v[in_img], s=1.0, c=color, alpha=0.4)
    title = f"frame {frame_id} | in_image_ratio={ratio:.3f}"
    if title_suffix:
        title += f" | {title_suffix}"
    ax.set_title(title)
    ax.text(0.01, 0.99, f"params_hash={params_hash}", transform=ax.transAxes, fontsize=6, color="white", va="top")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return ratio


def _velo_to_cam(points_velo: np.ndarray, calib: Dict[str, np.ndarray]) -> np.ndarray:
    if points_velo.size == 0:
        return np.empty((0, 3), dtype=float)
    pts_h = np.hstack([points_velo[:, :3], np.ones((points_velo.shape[0], 1), dtype=points_velo.dtype)])
    cam = (calib["t_velo_to_cam"] @ pts_h.T).T
    return cam[:, :3]


def _estimate_ground_v2(
    z_vals: np.ndarray,
    bin_m: float,
    min_share: float,
    pctl_fallback: float,
) -> Tuple[float, str, float]:
    if z_vals.size == 0:
        return 0.0, "pctl", 0.0
    z_min = float(np.min(z_vals))
    z_max = float(np.max(z_vals))
    if z_max - z_min < bin_m:
        return float(np.percentile(z_vals, pctl_fallback)), "pctl", 0.0
    bins = np.arange(z_min, z_max + bin_m, bin_m, dtype=float)
    hist, edges = np.histogram(z_vals, bins=bins)
    idx = int(np.argmax(hist))
    share = float(hist[idx]) / max(1, int(z_vals.size))
    mode_val = float((edges[idx] + edges[idx + 1]) * 0.5)
    if share >= min_share:
        return mode_val, "mode", share
    fallback_pctl = pctl_fallback if z_vals.size >= 5000 else max(pctl_fallback, 20)
    return float(np.percentile(z_vals, fallback_pctl)), "pctl", share


def _fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, float]:
    if points.shape[0] < 3:
        return np.array([0.0, 1.0, 0.0], dtype=float), 0.0
    centroid = np.mean(points, axis=0)
    uu, ss, vv = np.linalg.svd(points - centroid)
    normal = vv[-1, :]
    norm = float(np.linalg.norm(normal))
    if norm == 0:
        normal = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        normal = normal / norm
    if normal[1] < 0:
        normal = -normal
    d = -float(np.dot(normal, centroid))
    return normal, d


def _ransac_plane(
    points: np.ndarray,
    iters: int,
    dist_thresh: float,
    normal_min_y: float,
) -> Tuple[Optional[np.ndarray], Optional[float], np.ndarray]:
    if points.shape[0] < 3:
        return None, None, np.zeros((0,), dtype=bool)
    rng = np.random.default_rng(0)
    best_inliers = np.zeros((points.shape[0],), dtype=bool)
    best_count = 0
    for _ in range(int(iters)):
        idx = rng.choice(points.shape[0], size=3, replace=False)
        p1, p2, p3 = points[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm = float(np.linalg.norm(n))
        if norm < 1e-6:
            continue
        n = n / norm
        if abs(float(n[1])) < float(normal_min_y):
            continue
        d = -float(np.dot(n, p1))
        dist = np.abs(points @ n + d)
        inliers = dist < float(dist_thresh)
        count = int(np.sum(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_count == 0:
        return None, None, np.zeros((points.shape[0],), dtype=bool)
    n_refit, d_refit = _fit_plane_svd(points[best_inliers])
    dist = np.abs(points @ n_refit + d_refit)
    inliers = dist < float(dist_thresh)
    return n_refit, d_refit, inliers


def main() -> int:
    base_cfg = _load_yaml(Path("configs/pose_audit_0010.yaml"))
    run_dir = Path("runs") / f"pose_audit_0010_f250_500_{now_ts()}"
    if bool(base_cfg.get("OVERWRITE", True)):
        ensure_overwrite(run_dir)
    ensure_dir(run_dir)
    setup_logging(run_dir / "run.log")

    cfg, params_hash = _resolve_config(base_cfg, run_dir)

    scans: List[str] = []
    data_root = _auto_find_kitti_root(cfg, scans)
    if data_root is None:
        write_text(run_dir / "report.md", "missing_kitti_root\n" + "\n".join(scans))
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_kitti_root", "params_hash": params_hash})
        return 2

    drive_id = _select_drive(data_root, str(cfg["DRIVE_MATCH"]))
    frame_start = int(cfg["FRAME_START"])
    frame_end = int(cfg["FRAME_END"])
    frame_ids = [f"{i:010d}" for i in range(frame_start, frame_end + 1)]

    errors: List[str] = []
    missing: List[str] = []

    try:
        oxts_dir = _find_oxts_dir(data_root, drive_id)
    except Exception as exc:
        oxts_dir = None
        errors.append(f"missing_oxts_dir:{exc}")

    try:
        velodyne_dir = _find_velodyne_dir(data_root, drive_id)
    except Exception as exc:
        velodyne_dir = None
        errors.append(f"missing_velodyne_dir:{exc}")

    calib_dir = data_root / "calibration"
    if not calib_dir.exists():
        errors.append("missing_calibration_dir")

    img_dir = _find_image_dir(data_root, drive_id, str(cfg["OUTPUT_IMAGE_CAM"]))
    if img_dir is None:
        errors.append("missing_image_dir")

    cam_pose_file = _find_cam_pose_file(data_root, drive_id)
    cam_pose_dir = cam_pose_file.parent if cam_pose_file else None
    cam_pose_dir_str = str(cam_pose_dir) if cam_pose_dir else ""
    cam_pose_direction = "missing"
    cam_pose_map: Dict[str, np.ndarray] = {}
    if cam_pose_file:
        cam_pose_map, cam_pose_direction = _parse_pose_file(cam_pose_file)

    if errors:
        report = ["# Pose Audit Report", "", "## Missing Resources", *[f"- {e}" for e in errors], "", "## Scanned Paths"]
        report += [f"- {p}" for p in scans]
        write_text(run_dir / "report.md", "\n".join(report))
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_resources", "errors": errors, "params_hash": params_hash})
        return 2

    try:
        calib = load_kitti360_calib(data_root, str(cfg["OUTPUT_IMAGE_CAM"]))
        cam_to_pose, cam_to_pose_key = load_kitti360_cam_to_pose_key(data_root, str(cfg["OUTPUT_IMAGE_CAM"]))
    except Exception as exc:
        write_text(run_dir / "report.md", f"missing_calib:{exc}")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_calib", "params_hash": params_hash})
        return 2

    imu_to_velo, imu_to_velo_src = _parse_imu_to_velo(calib_dir)
    cam_to_velo = calib["t_cam_to_velo"]

    cam0_source = "oxts"
    if cam_pose_map and cam_pose_direction in {"cam_to_world", "world_to_cam"}:
        cam0_source = "pose_file"
    elif cam_pose_map and cam_pose_direction == "unknown":
        missing.append("cam_pose_direction_ambiguous")

    # build per-frame transforms
    pose_rows = []
    traj_cam0 = []
    traj_velo_cam = []
    traj_velo_imu = []
    delta_rows = []
    missing_frames = []
    delta_t_list = []
    delta_r_list = []
    cam_pose_delta_t = []
    cam_pose_delta_r = []

    for frame_id in frame_ids:
        try:
            x, y, z, roll, pitch, yaw = load_kitti360_pose_full(data_root, drive_id, frame_id)
        except Exception as exc:
            missing_frames.append(f"{frame_id}:pose:{exc}")
            continue
        t_w_pose = _pose_matrix(x, y, z, roll, pitch, yaw)
        t_w_cam0_from_oxts = t_w_pose @ np.linalg.inv(cam_to_pose)

        t_w_cam0_from_file = None
        if cam_pose_map and frame_id in cam_pose_map and cam_pose_direction in {"cam_to_world", "world_to_cam"}:
            t = cam_pose_map[frame_id]
            if cam_pose_direction == "world_to_cam":
                t = np.linalg.inv(t)
            t_w_cam0_from_file = t
            dt_cam, dr_cam = _delta_pose(t_w_cam0_from_oxts, t_w_cam0_from_file)
            cam_pose_delta_t.append(dt_cam)
            cam_pose_delta_r.append(dr_cam)

        t_w_cam0 = t_w_cam0_from_file if cam0_source == "pose_file" and t_w_cam0_from_file is not None else t_w_cam0_from_oxts
        t_w_velo_from_cam = t_w_cam0 @ cam_to_velo
        t_w_velo_from_imu = None
        if imu_to_velo is not None:
            t_w_velo_from_imu = t_w_pose @ imu_to_velo

        if t_w_velo_from_imu is not None:
            dt, dr = _delta_pose(t_w_velo_from_cam, t_w_velo_from_imu)
            delta_t_list.append(dt)
            delta_r_list.append(dr)
        else:
            dt, dr = None, None

        traj_cam0.append((float(t_w_cam0[0, 3]), float(t_w_cam0[1, 3]), float(t_w_cam0[2, 3])))
        traj_velo_cam.append((float(t_w_velo_from_cam[0, 3]), float(t_w_velo_from_cam[1, 3]), float(t_w_velo_from_cam[2, 3])))
        if t_w_velo_from_imu is not None:
            traj_velo_imu.append((float(t_w_velo_from_imu[0, 3]), float(t_w_velo_from_imu[1, 3]), float(t_w_velo_from_imu[2, 3])))

        pose_rows.append(
            {
                "frame_id": frame_id,
                "cam0_x": float(t_w_cam0[0, 3]),
                "cam0_y": float(t_w_cam0[1, 3]),
                "cam0_z": float(t_w_cam0[2, 3]),
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw),
                "delta_t_m": dt if dt is not None else "",
                "delta_rot_deg": dr if dr is not None else "",
                "params_hash": params_hash,
            }
        )

        if velodyne_dir is not None:
            p = velodyne_dir / f"{frame_id}.bin"
            if not p.exists():
                missing_frames.append(f"{frame_id}:velodyne_missing")
        if img_dir is not None and _find_image_path(img_dir, frame_id) is None:
            missing_frames.append(f"{frame_id}:image_missing")

    # trajectory gpkg
    warnings: List[str] = []
    gis_dir = ensure_dir(run_dir / "gis")
    traj_layers = []
    if len(traj_cam0) >= 2:
        traj_layers.append(
            ("traj_cam0", gpd.GeoDataFrame([{"params_hash": params_hash, "geometry": LineString(traj_cam0)}], crs="EPSG:32632"))
        )
    if len(traj_velo_cam) >= 2:
        traj_layers.append(
            ("traj_velo_from_cam", gpd.GeoDataFrame([{"params_hash": params_hash, "geometry": LineString(traj_velo_cam)}], crs="EPSG:32632"))
        )
    if len(traj_velo_imu) >= 2:
        traj_layers.append(
            ("traj_velo_from_imu", gpd.GeoDataFrame([{"params_hash": params_hash, "geometry": LineString(traj_velo_imu)}], crs="EPSG:32632"))
        )

    if pose_rows:
        frame_points = gpd.GeoDataFrame(
            pose_rows,
            geometry=[gpd.points_from_xy([r["cam0_x"]], [r["cam0_y"]])[0] for r in pose_rows],
            crs="EPSG:32632",
        )
        traj_layers.append(("frame_points", frame_points))

    gpkg_path = gis_dir / "pose_audit_utm32.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()
    for layer, gdf in traj_layers:
        validate_output_crs(gpkg_path, 32632, gdf, warnings)
        gdf.to_file(gpkg_path, layer=layer, driver="GPKG")

    # delta stats table
    if delta_t_list:
        delta_rows = [
            {"metric": "delta_t_p50", "value": float(np.percentile(delta_t_list, 50)), "params_hash": params_hash},
            {"metric": "delta_t_p90", "value": float(np.percentile(delta_t_list, 90)), "params_hash": params_hash},
            {"metric": "delta_t_p99", "value": float(np.percentile(delta_t_list, 99)), "params_hash": params_hash},
            {"metric": "delta_t_max", "value": float(np.max(delta_t_list)), "params_hash": params_hash},
            {"metric": "delta_r_p50_deg", "value": float(np.percentile(delta_r_list, 50)), "params_hash": params_hash},
            {"metric": "delta_r_p90_deg", "value": float(np.percentile(delta_r_list, 90)), "params_hash": params_hash},
            {"metric": "delta_r_p99_deg", "value": float(np.percentile(delta_r_list, 99)), "params_hash": params_hash},
            {"metric": "delta_r_max_deg", "value": float(np.max(delta_r_list)), "params_hash": params_hash},
        ]
        write_csv(run_dir / "tables" / "pose_delta_stats.csv", delta_rows, ["metric", "value", "params_hash"])

    if pose_rows:
        write_csv(
            run_dir / "tables" / "pose_delta.csv",
            pose_rows,
            ["frame_id", "cam0_x", "cam0_y", "cam0_z", "roll", "pitch", "yaw", "delta_t_m", "delta_rot_deg", "params_hash"],
        )

    # overlay images
    overlay_dir = ensure_dir(run_dir / "overlays")
    overlay_frames = _sample_frames(frame_start, frame_end, int(cfg["SAMPLE_FRAMES_FOR_OVERLAY"]))
    overlay_stats = []
    if img_dir is not None:
        for frame_id in overlay_frames:
            img_path = _find_image_path(img_dir, frame_id)
            if img_path is None:
                continue
            try:
                pts_raw = load_kitti360_lidar_points(data_root, drive_id, frame_id)
                pts_world = load_kitti360_lidar_points_world_full(data_root, drive_id, frame_id, cam_id=str(cfg["OUTPUT_IMAGE_CAM"]))
                x, y, z, roll, pitch, yaw = load_kitti360_pose_full(data_root, drive_id, frame_id)
            except Exception:
                continue
            img = None
            try:
                import matplotlib.pyplot as plt

                img = plt.imread(img_path)
            except Exception:
                img = None
            if img is None:
                continue
            n_total, n_in_total, ratio_total = _project_stats(
                pts_world, (x, y, z, roll, pitch, yaw), calib, img
            )
            ratio = _write_overlay(
                img_path=img_path,
                points_world=pts_world,
                pose=(x, y, z, roll, pitch, yaw),
                calib=calib,
                out_path=overlay_dir / f"frame_{frame_id}.png",
                params_hash=params_hash,
                frame_id=frame_id,
                max_points=int(cfg["LIDAR_MAX_POINTS_FOR_OVERLAY"]),
            )

            ground_only_enable = bool(cfg.get("GROUND_ONLY_ENABLE", True))
            ground_ratio_pre = ""
            ground_ratio = 0.0
            n_ground = 0
            n_in_ground = 0
            ratio_ground = 0.0
            ground_z_est = ""
            dz_used = ""
            ground_method = ""
            n_proj_candidates = 0
            ratio_ground_v2 = 0.0
            ground_ratio_pre_v2 = ""
            ground_z_est_v2 = ""
            dz_used_v2 = ""
            seed_count = 0
            inlier_count = 0
            inlier_ratio = ""
            plane_nx = ""
            plane_ny = ""
            plane_nz = ""
            dz_plane = ""
            ground_ratio_pre_v3 = ""
            ratio_ground_v3 = 0.0
            status_v3 = ""
            if ground_only_enable and pts_raw.size > 0 and pts_world.size > 0:
                n = min(int(pts_raw.shape[0]), int(pts_world.shape[0]))
                pts_raw = pts_raw[:n]
                pts_world = pts_world[:n]
                z_src = pts_raw[:, 2] if str(cfg.get("GROUND_XYZ_FRAME", "velo")).lower() == "velo" else pts_world[:, 2]

                ground_z = float(np.percentile(z_src, float(cfg["GROUND_PCTL"])))
                dz = float(cfg["GROUND_BAND_DZ_M"])
                ground_mask = np.abs(z_src - ground_z) <= dz
                dz_used_val = dz
                while np.sum(ground_mask) < 1000 and dz_used_val < 0.30:
                    dz_used_val = round(dz_used_val + 0.05, 2)
                    ground_mask = np.abs(z_src - ground_z) <= dz_used_val
                ground_z_est = ground_z
                dz_used = dz_used_val
                n_ground = int(np.sum(ground_mask))
                ground_ratio = float(n_ground) / max(1, n_total)
                ground_ratio_pre = ground_ratio
                pts_ground_world = pts_world[ground_mask]
                if pts_ground_world.shape[0] > int(cfg["GROUND_SAMPLE_MAX_POINTS"]):
                    rng = np.random.default_rng(0)
                    sel = rng.choice(pts_ground_world.shape[0], size=int(cfg["GROUND_SAMPLE_MAX_POINTS"]), replace=False)
                    pts_ground_world = pts_ground_world[sel]
                if pts_ground_world.size > 0:
                    n_ground, n_in_ground, ratio_ground = _project_stats(
                        pts_ground_world, (x, y, z, roll, pitch, yaw), calib, img
                    )
                    title_suffix = f"in_image_ratio_ground={ratio_ground:.3f} | ground_ratio_pre={ground_ratio:.3f} | dz={dz_used_val:.2f}"
                    _write_overlay(
                        img_path=img_path,
                        points_world=pts_ground_world,
                        pose=(x, y, z, roll, pitch, yaw),
                        calib=calib,
                        out_path=overlay_dir / f"overlay_ground_only_frame_{frame_id}.png",
                        params_hash=params_hash,
                        frame_id=frame_id,
                        max_points=int(cfg["GROUND_SAMPLE_MAX_POINTS"]),
                        color="yellow",
                        title_suffix=title_suffix,
                    )

                proj_pts_cam = _velo_to_cam(pts_raw, calib)
                if proj_pts_cam.size > 0:
                    x_cam = proj_pts_cam[:, 0]
                    y_cam = proj_pts_cam[:, 1]
                    z_cam = proj_pts_cam[:, 2]
                    dist_xz = np.sqrt(x_cam**2 + z_cam**2)
                    w = img.shape[1]
                    h = img.shape[0]
                    margin = float(cfg["PROJ_MARGIN_RATIO"])
                    u = calib["k"][0, 0] * x_cam / np.maximum(z_cam, 1e-6) + calib["k"][0, 2]
                    v = calib["k"][1, 1] * y_cam / np.maximum(z_cam, 1e-6) + calib["k"][1, 2]
                    front_mask = (
                        (z_cam > 0)
                        & (dist_xz >= float(cfg["PROJ_DIST_MIN_M"]))
                        & (dist_xz <= float(cfg["PROJ_DIST_MAX_M"]))
                        & (u >= -margin * w)
                        & (u <= (1.0 + margin) * w)
                        & (v >= -margin * h)
                        & (v <= (1.0 + margin) * h)
                    )
                    n_proj_candidates = int(np.sum(front_mask))
                    z_proj = z_src[front_mask]
                    if z_proj.size > 0:
                        ground_z_v2, method, share = _estimate_ground_v2(
                            z_proj,
                            bin_m=float(cfg["GROUND_MODE_BIN_M"]),
                            min_share=float(cfg["GROUND_MODE_MIN_SHARE"]),
                            pctl_fallback=float(cfg["GROUND_PCTL_FALLBACK"]),
                        )
                        ground_method = method
                        ground_z_est_v2 = ground_z_v2
                        dz_v2 = float(cfg["GROUND_DZ_INIT_M"])
                        ground_mask_v2 = np.abs(z_src - ground_z_v2) <= dz_v2
                        while np.sum(ground_mask_v2) < 1500 and dz_v2 < 0.35:
                            dz_v2 = round(dz_v2 + 0.05, 2)
                            ground_mask_v2 = np.abs(z_src - ground_z_v2) <= dz_v2
                        while np.sum(ground_mask_v2) > 40000 and dz_v2 > 0.15:
                            dz_v2 = round(dz_v2 - 0.05, 2)
                            ground_mask_v2 = np.abs(z_src - ground_z_v2) <= dz_v2
                        dz_used_v2 = dz_v2
                        pts_ground_world_v2 = pts_world[ground_mask_v2]
                        ground_ratio_pre_v2 = float(np.sum(ground_mask_v2)) / max(1, n_total)
                        if pts_ground_world_v2.shape[0] > int(cfg["GROUND_SAMPLE_MAX_POINTS"]):
                            rng = np.random.default_rng(0)
                            sel = rng.choice(
                                pts_ground_world_v2.shape[0], size=int(cfg["GROUND_SAMPLE_MAX_POINTS"]), replace=False
                            )
                            pts_ground_world_v2 = pts_ground_world_v2[sel]
                        if pts_ground_world_v2.size > 0:
                            _, _, ratio_ground_v2 = _project_stats(
                                pts_ground_world_v2, (x, y, z, roll, pitch, yaw), calib, img
                            )
                            title_suffix = (
                                f"in_image_ratio_ground={ratio_ground_v2:.3f} | ground_ratio_pre={ground_ratio_pre_v2:.3f} | "
                                f"method={method} | dz={dz_v2:.2f}"
                            )
                            _write_overlay(
                                img_path=img_path,
                                points_world=pts_ground_world_v2,
                                pose=(x, y, z, roll, pitch, yaw),
                                calib=calib,
                                out_path=overlay_dir / f"overlay_ground_only_v2_frame_{frame_id}.png",
                                params_hash=params_hash,
                                frame_id=frame_id,
                                max_points=int(cfg["GROUND_SAMPLE_MAX_POINTS"]),
                                color="yellow",
                                title_suffix=title_suffix,
                            )

                if bool(cfg.get("GROUND_V3_ENABLE", True)):
                    w = img.shape[1]
                    h = img.shape[0]
                    v_min = float(cfg["GROUND_V3_V_MIN_RATIO"])
                    v_relax = float(cfg["GROUND_V3_V_MIN_RELAX"])
                    u_min = float(cfg["GROUND_V3_U_MIN_RATIO"])
                    u_max = float(cfg["GROUND_V3_U_MAX_RATIO"])
                    z_cam = proj_pts_cam[:, 2]
                    x_cam = proj_pts_cam[:, 0]
                    y_cam = proj_pts_cam[:, 1]
                    dist_xz = np.sqrt(x_cam**2 + z_cam**2)
                    u, v, _, _ = project_points_cam0_to_image(proj_pts_cam, calib, (h, w), use_rect=True, y_flip=True)
                    margin = float(cfg["PROJ_MARGIN_RATIO"])
                    in_img_ext = (
                        (z_cam > 0)
                        & (dist_xz >= float(cfg["PROJ_DIST_MIN_M"]))
                        & (dist_xz <= float(cfg["PROJ_DIST_MAX_M"]))
                        & (u >= -margin * w)
                        & (u <= (1.0 + margin) * w)
                        & (v >= -margin * h)
                        & (v <= (1.0 + margin) * h)
                    )
                    seed_mask = (
                        in_img_ext
                        & (v >= v_min * h)
                        & (u >= u_min * w)
                        & (u <= u_max * w)
                    )
                    seed_count = int(np.sum(seed_mask))
                    if seed_count < int(cfg["GROUND_V3_SEED_MIN"]):
                        seed_mask = in_img_ext & (v >= v_relax * h)
                        seed_count = int(np.sum(seed_mask))
                    if seed_count < int(cfg["GROUND_V3_SEED_MIN"]):
                        status_v3 = "seed_insufficient"
                    else:
                        seed_pts = proj_pts_cam[seed_mask]
                        n, d, inliers = _ransac_plane(
                            seed_pts,
                            iters=int(cfg["GROUND_V3_RANSAC_ITERS"]),
                            dist_thresh=float(cfg["GROUND_V3_INLIER_D_M"]),
                            normal_min_y=float(cfg["GROUND_V3_NORMAL_MIN_Y"]),
                        )
                        if n is None:
                            status_v3 = "plane_fit_fail"
                        else:
                            inlier_count = int(np.sum(inliers))
                            inlier_ratio = float(inlier_count) / max(1, seed_pts.shape[0])
                            plane_nx, plane_ny, plane_nz = float(n[0]), float(n[1]), float(n[2])
                            dz_plane_val = float(cfg["GROUND_V3_DZ_PLANE_INIT"])
                            dz_plane = dz_plane_val
                            all_mask = in_img_ext
                            pts_all_cam = proj_pts_cam[all_mask]
                            if pts_all_cam.size == 0:
                                status_v3 = "no_proj_points"
                            else:
                                dist_all = np.abs(pts_all_cam @ n + float(d))
                                ground_mask_v3 = dist_all <= dz_plane_val
                                if np.sum(ground_mask_v3) < 1500 and dz_plane_val < 0.30:
                                    dz_plane_val = 0.15
                                    ground_mask_v3 = dist_all <= dz_plane_val
                                if np.sum(ground_mask_v3) > 40000 and dz_plane_val > 0.10:
                                    dz_plane_val = 0.10
                                    ground_mask_v3 = dist_all <= dz_plane_val
                                dz_plane = dz_plane_val
                                pts_ground_world_v3 = pts_world[all_mask][ground_mask_v3]
                                ground_ratio_pre_v3 = float(np.sum(ground_mask_v3)) / max(1, int(np.sum(all_mask)))
                                if pts_ground_world_v3.shape[0] > int(cfg["GROUND_SAMPLE_MAX_POINTS"]):
                                    rng = np.random.default_rng(0)
                                    sel = rng.choice(
                                        pts_ground_world_v3.shape[0],
                                        size=int(cfg["GROUND_SAMPLE_MAX_POINTS"]),
                                        replace=False,
                                    )
                                    pts_ground_world_v3 = pts_ground_world_v3[sel]
                                if pts_ground_world_v3.size > 0:
                                    n_g, n_in_g, ratio_ground_v3 = _project_stats(
                                        pts_ground_world_v3, (x, y, z, roll, pitch, yaw), calib, img
                                    )
                                    status_v3 = "ok"
                                    title_suffix = (
                                        f"in_image_ratio_ground={ratio_ground_v3:.3f} | ground_ratio_pre={ground_ratio_pre_v3:.3f} | "
                                        f"seed={seed_count} | inlier={inlier_ratio:.2f} | dz={dz_plane_val:.2f}"
                                    )
                                    _write_overlay(
                                        img_path=img_path,
                                        points_world=pts_ground_world_v3,
                                        pose=(x, y, z, roll, pitch, yaw),
                                        calib=calib,
                                        out_path=overlay_dir / f"overlay_ground_only_v3_frame_{frame_id}.png",
                                        params_hash=params_hash,
                                        frame_id=frame_id,
                                        max_points=int(cfg["GROUND_SAMPLE_MAX_POINTS"]),
                                        color="yellow",
                                        title_suffix=title_suffix,
                                    )

            overlay_stats.append(
                {
                    "frame_id": frame_id,
                    "n_points_total": n_total,
                    "n_points_in_image_total": n_in_total,
                    "in_image_ratio_total": ratio_total,
                    "n_points_ground": n_ground,
                    "n_points_in_image_ground": n_in_ground,
                    "in_image_ratio_ground": ratio_ground,
                    "ground_ratio_pre": ground_ratio_pre,
                    "ground_z_est": ground_z_est,
                    "dz_used": dz_used,
                    "ground_method": ground_method,
                    "ground_z_est_v2": ground_z_est_v2,
                    "dz_used_v2": dz_used_v2,
                    "n_proj_candidates": n_proj_candidates,
                    "in_image_ratio_ground_v2": ratio_ground_v2,
                    "ground_ratio_pre_v2": ground_ratio_pre_v2,
                    "seed_count": seed_count,
                    "inlier_count": inlier_count,
                    "inlier_ratio": inlier_ratio,
                    "plane_nx": plane_nx,
                    "plane_ny": plane_ny,
                    "plane_nz": plane_nz,
                    "dz_plane": dz_plane,
                    "ground_ratio_pre_v3": ground_ratio_pre_v3,
                    "in_image_ratio_ground_v3": ratio_ground_v3,
                    "status_v3": status_v3,
                    "in_image_ratio": ratio,
                    "params_hash": params_hash,
                }
            )
    if overlay_stats:
        write_csv(
            run_dir / "tables" / "overlay_stats.csv",
            overlay_stats,
            [
                "frame_id",
                "n_points_total",
                "n_points_in_image_total",
                "in_image_ratio_total",
                "n_points_ground",
                "n_points_in_image_ground",
                "in_image_ratio_ground",
                "ground_ratio_pre",
                "ground_z_est",
                "dz_used",
                "ground_method",
                "ground_z_est_v2",
                "dz_used_v2",
                "n_proj_candidates",
                "in_image_ratio_ground_v2",
                "ground_ratio_pre_v2",
                "seed_count",
                "inlier_count",
                "inlier_ratio",
                "plane_nx",
                "plane_ny",
                "plane_nz",
                "dz_plane",
                "ground_ratio_pre_v3",
                "in_image_ratio_ground_v3",
                "status_v3",
                "in_image_ratio",
                "params_hash",
            ],
        )

    # timestamps check
    ts_cam = data_root / "data_2d_raw" / drive_id / str(cfg["OUTPUT_IMAGE_CAM"]) / "timestamps.txt"
    ts_velo = data_root / "data_3d_raw" / drive_id / "velodyne_points" / "timestamps.txt"
    ts_rows = []
    if ts_cam.exists() and ts_velo.exists():
        cam_lines = ts_cam.read_text(encoding="utf-8").splitlines()
        velo_lines = ts_velo.read_text(encoding="utf-8").splitlines()
        for i, frame_id in enumerate(frame_ids):
            try:
                t_cam = float(cam_lines[int(frame_id)])
                t_velo = float(velo_lines[int(frame_id)])
            except Exception:
                continue
            ts_rows.append({"frame_id": frame_id, "delta_ms": (t_cam - t_velo) * 1000.0, "params_hash": params_hash})
        if ts_rows:
            write_csv(run_dir / "tables" / "timestamp_delta.csv", ts_rows, ["frame_id", "delta_ms", "params_hash"])

    # decision
    status = "PASS"
    reason = "ok"
    overlay_ratio_median = float(np.median([r["in_image_ratio"] for r in overlay_stats])) if overlay_stats else 0.0
    ground_ratio_vals = [float(r["ground_ratio_pre"]) for r in overlay_stats if r.get("ground_ratio_pre") not in ("", None)]
    in_img_ground_vals = [float(r["in_image_ratio_ground"]) for r in overlay_stats if r.get("in_image_ratio_ground") not in ("", None)]
    ground_ratio_p10 = float(np.percentile(ground_ratio_vals, 10)) if ground_ratio_vals else 0.0
    ground_ratio_p50 = float(np.percentile(ground_ratio_vals, 50)) if ground_ratio_vals else 0.0
    ground_ratio_p90 = float(np.percentile(ground_ratio_vals, 90)) if ground_ratio_vals else 0.0
    in_img_ground_med = float(np.median(in_img_ground_vals)) if in_img_ground_vals else 0.0
    ground_ratio_vals_v2 = [float(r["ground_ratio_pre_v2"]) for r in overlay_stats if r.get("ground_ratio_pre_v2") not in ("", None)]
    in_img_ground_vals_v2 = [float(r["in_image_ratio_ground_v2"]) for r in overlay_stats if r.get("in_image_ratio_ground_v2") not in ("", None)]
    ground_ratio_p50_v2 = float(np.percentile(ground_ratio_vals_v2, 50)) if ground_ratio_vals_v2 else 0.0
    in_img_ground_med_v2 = float(np.median(in_img_ground_vals_v2)) if in_img_ground_vals_v2 else 0.0
    zero_in_img_v1 = sum(1 for r in overlay_stats if float(r.get("in_image_ratio_ground") or 0.0) <= 0.0)
    zero_in_img_v2 = sum(1 for r in overlay_stats if float(r.get("in_image_ratio_ground_v2") or 0.0) <= 0.0)
    if overlay_stats and overlay_ratio_median < 0.05:
        status = "FAIL"
        reason = "projection_out_of_image"
    if delta_t_list:
        p90_dt = float(np.percentile(delta_t_list, 90))
        p90_dr = float(np.percentile(delta_r_list, 90))
        if p90_dt > 3.0 or p90_dr > 5.0:
            status = "FAIL"
            reason = "pose_chain_mismatch"
        elif p90_dt > 1.5 or p90_dr > 3.0:
            if status != "FAIL":
                status = "WARN"
                reason = "pose_chain_warn"
    if missing_frames and status == "PASS":
        status = "WARN"
        reason = "missing_frames"

    decision = {
        "status": status,
        "reason": reason,
        "drive_id": drive_id,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "params_hash": params_hash,
    }
    write_json(run_dir / "decision.json", decision)

    # report
    report = [
        "# Pose Audit Report",
        "",
        f"- drive_id: {drive_id}",
        f"- frame_range: {frame_start}-{frame_end}",
        f"- params_hash: {params_hash}",
        f"- status: {status}",
        f"- reason: {reason}",
        "",
        "## Detected Paths",
        f"- kitti_root: {data_root}",
        f"- oxts_dir: {oxts_dir}",
        f"- velodyne_dir: {velodyne_dir}",
        f"- image_dir: {img_dir}",
        f"- cam_pose_file: {cam_pose_file or ''}",
        f"- cam_pose_dir: {cam_pose_dir_str}",
        f"- cam_pose_direction: {cam_pose_direction}",
        f"- cam_to_pose_key: {cam_to_pose_key}",
        f"- imu_to_velo: {imu_to_velo_src}",
        "",
        "## Direction Notes",
        f"- cam0_source: {cam0_source}",
        f"- cam0_from_oxts: T_w_pose * inv(T_cam_to_pose)",
        f"- cam0_from_pose_file: {'used' if cam0_source == 'pose_file' else 'not_used_or_missing'}",
        "",
        "## Delta Stats (velo_from_cam vs velo_from_imu)",
    ]
    if delta_t_list:
        report += [
            f"- delta_t_p50: {float(np.percentile(delta_t_list, 50)):.3f} m",
            f"- delta_t_p90: {float(np.percentile(delta_t_list, 90)):.3f} m",
            f"- delta_t_p99: {float(np.percentile(delta_t_list, 99)):.3f} m",
            f"- delta_t_max: {float(np.max(delta_t_list)):.3f} m",
            f"- delta_r_p50: {float(np.percentile(delta_r_list, 50)):.3f} deg",
            f"- delta_r_p90: {float(np.percentile(delta_r_list, 90)):.3f} deg",
            f"- delta_r_p99: {float(np.percentile(delta_r_list, 99)):.3f} deg",
            f"- delta_r_max: {float(np.max(delta_r_list)):.3f} deg",
        ]
    else:
        report.append("- delta_stats: not_available (missing imu_to_velo or pose)")

    if cam_pose_delta_t:
        report += [
            "",
            "## Delta Stats (cam0_from_oxts vs cam0_from_pose_file)",
            f"- cam_pose_delta_t_p90: {float(np.percentile(cam_pose_delta_t, 90)):.3f} m",
            f"- cam_pose_delta_r_p90: {float(np.percentile(cam_pose_delta_r, 90)):.3f} deg",
        ]

    report += ["", "## Overlay Stats", f"- overlay_count: {len(overlay_stats)}", f"- overlay_ratio_median: {overlay_ratio_median:.3f}"]

    report += [
        "",
        "## Ground-only overlay sanity check",
        f"- ground_ratio_pre_p10: {ground_ratio_p10:.4f}",
        f"- ground_ratio_pre_p50: {ground_ratio_p50:.4f}",
        f"- ground_ratio_pre_p90: {ground_ratio_p90:.4f}",
        f"- in_image_ratio_ground_median: {in_img_ground_med:.4f}",
    ]
    if ground_ratio_p90 < 0.01:
        report.append("- 판단: ground_ratio_pre 极低，可能存在筛选/ROI/坐标问题")
    elif in_img_ground_med < 0.05:
        report.append("- 判断: ground_ratio_pre 正常但图内比例偏低，可能为视角遮挡或近场盲区")
    else:
        report.append("- 判断: ground-only 投影分布正常")

    report += [
        "",
        "## Ground-only overlay v1 vs v2",
        f"- in_image_ratio_ground_median: {in_img_ground_med:.4f} -> {in_img_ground_med_v2:.4f}",
        f"- ground_ratio_pre_median: {ground_ratio_p50:.4f} -> {ground_ratio_p50_v2:.4f}",
        f"- zero_in_image_frames: {zero_in_img_v1} -> {zero_in_img_v2}",
        f"- ground_only_version: {cfg.get('GROUND_ONLY_VERSION')}",
    ]

    in_img_ground_vals_v3 = [
        float(r["in_image_ratio_ground_v3"]) for r in overlay_stats if r.get("in_image_ratio_ground_v3") not in ("", None)
    ]
    ground_ratio_vals_v3 = [
        float(r["ground_ratio_pre_v3"]) for r in overlay_stats if r.get("ground_ratio_pre_v3") not in ("", None)
    ]
    in_img_ground_med_v3 = float(np.median(in_img_ground_vals_v3)) if in_img_ground_vals_v3 else 0.0
    ground_ratio_p50_v3 = float(np.percentile(ground_ratio_vals_v3, 50)) if ground_ratio_vals_v3 else 0.0
    zero_in_img_v3 = sum(1 for r in overlay_stats if float(r.get("in_image_ratio_ground_v3") or 0.0) <= 0.0)
    report += [
        "",
        "## Ground-only overlay v2 vs v3",
        f"- in_image_ratio_ground_median: {in_img_ground_med_v2:.4f} -> {in_img_ground_med_v3:.4f}",
        f"- ground_ratio_pre_median: {ground_ratio_p50_v2:.4f} -> {ground_ratio_p50_v3:.4f}",
        f"- zero_in_image_frames: {zero_in_img_v2} -> {zero_in_img_v3}",
    ]

    if missing_frames:
        report += ["", "## Missing Frames", *[f"- {m}" for m in missing_frames]]

    if warnings:
        report += ["", "## CRS Warnings", *[f"- {w}" for w in warnings]]

    write_text(run_dir / "report.md", "\n".join(report))

    if missing_frames:
        write_text(run_dir / "qa" / "failpack_frames.txt", "\n".join(missing_frames))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

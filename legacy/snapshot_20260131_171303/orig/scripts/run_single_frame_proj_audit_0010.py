from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.datasets.kitti360_io import (
    _find_velodyne_dir,
    load_kitti360_calib,
    load_kitti360_cam_to_pose,
    load_kitti360_lidar_points,
    load_kitti360_pose_full,
)
from pipeline.projection.projector import project_points_cam0_to_image
from scripts.pipeline_common import ensure_dir, ensure_overwrite, now_ts, setup_logging, write_csv, write_json, write_text


REQUIRED_KEYS = [
    "DRIVE_MATCH",
    "FRAMES",
    "OUTPUT_IMAGE_CAM",
    "OVERWRITE",
    "TARGET_EPSG",
    "KITTI_ROOT",
    "MAX_POINTS_OVERLAY",
    "MAX_POINTS_GROUND",
    "GROUND_X_MIN_M",
    "GROUND_X_MAX_M",
    "GROUND_Y_ABS_MAX_M",
    "GROUND_RANSAC_ITERS",
    "GROUND_INLIER_D_M",
    "GROUND_NORMAL_MIN_Z",
    "PROJ_MARGIN_RATIO",
    "RECT_FALLBACK_ENABLE",
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
        "DRIVE_MATCH": "_0010_",
        "FRAMES": [341, 250, 500],
        "OUTPUT_IMAGE_CAM": "image_00",
        "OVERWRITE": True,
        "TARGET_EPSG": 32632,
        "KITTI_ROOT": "",
        "MAX_POINTS_OVERLAY": 120000,
        "MAX_POINTS_GROUND": 80000,
        "GROUND_X_MIN_M": 3.0,
        "GROUND_X_MAX_M": 45.0,
        "GROUND_Y_ABS_MAX_M": 20.0,
        "GROUND_RANSAC_ITERS": 200,
        "GROUND_INLIER_D_M": 0.12,
        "GROUND_NORMAL_MIN_Z": 0.9,
        "PROJ_MARGIN_RATIO": 0.2,
        "RECT_FALLBACK_ENABLE": True,
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
        key = str(frame_id)
        out[key] = mat
        if key.isdigit():
            out[f"{int(key):010d}"] = mat
    return out, direction


def _ransac_plane(points: np.ndarray, iters: int, dist_thresh: float, normal_min_z: float) -> Tuple[Optional[np.ndarray], Optional[float], np.ndarray]:
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
        if abs(float(n[2])) < float(normal_min_z):
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
    # refine plane with inliers
    inlier_pts = points[best_inliers]
    centroid = np.mean(inlier_pts, axis=0)
    uu, ss, vv = np.linalg.svd(inlier_pts - centroid)
    n = vv[-1, :]
    n = n / max(1e-6, float(np.linalg.norm(n)))
    if n[2] < 0:
        n = -n
    d = -float(np.dot(n, centroid))
    dist = np.abs(points @ n + d)
    inliers = dist < float(dist_thresh)
    return n, d, inliers


def _transform_points(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([points[:, :3], np.ones((points.shape[0], 1), dtype=points.dtype)])
    out = (mat @ pts_h.T).T
    return out[:, :3]




def _score_variant(
    v_norm: np.ndarray,
    in_img_ground: float,
    ground_bottom_ratio: float,
) -> float:
    median_v = float(np.median(v_norm)) if v_norm.size > 0 else 0.0
    score = 3.0 * in_img_ground + 3.0 * ground_bottom_ratio + 2.0 * max(0.0, min(1.0, (median_v - 0.5) / 0.5))
    if median_v < 0.5:
        score -= 2.0
    return score


def _overlay_plot(
    img_path: Path,
    u_all: np.ndarray,
    v_all: np.ndarray,
    in_all: np.ndarray,
    u_ground: np.ndarray,
    v_ground: np.ndarray,
    in_ground: np.ndarray,
    out_path: Path,
    title: str,
    params_hash: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = plt.imread(img_path)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.imshow(img)
    if np.any(in_all):
        ax.scatter(u_all[in_all], v_all[in_all], s=0.5, c="cyan", alpha=0.2)
    if np.any(in_ground):
        ax.scatter(u_ground[in_ground], v_ground[in_ground], s=1.2, c="yellow", alpha=0.7)
    ax.set_title(title)
    ax.text(0.01, 0.99, f"params_hash={params_hash}", transform=ax.transAxes, fontsize=6, color="white", va="top")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    base_cfg = _load_yaml(Path("configs/single_frame_proj_audit_0010.yaml"))
    run_dir = Path("runs") / f"single_frame_proj_audit_0010_{now_ts()}"
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
    img_dir = _find_image_dir(data_root, drive_id, str(cfg["OUTPUT_IMAGE_CAM"]))
    if img_dir is None:
        write_text(run_dir / "report.md", "missing_image_dir")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_image_dir", "params_hash": params_hash})
        return 2

    try:
        velodyne_dir = _find_velodyne_dir(data_root, drive_id)
    except Exception as exc:
        write_text(run_dir / "report.md", f"missing_velodyne_dir:{exc}")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_velodyne_dir", "params_hash": params_hash})
        return 2

    cam_pose_file = _find_cam_pose_file(data_root, drive_id)
    if cam_pose_file is None:
        write_text(run_dir / "report.md", "missing_cam_pose_file")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_cam_pose_file", "params_hash": params_hash})
        return 2
    cam_pose_map, cam_pose_direction = _parse_pose_file(cam_pose_file)

    try:
        calib = load_kitti360_calib(data_root, str(cfg["OUTPUT_IMAGE_CAM"]))
        cam_to_pose = load_kitti360_cam_to_pose(data_root, str(cfg["OUTPUT_IMAGE_CAM"]))
    except Exception as exc:
        write_text(run_dir / "report.md", f"missing_calib:{exc}")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_calib", "params_hash": params_hash})
        return 2

    frames = [int(f) for f in cfg["FRAMES"]]
    frame_ids = [f"{f:010d}" for f in frames]
    variants = []
    for pose_v in ["P0", "P1"]:
        for ext_v in ["E0", "E1"]:
            for y_v in ["Y0", "Y1"]:
                variants.append({"pose": pose_v, "extrinsic": ext_v, "yflip": y_v})

    rows = []
    overlay_dir = ensure_dir(run_dir / "overlays")
    best_by_frame = {}

    for frame_id in frame_ids:
        img_path = _find_image_path(img_dir, frame_id)
        if img_path is None:
            rows.append({"frame_id": frame_id, "variant": "NA", "status": "missing_image", "params_hash": params_hash})
            continue
        try:
            points_velo = load_kitti360_lidar_points(data_root, drive_id, frame_id)
        except Exception:
            rows.append({"frame_id": frame_id, "variant": "NA", "status": "missing_velodyne", "params_hash": params_hash})
            continue

        if points_velo.size == 0:
            rows.append({"frame_id": frame_id, "variant": "NA", "status": "empty_points", "params_hash": params_hash})
            continue

        img = None
        try:
            import matplotlib.pyplot as plt

            img = plt.imread(img_path)
        except Exception:
            img = None
        if img is None:
            rows.append({"frame_id": frame_id, "variant": "NA", "status": "image_read_fail", "params_hash": params_hash})
            continue

        h, w = img.shape[0], img.shape[1]
        margin = float(cfg["PROJ_MARGIN_RATIO"])

        # ground fit in LiDAR frame
        pts = points_velo[:, :3].astype(np.float64)
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        mask = (
            (x >= float(cfg["GROUND_X_MIN_M"]))
            & (x <= float(cfg["GROUND_X_MAX_M"]))
            & (np.abs(y) <= float(cfg["GROUND_Y_ABS_MAX_M"]))
        )
        seed_pts = pts[mask]
        n_ground, d_ground, inliers = _ransac_plane(
            seed_pts,
            iters=int(cfg["GROUND_RANSAC_ITERS"]),
            dist_thresh=float(cfg["GROUND_INLIER_D_M"]),
            normal_min_z=float(cfg["GROUND_NORMAL_MIN_Z"]),
        )
        if n_ground is None:
            rows.append({"frame_id": frame_id, "variant": "NA", "status": "ground_fit_fail", "params_hash": params_hash})
            continue
        inlier_mask = np.zeros((pts.shape[0],), dtype=bool)
        seed_idx = np.where(mask)[0]
        if inliers.size == seed_idx.size:
            inlier_mask[seed_idx[inliers]] = True
        ground_pts = pts[inlier_mask]
        inlier_ratio = float(np.sum(inlier_mask)) / max(1, int(np.sum(mask)))

        # prepare pose and extrinsic
        if frame_id not in cam_pose_map:
            rows.append({"frame_id": frame_id, "variant": "NA", "status": "missing_cam_pose_frame", "params_hash": params_hash})
            continue
        cam_pose = cam_pose_map[frame_id]
        if cam_pose_direction == "world_to_cam":
            cam_pose = np.linalg.inv(cam_pose)
        # cam_pose is normalized to cam->world
        cam_pose_raw = cam_pose
        try:
            xw, yw, zw, roll, pitch, yaw = load_kitti360_pose_full(data_root, drive_id, frame_id)
        except Exception:
            rows.append({"frame_id": frame_id, "variant": "NA", "status": "missing_oxts_pose", "params_hash": params_hash})
            continue
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

        t_cam_to_velo = calib["t_cam_to_velo"]
        t_velo_to_cam = np.linalg.inv(t_cam_to_velo)

        # sample points for overlay
        if pts.shape[0] > int(cfg["MAX_POINTS_OVERLAY"]):
            rng = np.random.default_rng(0)
            sel = rng.choice(pts.shape[0], size=int(cfg["MAX_POINTS_OVERLAY"]), replace=False)
            pts_overlay = pts[sel]
        else:
            pts_overlay = pts
        if ground_pts.shape[0] > int(cfg["MAX_POINTS_GROUND"]):
            rng = np.random.default_rng(1)
            sel = rng.choice(ground_pts.shape[0], size=int(cfg["MAX_POINTS_GROUND"]), replace=False)
            ground_overlay = ground_pts[sel]
        else:
            ground_overlay = ground_pts

        rect_default = True
        rect_alt_needed = False
        best_score = -1e9
        best_variant = None

        for use_rect in [True]:
            r_rect = calib.get("r_rect") if use_rect else None
            for var in variants:
                pose_v = var["pose"]
                ext_v = var["extrinsic"]
                y_v = var["yflip"]

                t_velo_to_cam_v = t_velo_to_cam if ext_v == "E0" else t_cam_to_velo
                t_pose_velo = cam_to_pose @ t_velo_to_cam_v
                t_w_cam_var = cam_pose_raw if pose_v == "P0" else np.linalg.inv(cam_pose_raw)
                t_c_w_var = np.linalg.inv(t_w_cam_var)

                pts_pose = _transform_points(pts_overlay, t_pose_velo)
                pts_world = (r_world_pose @ pts_pose.T).T + np.array([xw, yw, zw], dtype=float)
                pts_cam = _transform_points(pts_world, t_c_w_var)
                u_all, v_all, valid_all, in_img_all = project_points_cam0_to_image(
                    pts_cam, calib, (h, w), use_rect=bool(use_rect), y_flip=(y_v == "Y1")
                )
                in_ratio_all = float(np.sum(in_img_all)) / max(1, pts_cam.shape[0])

                g_pose = _transform_points(ground_overlay, t_pose_velo)
                g_world = (r_world_pose @ g_pose.T).T + np.array([xw, yw, zw], dtype=float)
                g_cam = _transform_points(g_world, t_c_w_var)
                u_g, v_g, valid_g, in_img_g = project_points_cam0_to_image(
                    g_cam, calib, (h, w), use_rect=bool(use_rect), y_flip=(y_v == "Y1")
                )
                in_ratio_g = float(np.sum(in_img_g)) / max(1, g_cam.shape[0])
                v_norm = v_g[in_img_g] / max(1, float(h))
                ground_bottom_ratio = float(np.sum(v_norm > 0.60)) / max(1, v_norm.size)
                ground_median = float(np.median(v_norm)) if v_norm.size > 0 else 0.0
                ground_band = float(np.max(v_norm) - np.min(v_norm)) if v_norm.size > 0 else 0.0
                score = _score_variant(v_norm, in_ratio_g, ground_bottom_ratio)

                variant_id = f"{pose_v}_{ext_v}_{y_v}_{'R' if use_rect else 'NR'}"
                rows.append(
                    {
                        "frame_id": frame_id,
                        "variant": variant_id,
                        "pose_variant": pose_v,
                        "extrinsic_variant": ext_v,
                        "y_flip": y_v,
                        "rect_used": use_rect,
                        "in_image_ratio_all": in_ratio_all,
                        "in_image_ratio_ground": in_ratio_g,
                        "ground_bottom_ratio": ground_bottom_ratio,
                        "ground_median_v_norm": ground_median,
                        "ground_band_thickness": ground_band,
                        "ground_inlier_ratio": inlier_ratio,
                        "score": score,
                        "params_hash": params_hash,
                    }
                )

                title = (
                    f"frame {frame_id} | {variant_id} | score={score:.2f} | in_g={in_ratio_g:.2f} | "
                    f"bottom={ground_bottom_ratio:.2f} | med_v={ground_median:.2f}"
                )
                _overlay_plot(
                    img_path,
                    u_all,
                    v_all,
                    in_img_all,
                    u_g,
                    v_g,
                    in_img_g,
                    overlay_dir / f"frame_{frame_id}_variant_{variant_id}.png",
                    title,
                    params_hash,
                )

                if score > best_score:
                    best_score = score
                    best_variant = (variant_id, use_rect, u_all, v_all, in_img_all, u_g, v_g, in_img_g)

            if best_score < 0.5 and bool(cfg.get("RECT_FALLBACK_ENABLE", True)):
                rect_alt_needed = True

        if rect_alt_needed:
            r_rect = None
            for var in variants:
                pose_v = var["pose"]
                ext_v = var["extrinsic"]
                y_v = var["yflip"]
                t_velo_to_cam_v = t_velo_to_cam if ext_v == "E0" else t_cam_to_velo
                t_pose_velo = cam_to_pose @ t_velo_to_cam_v
                t_w_cam_var = cam_pose_raw if pose_v == "P0" else np.linalg.inv(cam_pose_raw)
                t_c_w_var = np.linalg.inv(t_w_cam_var)

                pts_pose = _transform_points(pts_overlay, t_pose_velo)
                pts_world = (r_world_pose @ pts_pose.T).T + np.array([xw, yw, zw], dtype=float)
                pts_cam = _transform_points(pts_world, t_c_w_var)
                u_all, v_all, valid_all, in_img_all = project_points_cam0_to_image(
                    pts_cam, calib, (h, w), use_rect=False, y_flip=(y_v == "Y1")
                )
                in_ratio_all = float(np.sum(in_img_all)) / max(1, pts_cam.shape[0])

                g_pose = _transform_points(ground_overlay, t_pose_velo)
                g_world = (r_world_pose @ g_pose.T).T + np.array([xw, yw, zw], dtype=float)
                g_cam = _transform_points(g_world, t_c_w_var)
                u_g, v_g, valid_g, in_img_g = project_points_cam0_to_image(
                    g_cam, calib, (h, w), use_rect=False, y_flip=(y_v == "Y1")
                )
                in_ratio_g = float(np.sum(in_img_g)) / max(1, g_cam.shape[0])
                v_norm = v_g[in_img_g] / max(1, float(h))
                ground_bottom_ratio = float(np.sum(v_norm > 0.60)) / max(1, v_norm.size)
                ground_median = float(np.median(v_norm)) if v_norm.size > 0 else 0.0
                ground_band = float(np.max(v_norm) - np.min(v_norm)) if v_norm.size > 0 else 0.0
                score = _score_variant(v_norm, in_ratio_g, ground_bottom_ratio)
                variant_id = f"{pose_v}_{ext_v}_{y_v}_NR"
                rows.append(
                    {
                        "frame_id": frame_id,
                        "variant": variant_id,
                        "pose_variant": pose_v,
                        "extrinsic_variant": ext_v,
                        "y_flip": y_v,
                        "rect_used": False,
                        "in_image_ratio_all": in_ratio_all,
                        "in_image_ratio_ground": in_ratio_g,
                        "ground_bottom_ratio": ground_bottom_ratio,
                        "ground_median_v_norm": ground_median,
                        "ground_band_thickness": ground_band,
                        "ground_inlier_ratio": inlier_ratio,
                        "score": score,
                        "params_hash": params_hash,
                    }
                )
                title = (
                    f"frame {frame_id} | {variant_id} | score={score:.2f} | in_g={in_ratio_g:.2f} | "
                    f"bottom={ground_bottom_ratio:.2f} | med_v={ground_median:.2f}"
                )
                _overlay_plot(
                    img_path,
                    u_all,
                    v_all,
                    in_img_all,
                    u_g,
                    v_g,
                    in_img_g,
                    overlay_dir / f"frame_{frame_id}_variant_{variant_id}.png",
                    title,
                    params_hash,
                )
                if score > best_score:
                    best_score = score
                    best_variant = (variant_id, False, u_all, v_all, in_img_all, u_g, v_g, in_img_g)

        if best_variant is not None:
            variant_id, use_rect, u_all, v_all, in_img_all, u_g, v_g, in_img_g = best_variant
            title = f"frame {frame_id} | best={variant_id} | score={best_score:.2f}"
            _overlay_plot(
                img_path,
                u_all,
                v_all,
                in_img_all,
                u_g,
                v_g,
                in_img_g,
                overlay_dir / f"frame_{frame_id}_best.png",
                title,
                params_hash,
            )
            best_by_frame[frame_id] = {"variant": variant_id, "score": best_score}

    if rows:
        write_csv(
            run_dir / "variants.csv",
            rows,
            [
                "frame_id",
                "variant",
                "pose_variant",
                "extrinsic_variant",
                "y_flip",
                "rect_used",
                "in_image_ratio_all",
                "in_image_ratio_ground",
                "ground_bottom_ratio",
                "ground_median_v_norm",
                "ground_band_thickness",
                "ground_inlier_ratio",
                "score",
                "status",
                "params_hash",
            ],
        )

    # select best variant by mean score
    by_variant: Dict[str, List[float]] = {}
    for r in rows:
        if r.get("status"):
            continue
        vid = r["variant"]
        by_variant.setdefault(vid, []).append(float(r["score"]))
    best_variant = None
    best_score = -1e9
    for vid, vals in by_variant.items():
        s = float(np.mean(vals)) if vals else -1e9
        if s > best_score:
            best_score = s
            best_variant = vid

    best_payload = {"best_variant": best_variant, "mean_score": best_score, "by_frame": best_by_frame, "params_hash": params_hash}
    write_json(run_dir / "best_variant.json", best_payload)

    # report
    report = [
        "# Single Frame Projection Audit Report",
        "",
        f"- drive_id: {drive_id}",
        f"- frames: {frame_ids}",
        f"- params_hash: {params_hash}",
        f"- cam_pose_file: {cam_pose_file}",
        f"- cam_pose_direction: {cam_pose_direction}",
        f"- best_variant: {best_variant}",
        f"- best_mean_score: {best_score:.3f}",
        "",
        "## Conclusion",
    ]
    if best_variant:
        report.append(f"- 建议采用变体: {best_variant}")
        if "P1" in best_variant:
            report.append("- 提示: cam0_to_world 可能需要按 world->cam 使用（或相反）")
        if "E1" in best_variant:
            report.append("- 提示: cam_to_velo 方向可能需要取逆")
        if "Y1" in best_variant:
            report.append("- 提示: 相机 y 轴可能需要翻转")
        if best_variant.endswith("_NR"):
            report.append("- 提示: rectification 可能误用或应禁用")
    else:
        report.append("- 未找到稳定变体，建议检查相机内参/图像尺寸与帧对齐")

    write_text(run_dir / "report.md", "\n".join(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

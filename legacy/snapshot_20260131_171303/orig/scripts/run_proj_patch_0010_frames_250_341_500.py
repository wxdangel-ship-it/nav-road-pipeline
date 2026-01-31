from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.datasets.kitti360_io import _find_velodyne_dir, load_kitti360_lidar_points
from pipeline.calib.io_kitti360_calib import load_kitti360_calib_bundle
from pipeline.calib.kitti360_projection import project_velo_to_image
from scripts.pipeline_common import ensure_dir, ensure_overwrite, now_ts, setup_logging, write_csv, write_json, write_text


REQUIRED_KEYS = [
    "FRAMES",
    "VARIANT_FIXED",
    "OUTPUT_IMAGE_CAM",
    "OVERLAY_MAX_POINTS",
    "NON_GROUND_SAMPLE",
    "GROUND_FIT_ENABLE",
    "GROUND_PLANE_DZ",
    "SEED_V_MIN_RATIO",
    "SEED_U_CENTER_RATIO",
    "RANGE_MIN_M",
    "RANGE_MAX_M",
    "OVERWRITE",
    "TARGET_EPSG",
    "KITTI_ROOT",
    "VIZPACK_ENABLE",
    "VIZPACK_V2_ENABLE",
    "VIZPACK_V2_MAX_ALL",
    "VIZPACK_V2_MAX_GROUND",
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
        "FRAMES": [250, 341, 500],
        "VARIANT_FIXED": "P0_E0_Y1_R",
        "OUTPUT_IMAGE_CAM": "image_00",
        "OVERLAY_MAX_POINTS": 120000,
        "NON_GROUND_SAMPLE": 20000,
        "GROUND_FIT_ENABLE": True,
        "GROUND_PLANE_DZ": 0.12,
        "SEED_V_MIN_RATIO": 0.65,
        "SEED_U_CENTER_RATIO": [0.2, 0.8],
        "RANGE_MIN_M": 3.0,
        "RANGE_MAX_M": 45.0,
        "OVERWRITE": True,
        "TARGET_EPSG": 32632,
        "KITTI_ROOT": "",
        "VIZPACK_ENABLE": True,
        "VIZPACK_V2_ENABLE": True,
        "VIZPACK_V2_MAX_ALL": 200000,
        "VIZPACK_V2_MAX_GROUND": 120000,
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


def _overlay_plot(img_path: Path, u: np.ndarray, v: np.ndarray, in_img: np.ndarray, out_path: Path, title: str, params_hash: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = plt.imread(img_path)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.imshow(img)
    if np.any(in_img):
        ax.scatter(u[in_img], v[in_img], s=1.0, c="yellow", alpha=0.7)
    ax.set_title(title)
    ax.text(0.01, 0.99, f"params_hash={params_hash}", transform=ax.transAxes, fontsize=6, color="white", va="top")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _overlay_plot_dual(
    img_path: Path,
    u_ground: np.ndarray,
    v_ground: np.ndarray,
    in_ground: np.ndarray,
    u_non: np.ndarray,
    v_non: np.ndarray,
    in_non: np.ndarray,
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
    if np.any(in_non):
        ax.scatter(u_non[in_non], v_non[in_non], s=0.8, c="green", alpha=0.3)
    if np.any(in_ground):
        ax.scatter(u_ground[in_ground], v_ground[in_ground], s=1.2, c="red", alpha=0.8)
    ax.set_title(title)
    ax.text(0.01, 0.99, f"params_hash={params_hash}", transform=ax.transAxes, fontsize=6, color="white", va="top")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _overlay_scatter(
    img: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    in_img: np.ndarray,
    out_path: Path,
    title: str,
    params_hash: str,
    color: str,
    alpha: float,
    size: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.imshow(img)
    if np.any(in_img):
        ax.scatter(u[in_img], v[in_img], s=size, c=color, alpha=alpha)
    ax.set_title(title)
    ax.text(0.01, 0.99, f"params_hash={params_hash}", transform=ax.transAxes, fontsize=6, color="white", va="top")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _overlay_density(
    img: np.ndarray,
    ui: np.ndarray,
    vi: np.ndarray,
    out_path: Path,
    title: str,
    params_hash: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = img.shape[0], img.shape[1]
    counts = np.zeros((h, w), dtype=np.int32)
    if ui.size > 0:
        np.add.at(counts, (vi, ui), 1)
    heat = np.log1p(counts).astype(np.float32)
    vmax = float(np.max(heat)) if heat.size else 0.0
    if vmax > 0:
        heat = heat / vmax
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.imshow(img)
    ax.imshow(heat, cmap="inferno", alpha=0.5, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.text(0.01, 0.99, f"params_hash={params_hash}", transform=ax.transAxes, fontsize=6, color="white", va="top")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _overlay_zbuffer(
    img: np.ndarray,
    ui: np.ndarray,
    vi: np.ndarray,
    depth: np.ndarray,
    out_path: Path,
    title: str,
    params_hash: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = img.shape[0], img.shape[1]
    depth_img = np.full((h * w,), np.inf, dtype=np.float32)
    if ui.size > 0:
        flat = vi * w + ui
        np.minimum.at(depth_img, flat, depth.astype(np.float32))
    depth_img = depth_img.reshape(h, w)
    mask = np.isfinite(depth_img)
    if np.any(mask):
        dmin = float(np.min(depth_img[mask]))
        dmax = float(np.max(depth_img[mask]))
        if dmax > dmin:
            depth_norm = (depth_img - dmin) / (dmax - dmin)
        else:
            depth_norm = np.zeros_like(depth_img)
    else:
        depth_norm = np.zeros_like(depth_img)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.imshow(img)
    ax.imshow(depth_norm, cmap="viridis", alpha=0.6, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.text(0.01, 0.99, f"params_hash={params_hash}", transform=ax.transAxes, fontsize=6, color="white", va="top")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_ply_xyz(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if points.size == 0:
        path.write_text("ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n", encoding="utf-8")
        return
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def _md5_short(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def main() -> int:
    base_cfg = _load_yaml(Path("configs/proj_patch_0010.yaml"))
    run_dir = Path("runs") / f"proj_patch_0010_{now_ts()}"
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

    drive_id = _select_drive(data_root, "_0010_")
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

    try:
        calib = load_kitti360_calib_bundle(
            data_root,
            drive_id,
            cam_id=str(cfg["OUTPUT_IMAGE_CAM"]),
            frame_id_for_size=frame_ids[0] if frame_ids else None,
        )
    except Exception as exc:
        write_text(run_dir / "report.md", f"missing_calib:{exc}")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_calib", "params_hash": params_hash})
        return 2

    frames = [int(f) for f in cfg["FRAMES"]]
    frame_ids = [f"{f:010d}" for f in frames]
    overlay_dir = ensure_dir(run_dir / "overlays")
    vizpack_root = ensure_dir(run_dir / "vizpack") if bool(cfg.get("VIZPACK_ENABLE", True)) else None
    vizpack_v2_root = ensure_dir(run_dir / "vizpack_v2") if bool(cfg.get("VIZPACK_V2_ENABLE", True)) else None
    vizpack_v2_fixed_root = ensure_dir(run_dir / "vizpack_v2_fixed") if bool(cfg.get("VIZPACK_V2_ENABLE", True)) else None
    table_rows = []
    per_frame_status = {}

    for frame_id in frame_ids:
        img_path = _find_image_path(img_dir, frame_id)
        if img_path is None:
            table_rows.append({"frame_id": frame_id, "any_warning": "missing_image", "params_hash": params_hash})
            per_frame_status[frame_id] = "FAIL"
            continue
        try:
            points_velo = load_kitti360_lidar_points(data_root, drive_id, frame_id)
        except Exception:
            table_rows.append({"frame_id": frame_id, "any_warning": "missing_velodyne", "params_hash": params_hash})
            per_frame_status[frame_id] = "FAIL"
            continue

        import matplotlib.pyplot as plt

        img = plt.imread(img_path)
        h, w = img.shape[0], img.shape[1]
        cx, cy = float(calib.k[0, 2]), float(calib.k[1, 2])
        if cx <= 0 or cy <= 0 or cx >= w or cy >= h:
            write_text(
                run_dir / "report.md",
                f"intrinsics_size_mismatch: cx={cx:.2f}, cy={cy:.2f}, w={w}, h={h}",
            )
            write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "intrinsics_size_mismatch", "params_hash": params_hash})
            return 2

        pts = points_velo[:, :3].astype(np.float64)
        if pts.shape[0] > int(cfg["OVERLAY_MAX_POINTS"]):
            rng = np.random.default_rng(0)
            sel = rng.choice(pts.shape[0], size=int(cfg["OVERLAY_MAX_POINTS"]), replace=False)
            pts_overlay = pts[sel]
        else:
            pts_overlay = pts

        warning = ""
        proj_all = project_velo_to_image(pts_overlay, calib, use_rect=True, y_flip_mode="fixed_true")
        u_all = proj_all["u"]
        v_all = proj_all["v"]
        in_img_all = proj_all["in_img"]
        in_ratio_all = float(np.sum(in_img_all)) / max(1, pts_overlay.shape[0])
        if in_ratio_all < 0.01:
            warning = "projection_zero"

        # ground fit in LiDAR frame
        ground_seed_count = 0
        ground_inlier_ratio = 0.0
        ground_ratio = 0.0
        in_ratio_ground = 0.0
        ground_bottom_ratio = 0.0
        ground_median_v = 0.0
        status = "PASS"
        n = None
        d = None
        if bool(cfg["GROUND_FIT_ENABLE"]):
            x = pts[:, 0]
            y = pts[:, 1]
            mask = (
                (x >= float(cfg["RANGE_MIN_M"]))
                & (x <= float(cfg["RANGE_MAX_M"]))
                & (np.abs(y) <= 20.0)
            )
            seed_pts = pts[mask]
            ground_seed_count = int(seed_pts.shape[0])
            n, d, inliers = _ransac_plane(
                seed_pts,
                iters=200,
                dist_thresh=float(cfg["GROUND_PLANE_DZ"]),
                normal_min_z=0.9,
            )
            if n is None:
                warning = "plane_fit_fail"
            else:
                ground_inlier_ratio = float(np.sum(inliers)) / max(1, ground_seed_count)
                all_dist = np.abs(pts @ n + d)
                ground_mask = all_dist <= float(cfg["GROUND_PLANE_DZ"])
                ground_ratio = float(np.sum(ground_mask)) / max(1, pts.shape[0])
                g_pts = pts[ground_mask]
                proj_g = project_velo_to_image(g_pts, calib, use_rect=True, y_flip_mode="fixed_true")
                u_g = proj_g["u"]
                v_g = proj_g["v"]
                in_img_g = proj_g["in_img"]
                in_ratio_ground = float(np.sum(in_img_g)) / max(1, g_pts.shape[0])
                v_norm = v_g[in_img_g] / max(1, float(h))
                ground_bottom_ratio = float(np.sum(v_norm > 0.60)) / max(1, v_norm.size)
                ground_median_v = float(np.median(v_norm)) if v_norm.size > 0 else 0.0

                _overlay_plot(
                    img_path,
                    u_g,
                    v_g,
                    in_img_g,
                    overlay_dir / f"frame_{frame_id}_ground.png",
                    f"frame {frame_id} | variant=P0_E0_Y1_R | in_g={in_ratio_ground:.2f} | bottom={ground_bottom_ratio:.2f}",
                    params_hash,
                )
                _overlay_plot(
                    img_path,
                    u_all,
                    v_all,
                    in_img_all,
                    overlay_dir / f"frame_{frame_id}_all.png",
                    f"frame {frame_id} | variant=P0_E0_Y1_R | in_all={in_ratio_all:.2f}",
                    params_hash,
                )
                _overlay_plot(
                    img_path,
                    np.concatenate([u_all, u_g]),
                    np.concatenate([v_all, v_g]),
                    np.concatenate([in_img_all, in_img_g]),
                    overlay_dir / f"frame_{frame_id}_all_ground.png",
                    f"frame {frame_id} | variant=P0_E0_Y1_R | in_g={in_ratio_ground:.2f} | bottom={ground_bottom_ratio:.2f}",
                    params_hash,
                )
        else:
            warning = "ground_fit_disabled"

        if in_ratio_ground < 0.20 or ground_bottom_ratio < 0.65:
            status = "FAIL"

        if frame_id.endswith("500") and status == "FAIL":
            _overlay_plot(
                img_path,
                u_all,
                v_all,
                in_img_all,
                overlay_dir / f"frame_{frame_id}_debug_scatter.png",
                f"frame {frame_id} | debug_scatter",
                params_hash,
            )

        if vizpack_root is not None:
            frame_dir = ensure_dir(vizpack_root / f"frame_{frame_id}")
            shutil.copy2(img_path, frame_dir / "image_00.png")
            # build ground/non-ground uv
            ground_mask = np.zeros((pts.shape[0],), dtype=bool)
            if bool(cfg["GROUND_FIT_ENABLE"]) and ground_inlier_ratio > 0 and n is not None and d is not None:
                all_dist = np.abs(pts @ n + d)
                ground_mask = all_dist <= float(cfg["GROUND_PLANE_DZ"])
            ng_mask = ~ground_mask
            ng_idx = np.where(ng_mask)[0]
            if ng_idx.size > int(cfg["NON_GROUND_SAMPLE"]):
                rng = np.random.default_rng(2)
                ng_idx = rng.choice(ng_idx, size=int(cfg["NON_GROUND_SAMPLE"]), replace=False)
            g_idx = np.where(ground_mask)[0]
            g_pts = pts[g_idx]
            ng_pts = pts[ng_idx]

            proj_g = project_velo_to_image(g_pts, calib, use_rect=True, y_flip_mode="fixed_true")
            u_g = proj_g["u"]
            v_g = proj_g["v"]
            in_img_g = proj_g["in_img"]

            proj_ng = project_velo_to_image(ng_pts, calib, use_rect=True, y_flip_mode="fixed_true")
            u_ng = proj_ng["u"]
            v_ng = proj_ng["v"]
            in_img_ng = proj_ng["in_img"]

            ground_rows = []
            for u, v, zc, in_img in zip(u_g, v_g, proj_g["z_cam"], in_img_g):
                ground_rows.append({"u": float(u), "v": float(v), "depth": float(zc), "in_image": int(in_img)})
            nonground_rows = []
            for u, v, zc, in_img in zip(u_ng, v_ng, proj_ng["z_cam"], in_img_ng):
                nonground_rows.append({"u": float(u), "v": float(v), "depth": float(zc), "in_image": int(in_img)})
            write_csv(frame_dir / "points_ground_uv.csv", ground_rows, ["u", "v", "depth", "in_image"])
            write_csv(frame_dir / "points_nonground_uv.csv", nonground_rows, ["u", "v", "depth", "in_image"])

            g_h = np.hstack([g_pts[:, :3], np.ones((g_pts.shape[0], 1), dtype=g_pts.dtype)])
            g_cam = (calib.t_c0_v @ g_h.T).T[:, :3]
            _write_ply_xyz(frame_dir / "points_ground_cam0.ply", g_cam)

            title_ground = (
                f"frame {frame_id} | variant=P0_E0_Y1_R | in_g={in_ratio_ground:.2f} | bottom={ground_bottom_ratio:.2f}"
            )
            _overlay_plot(frame_dir / "image_00.png", u_g, v_g, in_img_g, frame_dir / "overlay_ground.png", title_ground, params_hash)

            title_all = f"frame {frame_id} | variant=P0_E0_Y1_R | in_all={in_ratio_all:.2f}"
            _overlay_plot_dual(
                frame_dir / "image_00.png",
                u_g,
                v_g,
                in_img_g,
                u_ng,
                v_ng,
                in_img_ng,
                frame_dir / "overlay_all.png",
                title_all,
                params_hash,
            )

            # meta
            cam_ts = data_root / "data_2d_raw" / drive_id / str(cfg["OUTPUT_IMAGE_CAM"]) / "timestamps.txt"
            velo_ts = data_root / "data_3d_raw" / drive_id / "velodyne_points" / "timestamps.txt"
            delta_ms = ""
            if cam_ts.exists() and velo_ts.exists():
                cam_lines = cam_ts.read_text(encoding="utf-8").splitlines()
                velo_lines = velo_ts.read_text(encoding="utf-8").splitlines()
                try:
                    t_cam = float(cam_lines[int(frame_id)])
                    t_velo = float(velo_lines[int(frame_id)])
                    delta_ms = (t_cam - t_velo) * 1000.0
                except Exception:
                    delta_ms = ""
            meta = {
                "frame_id": frame_id,
                "variant": "P0_E0_Y1_R",
                "image_path": str(img_path),
                "lidar_path": str(Path(velodyne_dir) / f"{frame_id}.bin"),
                "image_bytes": int(Path(img_path).stat().st_size),
                "image_mtime": float(Path(img_path).stat().st_mtime),
                "lidar_bytes": int(Path(velodyne_dir).joinpath(f"{frame_id}.bin").stat().st_size),
                "lidar_mtime": float(Path(velodyne_dir).joinpath(f"{frame_id}.bin").stat().st_mtime),
                "image_md5_8": _md5_short(img_path),
                "lidar_md5_8": _md5_short(Path(velodyne_dir) / f"{frame_id}.bin"),
                "delta_ms": delta_ms,
                "in_image_ratio_all": in_ratio_all,
                "in_image_ratio_ground": in_ratio_ground,
                "ground_bottom_ratio": ground_bottom_ratio,
                "ground_inlier_ratio": ground_inlier_ratio,
                "params_hash": params_hash,
            }
            write_json(frame_dir / "meta.json", meta)

        if vizpack_v2_root is not None:
            frame_dir = ensure_dir(vizpack_v2_root / f"frame_{frame_id}")
            shutil.copy2(img_path, frame_dir / "image_00.png")

            pts_all = pts
            x_velo = pts[:, 0]
            lidar_range_mask = (x_velo >= float(cfg["RANGE_MIN_M"])) & (x_velo <= float(cfg["RANGE_MAX_M"]))
            cam_mask = lidar_range_mask
            fallback_mask = ""
            idx_all = np.where(cam_mask)[0]
            if idx_all.size == 0:
                cam_mask = np.ones((pts.shape[0],), dtype=bool)
                fallback_mask = "z_only"
                idx_all = np.where(cam_mask)[0]

            if idx_all.size > int(cfg["VIZPACK_V2_MAX_ALL"]):
                rng = np.random.default_rng(3)
                idx_all = rng.choice(idx_all, size=int(cfg["VIZPACK_V2_MAX_ALL"]), replace=False)
            pts_all_sel = pts_all[idx_all]

            ground_mask_all = np.zeros((pts.shape[0],), dtype=bool)
            if bool(cfg["GROUND_FIT_ENABLE"]) and ground_inlier_ratio > 0 and n is not None and d is not None:
                ground_mask_all = np.abs(pts @ n + d) <= float(cfg["GROUND_PLANE_DZ"])
            idx_ground = np.where(ground_mask_all & cam_mask)[0]
            if idx_ground.size > int(cfg["VIZPACK_V2_MAX_GROUND"]):
                rng = np.random.default_rng(4)
                idx_ground = rng.choice(idx_ground, size=int(cfg["VIZPACK_V2_MAX_GROUND"]), replace=False)
            pts_ground_sel = pts_all[idx_ground]

            proj_all = project_velo_to_image(pts_all_sel, calib, use_rect=True, y_flip_mode="fixed_true")
            u_all = proj_all["u"]
            v_all = proj_all["v"]
            in_img_all = proj_all["in_img"]
            proj_g = project_velo_to_image(pts_ground_sel, calib, use_rect=True, y_flip_mode="fixed_true")
            u_g = proj_g["u"]
            v_g = proj_g["v"]
            in_img_g = proj_g["in_img"]

            in_all = int(np.sum(in_img_all))
            in_g = int(np.sum(in_img_g))
            status_v2 = "PASS"
            reason_v2 = ""
            if in_all < 5000 or in_g < 2000:
                status_v2 = "FAIL"
                reason_v2 = "projection_low_in_image"

            title_all = f"frame {frame_id} | all | in={in_all}"
            title_g = f"frame {frame_id} | ground | in={in_g}"
            if status_v2 == "FAIL":
                title_all = f"FAIL {title_all}"
                title_g = f"FAIL {title_g}"

            _overlay_scatter(
                img,
                u_all,
                v_all,
                in_img_all,
                frame_dir / "overlay_all_scatter.png",
                title_all,
                params_hash,
                color="green",
                alpha=0.15,
                size=1.0,
            )
            _overlay_scatter(
                img,
                u_g,
                v_g,
                in_img_g,
                frame_dir / "overlay_ground_scatter.png",
                title_g,
                params_hash,
                color="yellow",
                alpha=0.25,
                size=2.0,
            )

            ui_all = u_all[in_img_all].astype(np.int32)
            vi_all = v_all[in_img_all].astype(np.int32)
            ui_g = u_g[in_img_g].astype(np.int32)
            vi_g = v_g[in_img_g].astype(np.int32)

            _overlay_density(
                img,
                ui_all,
                vi_all,
                frame_dir / "overlay_all_density.png",
                title_all,
                params_hash,
            )
            _overlay_density(
                img,
                ui_g,
                vi_g,
                frame_dir / "overlay_ground_density.png",
                title_g,
                params_hash,
            )

            depth_all = proj_all["z_cam"][in_img_all]
            depth_g = proj_g["z_cam"][in_img_g]
            _overlay_zbuffer(
                img,
                ui_all,
                vi_all,
                depth_all,
                frame_dir / "overlay_all_zbuffer.png",
                title_all,
                params_hash,
            )
            _overlay_zbuffer(
                img,
                ui_g,
                vi_g,
                depth_g,
                frame_dir / "overlay_ground_zbuffer.png",
                title_g,
                params_hash,
            )

            all_rows = [{"u": float(u), "v": float(v), "depth": float(d)} for u, v, d in zip(u_all[in_img_all], v_all[in_img_all], depth_all)]
            ground_rows = [{"u": float(u), "v": float(v), "depth": float(d)} for u, v, d in zip(u_g[in_img_g], v_g[in_img_g], depth_g)]
            write_csv(frame_dir / "uv_all.csv", all_rows, ["u", "v", "depth"])
            write_csv(frame_dir / "uv_ground.csv", ground_rows, ["u", "v", "depth"])

            meta_v2 = {
                "frame_id": frame_id,
                "variant": "P0_E0_Y1_R",
                "n_all": int(pts_all_sel.shape[0]),
                "n_all_in_image": in_all,
                "n_ground": int(pts_ground_sel.shape[0]),
                "n_ground_in_image": in_g,
                "in_image_ratio_all": float(in_all) / max(1, int(pts_all_sel.shape[0])),
                "in_image_ratio_ground": float(in_g) / max(1, int(pts_ground_sel.shape[0])),
                "status": status_v2,
                "reason": reason_v2,
                "cam_mask_fallback": fallback_mask,
                "params_hash": params_hash,
            }

            if frame_id.endswith("500") and status_v2 == "FAIL":
                meta_v2.update(
                    {
                        "image_path": str(img_path),
                        "lidar_path": str(Path(velodyne_dir) / f"{frame_id}.bin"),
                        "image_bytes": int(Path(img_path).stat().st_size),
                        "image_mtime": float(Path(img_path).stat().st_mtime),
                        "lidar_bytes": int(Path(velodyne_dir).joinpath(f"{frame_id}.bin").stat().st_size),
                        "lidar_mtime": float(Path(velodyne_dir).joinpath(f"{frame_id}.bin").stat().st_mtime),
                        "image_md5_8": _md5_short(img_path),
                        "lidar_md5_8": _md5_short(Path(velodyne_dir) / f"{frame_id}.bin"),
                        "image_size": f"{w}x{h}",
                        "intrinsics": f"{calib.k[0,2]:.2f},{calib.k[1,2]:.2f}",
                    }
                )
                _overlay_scatter(
                    img,
                    u_g,
                    v_g,
                    in_img_g,
                    frame_dir / "overlay_ground_scatter_FAIL.png",
                    title_g,
                    params_hash,
                    color="yellow",
                    alpha=0.25,
                    size=2.0,
                )
            write_json(frame_dir / "meta.json", meta_v2)

        if vizpack_v2_fixed_root is not None and frame_id in {"0000000250", "0000000341"}:
            frame_dir = ensure_dir(vizpack_v2_fixed_root / f"frame_{frame_id}")
            shutil.copy2(img_path, frame_dir / "image_00.png")

            pts_all = pts
            ground_mask_all = np.zeros((pts.shape[0],), dtype=bool)
            if bool(cfg["GROUND_FIT_ENABLE"]) and ground_inlier_ratio > 0 and n is not None and d is not None:
                ground_mask_all = np.abs(pts @ n + d) <= float(cfg["GROUND_PLANE_DZ"])
            g_idx = np.where(ground_mask_all)[0]
            if g_idx.size > int(cfg["VIZPACK_V2_MAX_GROUND"]):
                rng = np.random.default_rng(5)
                g_idx = rng.choice(g_idx, size=int(cfg["VIZPACK_V2_MAX_GROUND"]), replace=False)
            g_pts = pts_all[g_idx]

            proj_fix = project_velo_to_image(g_pts, calib, use_rect=True, y_flip_mode="fixed_true")
            u = proj_fix["u"]
            v = proj_fix["v"]
            z_cam = proj_fix["z_cam"]
            in_img = proj_fix["in_img"]

            z_med = float(np.median(z_cam[valid])) if np.any(valid) else 0.0
            z_max = float(np.max(z_cam[valid])) if np.any(valid) else 0.0
            u_min = float(np.min(u[in_img])) if np.any(in_img) else 0.0
            u_max = float(np.max(u[in_img])) if np.any(in_img) else 0.0
            v_min = float(np.min(v[in_img])) if np.any(in_img) else 0.0
            v_max = float(np.max(v[in_img])) if np.any(in_img) else 0.0
            u_range = u_max - u_min
            v_range = v_max - v_min
            ui = np.round(u[in_img]).astype(np.int32)
            vi = np.round(v[in_img]).astype(np.int32)
            if ui.size > 0:
                unique_pix = int(np.unique(vi * w + ui).size)
            else:
                unique_pix = 0

            status_fix = "PASS"
            reason_fix = ""
            if z_med < 3.0 or z_med > 45.0 or z_max > 200.0:
                status_fix = "FAIL"
                reason_fix = "z_cam_invalid"
            if u_range <= 50.0 or v_range <= 30.0:
                status_fix = "FAIL"
                reason_fix = "uv_range_small"
            if unique_pix <= 1000:
                status_fix = "FAIL"
                reason_fix = "unique_pixel_low"

            title = f"frame {frame_id} | in={int(np.sum(in_img))} | uniq={unique_pix} | z_med={z_med:.2f}"
            if status_fix == "FAIL":
                title = f"FAIL {title}"
            _overlay_scatter(
                img,
                u,
                v,
                in_img,
                frame_dir / "overlay_ground_scatter.png",
                title,
                params_hash,
                color="yellow",
                alpha=0.25,
                size=2.0,
            )
            ui = u[in_img].astype(np.int32)
            vi = v[in_img].astype(np.int32)
            _overlay_density(
                img,
                ui,
                vi,
                frame_dir / "overlay_ground_density.png",
                title,
                params_hash,
            )

            rows = [{"u": float(uu), "v": float(vv), "z_cam": float(zz)} for uu, vv, zz in zip(u[in_img], v[in_img], z_cam[in_img])]
            write_csv(frame_dir / "uv_ground.csv", rows, ["u", "v", "z_cam"])

            meta_fix = {
                "frame_id": frame_id,
                "variant": "P0_E0_Y1_R",
                "n_ground": int(g_pts.shape[0]),
                "n_ground_in_image": int(np.sum(in_img)),
                "z_cam_median": z_med,
                "z_cam_max": z_max,
                "u_range": u_range,
                "v_range": v_range,
                "unique_pixel_count": unique_pix,
                "status": status_fix,
                "reason": reason_fix,
                "matrices": {
                "k": calib.k.tolist(),
                "r_rect": calib.r_rect_00.tolist() if calib.r_rect_00 is not None else None,
                "t_c0_v": calib.t_c0_v.tolist(),
                },
                "params_hash": params_hash,
            }
            write_json(frame_dir / "meta.json", meta_fix)

        table_rows.append(
            {
                "frame_id": frame_id,
                "image_path": str(img_path),
                "lidar_path": str(Path(velodyne_dir) / f"{frame_id}.bin"),
                "image_size": f"{w}x{h}",
                "image_bytes": int(Path(img_path).stat().st_size),
                "lidar_point_count": int(points_velo.shape[0]),
                "ground_seed_count": ground_seed_count,
                "ground_inlier_ratio": ground_inlier_ratio,
                "in_image_ratio_all": in_ratio_all,
                "in_image_ratio_ground": in_ratio_ground,
                "ground_bottom_ratio": ground_bottom_ratio,
                "ground_median_v_norm": ground_median_v,
                "intrinsics_used": f"{calib.k[0,0]:.2f},{calib.k[1,1]:.2f},{calib.k[0,2]:.2f},{calib.k[1,2]:.2f}",
                "any_warning": warning,
                "params_hash": params_hash,
            }
        )
        per_frame_status[frame_id] = status

    write_csv(
        run_dir / "tables" / "frame_diag.csv",
        table_rows,
        [
            "frame_id",
            "image_path",
            "lidar_path",
            "image_size",
            "image_bytes",
            "lidar_point_count",
            "ground_seed_count",
            "ground_inlier_ratio",
            "in_image_ratio_all",
            "in_image_ratio_ground",
            "ground_bottom_ratio",
            "ground_median_v_norm",
            "intrinsics_used",
            "any_warning",
            "params_hash",
        ],
    )

    status = "PASS"
    if per_frame_status.get("0000000500") == "FAIL" and per_frame_status.get("0000000250") == "PASS":
        status = "WARN"
    if any(v == "FAIL" for v in per_frame_status.values()):
        if status != "WARN":
            status = "FAIL"

    decision = {"status": status, "frames": frame_ids, "variant": "P0_E0_Y1_R", "params_hash": params_hash}
    write_json(run_dir / "decision.json", decision)

    report = [
        "# Projection Patch Report",
        "",
        f"- drive_id: {drive_id}",
        f"- frames: {frame_ids}",
        f"- variant: P0_E0_Y1_R",
        f"- status: {status}",
        f"- params_hash: {params_hash}",
        "",
        "## Frame Diagnostics",
    ]
    for frame_id in frame_ids:
        report.append(f"- {frame_id}: {per_frame_status.get(frame_id, 'NA')}")
    report += [
        "",
        "## Vizpack v2 Fixed (ground-only)",
        f"- output_dir: {str(run_dir / 'vizpack_v2_fixed')}",
        "- frames: 0000000250, 0000000341",
        "- checks: z_cam range, uv range, unique_pixel_count",
    ]
    if per_frame_status.get("0000000500") == "FAIL":
        report += [
            "",
            "## Frame 500 Diagnosis (ranked)",
            "1) 帧错位（图像/点云非同一时刻）",
            "2) 内参/rectification 读取异常",
            "3) 地面拟合失败或地面不占主导",
        ]
    write_text(run_dir / "report.md", "\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

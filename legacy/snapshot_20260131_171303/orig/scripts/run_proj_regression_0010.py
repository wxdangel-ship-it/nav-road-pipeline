from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.calib.io_kitti360_calib import load_kitti360_calib_bundle
from pipeline.calib.kitti360_projection import project_velo_to_image
from pipeline.calib.proj_sanity import (
    validate_depth,
    validate_in_image_ratio,
    validate_uv_spread,
)
from pipeline.datasets.kitti360_io import _find_velodyne_dir, load_kitti360_lidar_points
from scripts.pipeline_common import ensure_dir, ensure_overwrite, now_ts, setup_logging, write_csv, write_json, write_text


REQUIRED_KEYS = [
    "DRIVE_MATCH",
    "FRAMES",
    "USE_RECT",
    "OVERLAY_MAX_POINTS_ALL",
    "OVERLAY_MAX_POINTS_GROUND",
    "EXPORT_MODE",
    "GROUND_RANSAC_DZ",
    "OUTPUT_DIR",
    "OVERWRITE",
    "TARGET_EPSG",
    "KITTI_ROOT",
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
        "DRIVE_MATCH": "_0010_",
        "FRAMES": [250, 341, 500],
        "USE_RECT": True,
        "OVERLAY_MAX_POINTS_ALL": 200000,
        "OVERLAY_MAX_POINTS_GROUND": 120000,
        "EXPORT_MODE": "single_frame",
        "GROUND_RANSAC_DZ": 0.12,
        "OUTPUT_DIR": "",
        "OVERWRITE": True,
        "TARGET_EPSG": 32632,
        "KITTI_ROOT": "",
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


def _overlay_scatter(img: np.ndarray, u: np.ndarray, v: np.ndarray, in_img: np.ndarray, out_path: Path, title: str, params_hash: str, color: str, alpha: float, size: float) -> None:
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


def _overlay_density(img: np.ndarray, ui: np.ndarray, vi: np.ndarray, out_path: Path, title: str, params_hash: str) -> None:
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


def _overlay_zbuffer(img: np.ndarray, ui: np.ndarray, vi: np.ndarray, depth: np.ndarray, out_path: Path, title: str, params_hash: str) -> None:
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


def main() -> int:
    base_cfg = _load_yaml(Path("configs/proj_regression_0010.yaml"))
    run_dir = Path("runs") / f"proj_regression_0010_{now_ts()}"
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
    frames = [int(f) for f in cfg["FRAMES"]]
    frame_ids = [f"{f:010d}" for f in frames]

    img_dir = _find_image_dir(data_root, drive_id, "image_00")
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
        calib = load_kitti360_calib_bundle(data_root, drive_id, cam_id="image_00", frame_id_for_size=frame_ids[0])
    except Exception as exc:
        write_text(run_dir / "report.md", f"missing_calib:{exc}")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_calib", "params_hash": params_hash})
        return 2

    frames_dir = ensure_dir(run_dir / "frames")
    per_frame = []
    status_map = {}

    for frame_id in frame_ids:
        img_path = _find_image_path(img_dir, frame_id)
        if img_path is None:
            per_frame.append({"frame_id": frame_id, "status": "FAIL", "reason": "missing_image", "params_hash": params_hash})
            status_map[frame_id] = "FAIL"
            continue
        points = load_kitti360_lidar_points(data_root, drive_id, frame_id)
        pts = points[:, :3].astype(np.float64)

        import matplotlib.pyplot as plt

        img = plt.imread(img_path)
        h, w = img.shape[0], img.shape[1]

        # ground ransac in velo
        mask_seed = (pts[:, 0] >= 3.0) & (pts[:, 0] <= 45.0) & (np.abs(pts[:, 1]) <= 20.0)
        seed_pts = pts[mask_seed]
        n, d, inliers = _ransac_plane(seed_pts, iters=200, dist_thresh=float(cfg["GROUND_RANSAC_DZ"]), normal_min_z=0.9)
        ground_mask = np.zeros((pts.shape[0],), dtype=bool)
        if n is not None:
            dist = np.abs(pts @ n + d)
            ground_mask = dist <= float(cfg["GROUND_RANSAC_DZ"])

        idx_all = np.arange(pts.shape[0])
        if idx_all.size > int(cfg["OVERLAY_MAX_POINTS_ALL"]):
            rng = np.random.default_rng(0)
            idx_all = rng.choice(idx_all, size=int(cfg["OVERLAY_MAX_POINTS_ALL"]), replace=False)
        idx_ground = np.where(ground_mask)[0]
        if idx_ground.size > int(cfg["OVERLAY_MAX_POINTS_GROUND"]):
            rng = np.random.default_rng(1)
            idx_ground = rng.choice(idx_ground, size=int(cfg["OVERLAY_MAX_POINTS_GROUND"]), replace=False)

        pts_all = pts[idx_all]
        pts_ground = pts[idx_ground]

        proj_all = project_velo_to_image(pts_all, calib, use_rect=bool(cfg["USE_RECT"]), y_flip_mode="fixed_true", sanity=False)
        proj_ground = project_velo_to_image(pts_ground, calib, use_rect=bool(cfg["USE_RECT"]), y_flip_mode="fixed_true", sanity=False)

        u_all, v_all, in_all = proj_all["u"], proj_all["v"], proj_all["in_img"]
        u_g, v_g, in_g = proj_ground["u"], proj_ground["v"], proj_ground["in_img"]
        z_all = proj_all["z_cam"]
        z_g = proj_ground["z_cam"]

        status = "PASS"
        reason = ""
        stats = {}
        try:
            stats.update(validate_depth(z_g))
            stats.update(validate_uv_spread(u_g, v_g, in_g))
            stats.update(validate_in_image_ratio(in_g))
        except Exception as exc:
            status = "FAIL"
            reason = str(exc)

        frame_out = ensure_dir(frames_dir / frame_id)
        title_all = f"frame {frame_id} | all | in={int(np.sum(in_all))}"
        title_ground = f"frame {frame_id} | ground | in={int(np.sum(in_g))}"
        if status == "FAIL":
            title_all = f"FAIL {title_all}"
            title_ground = f"FAIL {title_ground}"

        _overlay_scatter(img, u_all, v_all, in_all, frame_out / "overlay_all_scatter.png", title_all, params_hash, "green", 0.15, 1.0)
        _overlay_scatter(img, u_g, v_g, in_g, frame_out / "overlay_ground_scatter.png", title_ground, params_hash, "yellow", 0.25, 2.0)

        ui_all = u_all[in_all].astype(np.int32)
        vi_all = v_all[in_all].astype(np.int32)
        ui_g = u_g[in_g].astype(np.int32)
        vi_g = v_g[in_g].astype(np.int32)

        _overlay_density(img, ui_all, vi_all, frame_out / "overlay_all_density.png", title_all, params_hash)
        _overlay_density(img, ui_g, vi_g, frame_out / "overlay_ground_density.png", title_ground, params_hash)

        _overlay_zbuffer(img, ui_all, vi_all, z_all[in_all], frame_out / "overlay_all_zbuffer.png", title_all, params_hash)
        _overlay_zbuffer(img, ui_g, vi_g, z_g[in_g], frame_out / "overlay_ground_zbuffer.png", title_ground, params_hash)

        rows_all = [{"u": float(u), "v": float(v), "z_cam": float(z)} for u, v, z in zip(u_all[in_all], v_all[in_all], z_all[in_all])]
        rows_ground = [{"u": float(u), "v": float(v), "z_cam": float(z)} for u, v, z in zip(u_g[in_g], v_g[in_g], z_g[in_g])]
        write_csv(frame_out / "uv_all.csv", rows_all, ["u", "v", "z_cam"])
        write_csv(frame_out / "uv_ground.csv", rows_ground, ["u", "v", "z_cam"])

        meta = {
            "frame_id": frame_id,
            "status": status,
            "reason": reason,
            "n_all": int(pts_all.shape[0]),
            "n_all_in_image": int(np.sum(in_all)),
            "n_ground": int(pts_ground.shape[0]),
            "n_ground_in_image": int(np.sum(in_g)),
            "params_hash": params_hash,
            "stats": stats,
        }
        write_json(frame_out / "meta.json", meta)

        per_frame.append(
            {
                "frame_id": frame_id,
                "status": status,
                "reason": reason,
                "n_all_in": int(np.sum(in_all)),
                "n_ground_in": int(np.sum(in_g)),
                "params_hash": params_hash,
            }
        )
        status_map[frame_id] = status

    write_csv(run_dir / "tables" / "frame_stats.csv", per_frame, ["frame_id", "status", "reason", "n_all_in", "n_ground_in", "params_hash"])

    status = "PASS"
    if status_map.get("0000000250") == "FAIL" or status_map.get("0000000341") == "FAIL":
        status = "FAIL"
    elif status_map.get("0000000500") == "FAIL":
        status = "WARN"
    decision = {"status": status, "frames": frame_ids, "params_hash": params_hash}
    write_json(run_dir / "decision.json", decision)

    report = [
        "# Projection Regression Report",
        "",
        f"- drive_id: {drive_id}",
        f"- frames: {frame_ids}",
        f"- status: {status}",
        f"- params_hash: {params_hash}",
        "",
        "## Frame Status",
    ]
    for row in per_frame:
        report.append(f"- {row['frame_id']}: {row['status']} ({row['reason']})")
    write_text(run_dir / "report.md", "\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

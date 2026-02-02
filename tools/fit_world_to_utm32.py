from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_csv, write_json, write_text

LOG = logging.getLogger("world_to_utm32_fit")

# =========================
# 参数区（按需修改）
# =========================
KITTI_ROOT = r"E:\KITTI360\KITTI-360"
DRIVE_ID = "2013_05_28_drive_0010_sync"
FRAME_START = 0
FRAME_END = 3835
MAX_SAMPLE_FRAMES = 300
SAMPLE_STRIDE = 10
RMS_PASS_M = 1.2
RMS_FAIL_M = 1.8
TRIM_TOP_PCT = 0.05


def _find_pose_file(data_root: Path, drive_id: str, names: List[str]) -> Path:
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
    raise FileNotFoundError(f"missing_pose_file:{names}")


def _find_oxts_dir(data_root: Path, drive_id: str) -> Path:
    candidates = [
        data_root / "data_poses_oxts" / drive_id / "oxts" / "data",
        data_root / "data_poses_oxts_extract" / drive_id / "oxts" / "data",
        data_root / "data_poses" / drive_id / "oxts" / "data",
        data_root / "data_poses" / "oxts" / drive_id / "oxts" / "data",
        data_root / "data_poses" / "oxts" / drive_id / "data",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"missing_oxts_dir:{drive_id}")


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


def _read_oxts_frame(oxts_dir: Path, frame_id: str) -> Tuple[float, float, float]:
    path = oxts_dir / f"{frame_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"missing_oxts:{path}")
    parts = path.read_text(encoding="utf-8").strip().split()
    if len(parts) < 6:
        raise ValueError("parse_error:oxts")
    lat = float(parts[0])
    lon = float(parts[1])
    alt = float(parts[2])
    return lat, lon, alt


def _fit_similarity_2d(src: np.ndarray, dst: np.ndarray) -> Tuple[float, float, float, float]:
    # src, dst: Nx2
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / float(src.shape[0])
    u, s, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    var_src = float(np.mean(np.sum(src_c**2, axis=1)))
    scale = float(np.sum(s) / var_src) if var_src > 0 else 1.0
    t = mu_dst - scale * (r @ mu_src)
    yaw = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return float(t[0]), float(t[1]), yaw, scale


def _apply_similarity_2d(src: np.ndarray, dx: float, dy: float, yaw_deg: float, scale: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    x = src[:, 0]
    y = src[:, 1]
    x2 = scale * (c * x - s * y) + dx
    y2 = scale * (s * x + c * y) + dy
    return np.stack([x2, y2], axis=1)


def main() -> int:
    data_root = Path(KITTI_ROOT)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root_missing:{data_root}")

    run_dir = Path("runs") / f"world_to_utm32_fit_0010_full_{now_ts()}"
    ensure_overwrite(run_dir)
    setup_logging(run_dir / "logs" / "run.log")
    LOG.info("run_start")

    cam0_to_world_path = _find_pose_file(data_root, DRIVE_ID, ["cam0_to_world.txt"])
    cam0_to_world = _parse_pose_map(cam0_to_world_path)
    oxts_dir = _find_oxts_dir(data_root, DRIVE_ID)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)

    full_ids = [f"{i:010d}" for i in range(int(FRAME_START), int(FRAME_END) + 1)]
    sample_ids = full_ids[:: max(1, int(SAMPLE_STRIDE))]

    pairs: List[Dict[str, object]] = []
    src_xy: List[List[float]] = []
    dst_xy: List[List[float]] = []
    dz_list: List[float] = []
    used_frames: List[str] = []

    for fid in sample_ids:
        if fid not in cam0_to_world:
            continue
        try:
            lat, lon, alt = _read_oxts_frame(oxts_dir, fid)
        except Exception:
            continue
        utm_e, utm_n = transformer.transform(lon, lat)
        t_w_c0 = cam0_to_world[fid]
        wx, wy, wz = float(t_w_c0[0, 3]), float(t_w_c0[1, 3]), float(t_w_c0[2, 3])
        src_xy.append([wx, wy])
        dst_xy.append([utm_e, utm_n])
        dz_list.append(float(alt - wz))
        used_frames.append(fid)

    if len(src_xy) < 3:
        raise RuntimeError("insufficient_common_frames")

    samples_total = len(src_xy)
    if samples_total > MAX_SAMPLE_FRAMES:
        idx = np.linspace(0, samples_total - 1, MAX_SAMPLE_FRAMES, dtype=int).tolist()
        src_xy = [src_xy[i] for i in idx]
        dst_xy = [dst_xy[i] for i in idx]
        dz_list = [dz_list[i] for i in idx]
        used_frames = [used_frames[i] for i in idx]

    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    dx0, dy0, yaw0, scale0 = _fit_similarity_2d(src, dst)
    dz = float(np.median(dz_list)) if dz_list else 0.0
    pred0 = _apply_similarity_2d(src, dx0, dy0, yaw0, scale0)
    residual0 = np.linalg.norm(pred0 - dst, axis=1)
    rms_raw = float(np.sqrt(np.mean(residual0**2)))

    keep = max(3, int(math.floor(len(residual0) * (1.0 - TRIM_TOP_PCT))))
    if keep < len(residual0):
        order = np.argsort(residual0)
        inlier_idx = order[:keep]
    else:
        inlier_idx = np.arange(len(residual0))
    src_in = src[inlier_idx]
    dst_in = dst[inlier_idx]
    dx, dy, yaw_deg, scale = _fit_similarity_2d(src_in, dst_in)
    pred1 = _apply_similarity_2d(src_in, dx, dy, yaw_deg, scale)
    residual1 = np.linalg.norm(pred1 - dst_in, axis=1)
    rms_inlier = float(np.sqrt(np.mean(residual1**2)))
    p95_inlier = float(np.percentile(residual1, 95)) if residual1.size else 0.0
    max_inlier = float(np.max(residual1)) if residual1.size else 0.0
    inlier_ratio = float(src_in.shape[0]) / float(src.shape[0])

    for i, fid in enumerate(used_frames):
        pairs.append(
            {
                "frame": fid,
                "world_x": float(src[i, 0]),
                "world_y": float(src[i, 1]),
                "utm_e": float(dst[i, 0]),
                "utm_n": float(dst[i, 1]),
                "residual": float(residual0[i]),
            }
        )

    gate_status = "PASS"
    if rms_inlier > RMS_FAIL_M:
        gate_status = "FAIL"
    elif rms_inlier > RMS_PASS_M:
        gate_status = "WARN"

    report = {
        "drive_id": DRIVE_ID,
        "frame_range": [FRAME_START, FRAME_END],
        "common_frames_used": int(len(src_xy)),
        "frame_span": [FRAME_START, FRAME_END],
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
        "yaw_deg": float(yaw_deg),
        "scale": float(scale),
        "rms_raw_m": float(rms_raw),
        "rms_inlier_m": float(rms_inlier),
        "p95_inlier_m": float(p95_inlier),
        "max_inlier_m": float(max_inlier),
        "inlier_ratio": float(inlier_ratio),
        "samples_total": int(samples_total),
        "samples_used": int(len(src_xy)),
        "outlier_rule": f"trim_top_pct_{TRIM_TOP_PCT}",
        "gate_status": gate_status,
        "gate_thresholds": {"pass": RMS_PASS_M, "fail": RMS_FAIL_M},
        "paths": {"cam0_to_world": str(cam0_to_world_path), "oxts_dir": str(oxts_dir)},
    }

    summary = (
        f"world->utm32 fit: frames={len(src_xy)}, rms_inlier={rms_inlier:.3f}m, scale={scale:.4f}, yaw={yaw_deg:.3f}deg "
        f"=> {gate_status}"
    )

    write_json(run_dir / "report" / "world_to_utm32_report.json", report)
    write_csv(
        run_dir / "report" / "world_to_utm32_pairs.csv",
        pairs,
        ["frame", "world_x", "world_y", "utm_e", "utm_n", "residual"],
    )
    write_text(run_dir / "report" / "world_to_utm32_summary.md", summary)

    LOG.info("run_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

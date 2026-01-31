from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline._io import load_yaml
from pipeline.calib.kitti360_backproject import (
    configure_default_context,
    pixel_to_ray_c0rect,
    ray_c0rect_to_ray_c0,
    pixel_to_world_on_ground,
    world_to_pixel_cam0,
)
from pipeline.calib.io_kitti360_calib import Kitti360Calib
from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_csv, write_json, write_text


def _find_data_root(cfg_root: str) -> Path:
    if cfg_root:
        path = Path(cfg_root)
        if path.exists():
            return path
    env_root = os.environ.get("POC_DATA_ROOT", "").strip()
    if env_root:
        path = Path(env_root)
        if path.exists():
            return path
    default_root = Path(r"E:\KITTI360\KITTI-360")
    if default_root.exists():
        return default_root
    raise SystemExit("missing data root: set POC_DATA_ROOT or config.kitti_root")


def _find_latest_stage12_run() -> Optional[Path]:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.glob("image_stage12_crosswalk_0010_250_500_*") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_latest_dtm() -> Optional[Path]:
    runs_dir = Path("runs")
    candidates = []
    for p in runs_dir.glob("lidar_ground_0010_f250_500_*"):
        if not p.is_dir():
            continue
        cand = p / "rasters" / "dtm_median_clean_utm32.tif"
        if cand.exists():
            candidates.append(cand)
        else:
            cand2 = p / "rasters" / "dtm_median_utm32.tif"
            if cand2.exists():
                candidates.append(cand2)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_image_dir(data_root: Path, drive: str, camera: str) -> Path:
    candidates = [
        data_root / "data_2d_raw" / drive / camera / "data_rect",
        data_root / "data_2d_raw" / drive / camera / "data",
        data_root / drive / camera / "data_rect",
        data_root / drive / camera / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"image data not found for drive={drive} camera={camera}")


def _find_frame_path(image_dir: Path, frame_id: str) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg"):
        path = image_dir / f"{frame_id}{ext}"
        if path.exists():
            return path
    return None


def _load_mask(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img) > 0


def _resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    h, w = mask.shape[:2]
    if (w, h) == size:
        return mask
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    resized = img.resize(size, resample=Image.NEAREST)
    return np.array(resized) > 0


def _dtm_median(dtm_path: Optional[Path]) -> Optional[float]:
    if dtm_path is None or not dtm_path.exists():
        return None
    try:
        import rasterio

        with rasterio.open(dtm_path) as ds:
            arr = ds.read(1, masked=True)
            if arr is None:
                return None
            data = np.array(arr, dtype=float)
            data = data[np.isfinite(data)]
            if data.size == 0:
                return None
            return float(np.median(data))
    except Exception:
        return None


def _dump_calib(calib: Kitti360Calib, t_w_c0: np.ndarray) -> Dict[str, object]:
    k_rect = calib.p_rect_00[:3, :3] if calib.p_rect_00 is not None else calib.k
    return {
        "image_size": [int(calib.image_size[0]), int(calib.image_size[1])],
        "K_rect": k_rect.tolist(),
        "R_rect_00": calib.r_rect_00.tolist() if calib.r_rect_00 is not None else None,
        "P_rect_00": calib.p_rect_00.tolist() if calib.p_rect_00 is not None else None,
        "T_W_C0": t_w_c0.tolist(),
        "T_W_C0_t": t_w_c0[:3, 3].tolist(),
        "T_W_C0_R": t_w_c0[:3, :3].tolist(),
    }


def _draw_overlay(img: Image.Image, pts: np.ndarray, reproj: np.ndarray, out_path: Path, title: str) -> None:
    canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for u, v in pts:
        draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=(255, 0, 0))
    for u, v in reproj:
        draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=(0, 255, 0))
    draw.text((8, 8), title, fill=(255, 255, 255))
    canvas.save(out_path)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/world_crosswalk_candidates_0010_f250_500.yaml")
    ap.add_argument("--stage12-run", default="")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    drive_id = str(cfg.get("drive_id") or "")
    image_cam = str(cfg.get("image_cam") or "image_00")
    frame_id = "0000000290"

    run_id = now_ts()
    run_dir = Path("runs") / f"backproject_cycle_gate_0010_f290_{run_id}"
    ensure_overwrite(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")

    if args.stage12_run:
        stage12_run = Path(args.stage12_run)
    else:
        stage12_run = _find_latest_stage12_run()
    if stage12_run is None or not stage12_run.exists():
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "stage12_run_not_found"})
        raise SystemExit("stage12 run not found")

    mask_dir = stage12_run / str(cfg.get("input_mask_dir") or "stage2/masks")
    mask_path = mask_dir / f"frame_{frame_id}.png"
    if not mask_path.exists():
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "mask_missing"})
        raise SystemExit("mask missing")

    data_root = _find_data_root(str(cfg.get("kitti_root") or ""))
    image_dir = _find_image_dir(data_root, drive_id, image_cam)
    img_path = _find_frame_path(image_dir, frame_id)
    if img_path is None or not img_path.exists():
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "image_missing"})
        raise SystemExit("image missing")

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    mask = _load_mask(mask_path)
    mask_size_before = (mask.shape[1], mask.shape[0])
    mask = _resize_mask(mask, (w, h))

    back_cfg = cfg.get("backproject") or {}
    dtm_path_cfg = str(back_cfg.get("dtm_path") or "auto_latest_clean_dtm")
    dtm_path = _find_latest_dtm() if dtm_path_cfg == "auto_latest_clean_dtm" else Path(dtm_path_cfg)
    dtm_median = _dtm_median(dtm_path)

    ctx = configure_default_context(data_root, drive_id, cam_id=image_cam, dtm_path=dtm_path, frame_id_for_size=frame_id)
    t_w_c0 = ctx.pose_provider.get_t_w_c0(frame_id)
    camera_height = float(back_cfg.get("camera_height_m", 1.6))
    origin_z = float(t_w_c0[2, 3])
    cam_z0 = origin_z - camera_height
    if dtm_median is not None and float(dtm_median) < origin_z:
        z0_use = float(dtm_median)
    else:
        z0_use = cam_z0

    ys, xs = np.where(mask)
    if len(xs) == 0:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "empty_mask"})
        raise SystemExit("empty mask")

    rng = random.Random(20260130)
    idxs = rng.sample(range(len(xs)), min(500, len(xs)))
    pts = np.array([[float(xs[i]), float(ys[i])] for i in idxs], dtype=float)

    reproj = []
    dists = []
    mode = "lidar_clean_dtm" if dtm_path is not None else "fixed_plane"
    t_neg = 0
    t_pos = 0
    t_none = 0
    for u, v in pts:
        ray_rect = pixel_to_ray_c0rect(float(u), float(v), ctx.calib)
        ray_c0 = ray_c0rect_to_ray_c0(ray_rect, ctx.calib)
        direction = t_w_c0[:3, :3] @ ray_c0
        direction = direction / max(1e-12, float(np.linalg.norm(direction)))
        if abs(direction[2]) > 1e-9:
            t_plane = (z0_use - float(t_w_c0[2, 3])) / float(direction[2])
        else:
            t_plane = None
        if t_plane is None:
            t_none += 1
        elif t_plane <= 0:
            t_neg += 1
        else:
            t_pos += 1
        pt = pixel_to_world_on_ground(
            frame_id,
            u,
            v,
            {"mode": mode, "z0": z0_use, "dtm_path": str(dtm_path) if dtm_path else ""},
            ctx=ctx,
        )
        if pt is None:
            continue
        xyz = np.array([[pt[0], pt[1], pt[2]]], dtype=float)
        uu, vv, valid = world_to_pixel_cam0(frame_id, xyz, ctx=ctx)
        if len(valid) == 0 or not bool(valid[0]):
            continue
        du = float(abs(uu[0] - u))
        dv = float(abs(vv[0] - v))
        d = float(np.hypot(du, dv))
        dists.append(d)
        reproj.append([float(uu[0]), float(vv[0])])

    valid_ratio = len(dists) / max(1, len(pts))
    cycle_p50 = float(np.percentile(dists, 50)) if dists else None
    cycle_p90 = float(np.percentile(dists, 90)) if dists else None

    status = "FAIL"
    if cycle_p90 is not None and cycle_p90 <= 2.0 and valid_ratio >= 0.30:
        status = "PASS"

    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    (run_dir / "debug").mkdir(parents=True, exist_ok=True)

    write_csv(
        run_dir / "tables" / "cycle_points.csv",
        [{"u": float(p[0]), "v": float(p[1])} for p in pts],
        ["u", "v"],
    )

    if dists:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(dists, bins=30)
            ax.set_title(f"pixel_cycle p90={cycle_p90:.2f}")
            fig.tight_layout()
            fig.savefig(run_dir / "images" / "cycle_hist.png")
            plt.close(fig)
        except Exception:
            pass

    _draw_overlay(img, pts, np.array(reproj, dtype=float), run_dir / "images" / "cycle_overlay.png", "cycle")

    calib_dump = _dump_calib(ctx.calib, t_w_c0)
    write_json(run_dir / "debug" / "calib_dump.json", calib_dump)
    trace = {
        "frame_id": frame_id,
        "z0_use": z0_use,
        "origin_z": float(t_w_c0[2, 3]),
        "sample_count": len(pts),
        "cycle_valid": len(dists),
        "t_plane_pos": t_pos,
        "t_plane_neg": t_neg,
        "t_plane_none": t_none,
    }
    write_json(
        run_dir / "debug" / "offset_dump.json",
        {
            "mask_size": list(mask_size_before),
            "image_size": [w, h],
            "offset_x": 0,
            "offset_y": 0,
            "unknown_crop": True,
            "trace": trace,
        },
    )

    decision = {
        "status": status,
        "frame_id": frame_id,
        "cycle_p50": cycle_p50,
        "cycle_p90": cycle_p90,
        "valid_ratio": round(valid_ratio, 4),
        "stage12_run": str(stage12_run),
        "dtm_path": str(dtm_path) if dtm_path else "",
    }
    write_json(run_dir / "decision.json", decision)

    report_lines = [
        "# Backproject cycle gate (0010 f290)",
        "",
        f"- status: {status}",
        f"- image_size: {w}x{h}",
        f"- mask_size: {mask_size_before[0]}x{mask_size_before[1]}",
        f"- crop_offset: (0,0) unknown_crop",
        f"- cycle_p50: {cycle_p50 if cycle_p50 is not None else 'NA'}",
        f"- cycle_p90: {cycle_p90 if cycle_p90 is not None else 'NA'}",
        f"- valid_ratio: {valid_ratio:.3f}",
        "",
        "## calib",
        f"- K_rect: {calib_dump.get('K_rect')}",
        f"- R_rect_00: {calib_dump.get('R_rect_00')}",
        f"- P_rect_00: {calib_dump.get('P_rect_00')}",
        f"- T_W_C0_t: {calib_dump.get('T_W_C0_t')}",
        "",
        "## outputs",
        "- images/cycle_overlay.png",
        "- images/cycle_hist.png",
        "- tables/cycle_points.csv",
        "- debug/calib_dump.json",
        "- debug/offset_dump.json",
    ]
    write_text(run_dir / "report.md", "\n".join(report_lines) + "\n")
    (run_dir / "params_hash.txt").write_text(_hash_file(run_dir / "report.md"), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

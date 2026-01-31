from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    world_to_pixel_cam0,
)
from pipeline.calib.kitti360_projection import project_cam0_to_image
from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_csv, write_json, write_text


FRAME_ID = "0000000290"


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


def _extract_contours(mask: np.ndarray) -> List[np.ndarray]:
    try:
        import cv2

        mask_u8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out = []
        for c in contours:
            if c is None or len(c) == 0:
                continue
            pts = c.reshape(-1, 2).astype(float)
            out.append(pts)
        return out
    except Exception:
        pass
    try:
        from skimage import measure

        contours = measure.find_contours(mask.astype(float), 0.5)
        out = []
        for c in contours:
            if c is None or len(c) == 0:
                continue
            pts = np.stack([c[:, 1], c[:, 0]], axis=1)
            out.append(pts.astype(float))
        return out
    except Exception:
        return []


def _sample_contour_points(contours: List[np.ndarray], step_px: int) -> np.ndarray:
    pts_all = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        pts = contour[:: max(1, step_px)].copy()
        for u, v in pts:
            pts_all.append([float(u), float(v)])
    return np.array(pts_all, dtype=float)


def _mask_overlay(img: Image.Image, mask: np.ndarray, out_path: Path) -> None:
    base = img.convert("RGBA")
    color = (255, 0, 0, 90)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    arr = np.array(overlay)
    sel = mask.astype(bool)
    arr[sel] = color
    overlay = Image.fromarray(arr, mode="RGBA")
    out = Image.alpha_composite(base, overlay)
    out.convert("RGB").save(out_path)


def _points_overlay(img: Image.Image, pts: np.ndarray, out_path: Path, color: Tuple[int, int, int]) -> None:
    canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for u, v in pts:
        draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=color)
    canvas.save(out_path)


def _cycle_overlay(img: Image.Image, pts: np.ndarray, reproj: np.ndarray, out_path: Path) -> None:
    canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for u, v in pts:
        draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=(255, 0, 0))
    for u, v in reproj:
        draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=(0, 255, 0))
    canvas.save(out_path)


def _line_overlay(img: Image.Image, pts_a: np.ndarray, pts_b: np.ndarray, out_path: Path) -> None:
    canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    if len(pts_a) >= 2:
        draw.line([tuple(p) for p in pts_a] + [tuple(pts_a[0])], fill=(255, 0, 0), width=2)
    if len(pts_b) >= 2:
        draw.line([tuple(p) for p in pts_b] + [tuple(pts_b[0])], fill=(0, 255, 0), width=2)
    canvas.save(out_path)


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


def _sample_dtm(ctx, x: float, y: float) -> Optional[float]:
    if ctx.dtm is None:
        return None
    try:
        val = next(ctx.dtm.sample([(float(x), float(y))]))
    except Exception:
        return None
    if val is None or len(val) == 0:
        return None
    z = float(val[0])
    if ctx.dtm_nodata is not None and np.isfinite(ctx.dtm_nodata):
        if abs(z - float(ctx.dtm_nodata)) < 1e-6:
            return None
    if not np.isfinite(z):
        return None
    return z


def _ray_from_pixel(u: float, v: float, calib, rect: bool) -> np.ndarray:
    if rect:
        ray_rect = pixel_to_ray_c0rect(float(u), float(v), calib)
        ray_c0 = ray_c0rect_to_ray_c0(ray_rect, calib)
        return ray_c0
    k = calib.k
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    x = (u - cx) / fx
    y = -(v - cy) / fy
    ray = np.array([x, y, 1.0], dtype=float)
    norm = float(np.linalg.norm(ray))
    return ray / norm if norm > 0 else ray


def _pixel_to_world(
    frame_id: str,
    u: float,
    v: float,
    ctx,
    rect: bool,
    z0_use: float,
    dtm_iter: int,
) -> Optional[np.ndarray]:
    ray_c0 = _ray_from_pixel(u, v, ctx.calib, rect=rect)
    t_w_c0 = ctx.pose_provider.get_t_w_c0(frame_id)
    origin = t_w_c0[:3, 3]
    direction = t_w_c0[:3, :3] @ ray_c0
    direction = direction / max(1e-12, float(np.linalg.norm(direction)))
    z_est = float(z0_use)
    for _ in range(max(1, dtm_iter)):
        if abs(direction[2]) < 1e-9:
            return None
        t = (z_est - float(origin[2])) / float(direction[2])
        if t <= 0:
            return None
        pt = origin + t * direction
        if ctx.dtm is None:
            return pt
        dtm_z = _sample_dtm(ctx, pt[0], pt[1])
        if dtm_z is None:
            return pt
        z_est = float(dtm_z)
    return pt


def _world_to_pixel(world_pts: np.ndarray, frame_id: str, ctx, rect: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rect:
        proj = world_to_pixel_cam0(frame_id, world_pts, ctx=ctx)
        return proj[0], proj[1], proj[2]
    t_w_c0 = ctx.pose_provider.get_t_w_c0(frame_id)
    t_c0_w = np.linalg.inv(t_w_c0)
    pts_h = np.hstack([world_pts[:, :3], np.ones((world_pts.shape[0], 1), dtype=world_pts.dtype)])
    cam = (t_c0_w @ pts_h.T).T
    proj = project_cam0_to_image(cam[:, :3], ctx.calib, use_rect=False, y_flip=True)
    return proj["u"], proj["v"], proj["valid"]


def _cycle(
    pts: np.ndarray,
    offset: Tuple[float, float],
    rect: bool,
    ctx,
    frame_id: str,
    z0_use: float,
    dtm_iter: int,
) -> Dict[str, object]:
    xs = []
    ys = []
    xs2 = []
    ys2 = []
    errs = []
    reproj_pts = []
    for u, v in pts:
        u0 = float(u) + float(offset[0])
        v0 = float(v) + float(offset[1])
        world = _pixel_to_world(frame_id, u0, v0, ctx, rect=rect, z0_use=z0_use, dtm_iter=dtm_iter)
        if world is None:
            continue
        xyz = np.array([[world[0], world[1], world[2]]], dtype=float)
        uu, vv, valid = _world_to_pixel(xyz, frame_id, ctx, rect=rect)
        if len(valid) == 0 or not bool(valid[0]):
            continue
        u2 = float(uu[0])
        v2 = float(vv[0])
        xs.append(u0)
        ys.append(v0)
        xs2.append(u2)
        ys2.append(v2)
        err = float(np.hypot(u2 - u0, v2 - v0))
        errs.append(err)
        reproj_pts.append([u2, v2])
    if errs:
        p50 = float(np.percentile(errs, 50))
        p90 = float(np.percentile(errs, 90))
    else:
        p50 = None
        p90 = None
    valid_ratio = len(errs) / max(1, len(pts))
    return {
        "u": xs,
        "v": ys,
        "u2": xs2,
        "v2": ys2,
        "errs": errs,
        "p50": p50,
        "p90": p90,
        "valid_ratio": valid_ratio,
        "reproj_pts": np.array(reproj_pts, dtype=float),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/world_crosswalk_candidates_0010_f250_500.yaml")
    ap.add_argument("--stage12-run", default="")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    drive_id = str(cfg.get("drive_id") or "")
    image_cam = str(cfg.get("image_cam") or "image_00")
    back_cfg = cfg.get("backproject") or {}
    dtm_iter = int(back_cfg.get("dtm_iterations", 2))

    run_id = now_ts()
    run_dir = Path("runs") / f"backproject_debug_f290_{run_id}"
    ensure_overwrite(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)

    if args.stage12_run:
        stage12_run = Path(args.stage12_run)
    else:
        stage12_run = _find_latest_stage12_run()
    if stage12_run is None or not stage12_run.exists():
        raise SystemExit("stage12 run not found")

    mask_dir = stage12_run / str(cfg.get("input_mask_dir") or "stage2/masks")
    mask_path = mask_dir / f"frame_{FRAME_ID}.png"
    if not mask_path.exists():
        raise SystemExit("mask missing")

    data_root = _find_data_root(str(cfg.get("kitti_root") or ""))
    image_dir = _find_image_dir(data_root, drive_id, image_cam)
    img_path = _find_frame_path(image_dir, FRAME_ID)
    if img_path is None or not img_path.exists():
        raise SystemExit("image missing")

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    mask = _load_mask(mask_path)
    mask_size_before = (mask.shape[1], mask.shape[0])
    mask = _resize_mask(mask, (w, h))
    mask_size_after = (mask.shape[1], mask.shape[0])

    contours = _extract_contours(mask)
    contour_pts = _sample_contour_points(contours, step_px=int(back_cfg.get("contour_sample_step_px", 2)))
    if contour_pts.size == 0:
        raise SystemExit("no contour points")

    _mask_overlay(img, mask, run_dir / "images" / "mask_full.png")
    _points_overlay(img, contour_pts, run_dir / "images" / "contour_points_full.png", (255, 0, 0))

    dtm_path_cfg = str(back_cfg.get("dtm_path") or "auto_latest_clean_dtm")
    dtm_path = _find_latest_dtm() if dtm_path_cfg == "auto_latest_clean_dtm" else Path(dtm_path_cfg)
    dtm_median = _dtm_median(dtm_path)

    ctx = configure_default_context(data_root, drive_id, cam_id=image_cam, dtm_path=dtm_path, frame_id_for_size=FRAME_ID)
    t_w_c0 = ctx.pose_provider.get_t_w_c0(FRAME_ID)
    origin_z = float(t_w_c0[2, 3])
    cam_z0 = origin_z - float(back_cfg.get("camera_height_m", 1.6))
    z0_use = float(dtm_median) if dtm_median is not None and float(dtm_median) < origin_z else cam_z0

    roi_offset = None
    cfg_path = stage12_run / "resolved_config.yaml"
    if cfg_path.exists():
        stage12_cfg = load_yaml(cfg_path)
        st1 = stage12_cfg.get("stage1") or {}
        bottom = st1.get("roi_bottom_crop")
        side = st1.get("roi_side_crop")
        if bottom is not None and side is not None:
            try:
                x0 = float(side[0]) * w
                y0 = float(bottom) * h
                roi_offset = (x0, y0)
            except Exception:
                roi_offset = None

    rng = random.Random(20260130)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise SystemExit("empty mask")
    idxs = rng.sample(range(len(xs)), min(500, len(xs)))
    pts = np.array([[float(xs[i]), float(ys[i])] for i in idxs], dtype=float)

    modes = []
    offset_a = (0.0, 0.0)
    offset_b = roi_offset if roi_offset is not None else None
    for offset, offset_name in [(offset_a, "offset0"), (offset_b, "offset_roi")]:
        if offset is None:
            continue
        for rect in (True, False):
            modes.append({"offset": offset, "offset_name": offset_name, "rect": rect})

    results = []
    for mode in modes:
        res = _cycle(pts, mode["offset"], mode["rect"], ctx, FRAME_ID, z0_use, dtm_iter)
        results.append({**mode, **res})

    if not results:
        raise SystemExit("no cycle results")
    best = min(results, key=lambda r: r["p90"] if r["p90"] is not None else 1e9)

    cycle_pts_rows = []
    for r in results:
        for u, v, u2, v2, err in zip(r["u"], r["v"], r["u2"], r["v2"], r["errs"]):
            cycle_pts_rows.append(
                {
                    "mode": f"{r['offset_name']}_{'rect' if r['rect'] else 'nonrect'}",
                    "u": u,
                    "v": v,
                    "u2": u2,
                    "v2": v2,
                    "err": err,
                }
            )
    write_csv(run_dir / "tables" / "cycle_points.csv", cycle_pts_rows, ["mode", "u", "v", "u2", "v2", "err"])

    _cycle_overlay(img, pts, best["reproj_pts"], run_dir / "images" / "cycle_overlay.png")

    contour_offset = best["offset"]
    contour_pts_full = contour_pts + np.array([contour_offset[0], contour_offset[1]], dtype=float)
    world_pts = []
    for u, v in contour_pts_full:
        wpt = _pixel_to_world(FRAME_ID, float(u), float(v), ctx, rect=best["rect"], z0_use=z0_use, dtm_iter=dtm_iter)
        if wpt is None:
            continue
        world_pts.append([wpt[0], wpt[1], wpt[2]])
    if world_pts:
        world_pts = np.array(world_pts, dtype=float)
        u2, v2, valid = _world_to_pixel(world_pts, FRAME_ID, ctx, rect=best["rect"])
        reproj_line = np.array([[float(uu), float(vv)] for uu, vv, ok in zip(u2, v2, valid) if ok], dtype=float)
    else:
        reproj_line = np.zeros((0, 2), dtype=float)
    _line_overlay(img, contour_pts_full, reproj_line, run_dir / "images" / "roundtrip_overlay.png")

    contour_rows = [{"u": float(p[0]), "v": float(p[1])} for p in contour_pts_full]
    write_csv(run_dir / "tables" / "contour_points.csv", contour_rows, ["u", "v"])

    def _range(vals: List[float]) -> List[float]:
        if not vals:
            return []
        return [float(min(vals)), float(max(vals))]

    meta = {
        "image_size": [w, h],
        "mask_size_before": list(mask_size_before),
        "mask_size_after": list(mask_size_after),
        "crop_offset_used": list(roi_offset) if roi_offset is not None else None,
        "projection_model_used": {
            "P_rect_00": ctx.calib.p_rect_00.tolist() if ctx.calib.p_rect_00 is not None else None,
            "R_rect_00": ctx.calib.r_rect_00.tolist() if ctx.calib.r_rect_00 is not None else None,
            "K_rect": (ctx.calib.p_rect_00[:3, :3].tolist() if ctx.calib.p_rect_00 is not None else ctx.calib.k.tolist()),
        },
        "backproject_model_used": "rect" if best["rect"] else "nonrect",
        "cycle_p90": best["p90"],
        "cycle_valid_ratio": best["valid_ratio"],
        "uv_range": {"u": _range(best["u"]), "v": _range(best["v"])},
        "u2v2_range": {"u2": _range(best["u2"]), "v2": _range(best["v2"])},
        "modes": [
            {
                "mode": f"{r['offset_name']}_{'rect' if r['rect'] else 'nonrect'}",
                "offset": list(r["offset"]),
                "p50": r["p50"],
                "p90": r["p90"],
                "valid_ratio": r["valid_ratio"],
                "uv_range": {"u": _range(r["u"]), "v": _range(r["v"])},
                "u2v2_range": {"u2": _range(r["u2"]), "v2": _range(r["v2"])},
            }
            for r in results
        ],
        "pose_summary": {
            "T_W_C0_t": t_w_c0[:3, 3].tolist(),
            "T_W_C0_R": t_w_c0[:3, :3].tolist(),
            "T_C0_V": ctx.calib.t_c0_v.tolist(),
        },
    }
    write_json(run_dir / "tables" / "meta.json", meta)

    report_lines = [
        "# Backproject debug f290",
        "",
        f"- image_size: {w}x{h}",
        f"- mask_size_before: {mask_size_before[0]}x{mask_size_before[1]}",
        f"- mask_size_after: {mask_size_after[0]}x{mask_size_after[1]}",
        f"- crop_offset_used: {roi_offset if roi_offset is not None else 'null'}",
        f"- best_mode: {best['offset_name']}_{'rect' if best['rect'] else 'nonrect'}",
        f"- cycle_p90_best: {best['p90']}",
        f"- cycle_valid_ratio_best: {best['valid_ratio']:.3f}",
        "",
        "## outputs",
        "- images/mask_full.png",
        "- images/contour_points_full.png",
        "- images/cycle_overlay.png",
        "- images/roundtrip_overlay.png",
        "- tables/meta.json",
        "- tables/cycle_points.csv",
        "- tables/contour_points.csv",
    ]
    write_text(run_dir / "report.md", "\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

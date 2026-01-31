from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    from shapely.ops import unary_union
except Exception:  # pragma: no cover - optional at runtime
    unary_union = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline._io import load_yaml
from pipeline.calib.kitti360_backproject import (
    BackprojectContext,
    configure_default_context,
    pixel_to_world_on_ground,
    world_to_pixel_cam0,
)
from pipeline.calib.kitti360_world import kitti_world_to_utm32, utm32_to_kitti_world
from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_csv, write_json, write_text


CFG_DEFAULT = Path("configs/world_crosswalk_from_stage2_0010_f000_300.yaml")
DRIVE_ID = "2013_05_28_drive_0010_sync"


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
    default_root = Path(r"E:\\KITTI360\\KITTI-360")
    if default_root.exists():
        return default_root
    raise SystemExit("missing data root: set POC_DATA_ROOT or config.kitti_root")


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


def _find_latest_pass_stage2_run() -> Optional[Path]:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.glob("image_stage12_ensemble_0010_000_300_*") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for cand in candidates:
        dec = cand / "decision.json"
        if not dec.exists():
            continue
        try:
            payload = json.loads(dec.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = str(payload.get("status", "")).strip().upper()
        if status == "PASS":
            return cand
    return None


def _find_latest_dtm_0010() -> Optional[Path]:
    runs_dir = Path("runs")
    candidates = []
    for p in runs_dir.glob("lidar_ground_0010_*"):
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


def _sample_contour_points(
    contours: List[np.ndarray],
    step_px: int,
    img_w: int,
    img_h: int,
    bottom_crop: float,
    side_crop: Tuple[float, float],
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    sampled_contours: List[np.ndarray] = []
    sampled_pts: List[Tuple[float, float]] = []
    min_v = bottom_crop * img_h
    min_u = side_crop[0] * img_w
    max_u = side_crop[1] * img_w
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        pts = contour[:: max(1, step_px)].copy()
        keep = []
        for u, v in pts:
            if v < min_v:
                continue
            if u < min_u or u > max_u:
                continue
            keep.append([u, v])
            sampled_pts.append((float(u), float(v)))
        if len(keep) >= 3:
            sampled_contours.append(np.array(keep, dtype=float))
    return sampled_contours, sampled_pts


def _filter_with_fallback(
    contours: List[np.ndarray],
    step_px: int,
    img_w: int,
    img_h: int,
    bottom_crop: float,
    side_crop: Tuple[float, float],
    bottom_crop_fb: float,
    side_crop_fb: Tuple[float, float],
) -> Tuple[List[np.ndarray], List[Tuple[float, float]], str]:
    sampled_contours, sampled_pts = _sample_contour_points(
        contours, step_px, img_w, img_h, bottom_crop, side_crop
    )
    if sampled_pts:
        return sampled_contours, sampled_pts, "strict"
    sampled_contours, sampled_pts = _sample_contour_points(
        contours, step_px, img_w, img_h, bottom_crop_fb, side_crop_fb
    )
    return sampled_contours, sampled_pts, "fallback"

def _make_valid(geom):
    if geom is None:
        return None
    try:
        from shapely.make_valid import make_valid

        return make_valid(geom)
    except Exception:
        try:
            return geom.buffer(0)
        except Exception:
            return geom


def _polygon_from_world_points(world_contours: List[np.ndarray]):
    try:
        from shapely.geometry import Polygon, MultiPolygon
    except Exception:
        return None
    polys = []
    for pts in world_contours:
        if pts.shape[0] < 3:
            continue
        try:
            poly = Polygon(pts)
        except Exception:
            continue
        if poly.is_empty:
            continue
        try:
            if not poly.is_valid:
                poly = poly.buffer(0)
        except Exception:
            pass
        if poly.is_empty:
            continue
        if poly.geom_type == "Polygon":
            polys.append(poly)
        elif poly.geom_type == "MultiPolygon":
            polys.extend([p for p in poly.geoms if not p.is_empty])
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    return MultiPolygon(polys)


def _sample_dtm(ctx: BackprojectContext, frame_id: str, x: float, y: float, z: float) -> Optional[float]:
    if ctx.dtm is None:
        return None
    pts_wk = np.array([[float(x), float(y), float(z)]], dtype=np.float64)
    pts_wu = kitti_world_to_utm32(pts_wk, ctx.data_root, ctx.drive_id, frame_id)
    x_utm = float(pts_wu[0, 0])
    y_utm = float(pts_wu[0, 1])
    try:
        val = next(ctx.dtm.sample([(x_utm, y_utm)]))
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


def _sample_dtm_utm(ctx: BackprojectContext, x_utm: float, y_utm: float) -> Optional[float]:
    if ctx.dtm is None:
        return None
    try:
        val = next(ctx.dtm.sample([(float(x_utm), float(y_utm))]))
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


def _backproject_point(
    frame_id: str,
    u: float,
    v: float,
    ctx: BackprojectContext,
    use_dtm: bool,
    dtm_iter: int,
    z0: float,
) -> Tuple[Optional[np.ndarray], str, bool]:
    z_est = float(z0)
    dtm_hit = False
    reason = "ok"
    for _ in range(max(1, dtm_iter)):
        try:
            pt = pixel_to_world_on_ground(frame_id, u, v, {"mode": "fixed_plane", "z0": z_est}, ctx=ctx)
        except Exception:
            return None, "missing_pose", False
        if pt is None:
            return None, "ray_upwards", False
        if use_dtm and ctx.dtm is not None:
            dtm_z = _sample_dtm(ctx, frame_id, pt[0], pt[1], pt[2])
            if dtm_z is None:
                reason = "dtm_nodata"
                break
            dtm_hit = True
            z_est = float(dtm_z)
        else:
            break
    return pt, reason, dtm_hit


def _check_utm_bbox(geom) -> Tuple[bool, Dict[str, float]]:
    if geom is None or geom.is_empty:
        return False, {}
    minx, miny, maxx, maxy = geom.bounds
    bbox = {"e_min": float(minx), "e_max": float(maxx), "n_min": float(miny), "n_max": float(maxy)}
    ok = True
    if not (100000 <= minx <= 900000 and 100000 <= maxx <= 900000):
        ok = False
    if not (1000000 <= miny <= 9000000 and 1000000 <= maxy <= 9000000):
        ok = False
    return ok, bbox


def _sample_boundary_points(geom, n: int) -> List[Tuple[float, float]]:
    if geom is None or geom.is_empty:
        return []
    boundary = geom.boundary
    if boundary.is_empty or boundary.length <= 0:
        return []
    length = boundary.length
    pts = []
    for i in range(n):
        p = boundary.interpolate((i / max(1, n)) * length)
        pts.append((float(p.x), float(p.y)))
    return pts


def _montage(images: List[Tuple[str, Path]], out_path: Path, cols: int = 4) -> None:
    if not images:
        return
    loaded = []
    labels = []
    for label, path in images:
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        loaded.append(img)
        labels.append(label)
    if not loaded:
        return
    cols = max(1, cols)
    rows = (len(loaded) + cols - 1) // cols
    w, h = loaded[0].size
    canvas = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    for idx, img in enumerate(loaded):
        r = idx // cols
        c = idx % cols
        canvas.paste(img, (c * w, r * h))
        draw = ImageDraw.Draw(canvas)
        draw.text((c * w + 8, r * h + 8), labels[idx], fill=(255, 255, 255))
    canvas.save(out_path)


def _pie_chart(counts: Dict[str, int], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        labels = list(counts.keys())
        sizes = [counts[k] for k in labels]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%")
        ax.axis("equal")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    except Exception:
        img = Image.new("RGB", (640, 480), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        y = 20
        for k, v in counts.items():
            draw.text((20, y), f"{k}: {v}", fill=(255, 255, 255))
            y += 20
        img.save(out_path)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _geom_iou(a, b) -> float:
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    try:
        inter = a.intersection(b).area
        uni = a.union(b).area
    except Exception:
        return 0.0
    if uni <= 0:
        return 0.0
    return float(inter / uni)


def _cluster_candidates(
    canonical_items: List[Dict[str, object]],
    raw_items_by_frame: Dict[str, object],
    merge_dist_m: float,
    merge_iou_min: float,
    min_support: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    clusters: List[Dict[str, object]] = []
    for item in canonical_items:
        geom = item["geometry"]
        center = geom.centroid
        best_idx = -1
        best_iou = 0.0
        for idx, cluster in enumerate(clusters):
            cgeom = cluster["geometry"]
            dist = float(center.distance(cgeom.centroid))
            if dist > merge_dist_m:
                continue
            iou = _geom_iou(geom, cgeom)
            if iou < merge_iou_min:
                continue
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0:
            cluster = clusters[best_idx]
            cluster["frames"].append(item["frame_id"])
            cluster["geometry"] = unary_union([cluster["geometry"], geom]) if unary_union else cluster["geometry"]
        else:
            clusters.append({"frames": [item["frame_id"]], "geometry": geom})
    merged = []
    merged_raw = []
    for idx, cluster in enumerate(clusters, start=1):
        frames = sorted(set(cluster["frames"]))
        if len(frames) < min_support:
            continue
        geom = cluster["geometry"]
        if geom is None or geom.is_empty:
            continue
        center = geom.centroid
        rect = geom.minimum_rotated_rectangle
        coords = list(rect.exterior.coords) if hasattr(rect, "exterior") else []
        length = 0.0
        width = 0.0
        angle = 0.0
        if len(coords) >= 4:
            pts = np.array(coords[:-1], dtype=np.float64)
            edges = pts[1:] - pts[:-1]
            lengths = np.linalg.norm(edges, axis=1)
            if lengths.size > 0:
                i = int(np.argmax(lengths))
                edge = edges[i]
                angle = float(np.degrees(np.arctan2(edge[1], edge[0])) % 180.0)
                length = float(np.max(lengths))
                width = float(np.min(lengths))
        merged.append(
            {
                "candidate_id": f"cand_{idx:03d}",
                "support_frames": len(frames),
                "geometry": geom,
                "frame_id_ref": frames[0],
                "center_x": float(center.x),
                "center_y": float(center.y),
                "angle_deg": angle,
                "length_m": length,
                "width_m": width,
            }
        )
        raw_geoms = [raw_items_by_frame.get(fid) for fid in frames if raw_items_by_frame.get(fid) is not None]
        raw_union = unary_union(raw_geoms) if raw_geoms and unary_union else None
        if raw_union is not None and not raw_union.is_empty:
            merged_raw.append(
                {
                    "candidate_id": f"cand_{idx:03d}",
                    "support_frames": len(frames),
                    "geometry": raw_union,
                    "frame_id_ref": frames[0],
                }
            )
    return merged, merged_raw

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CFG_DEFAULT))
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    if not cfg:
        raise SystemExit(f"missing config: {cfg_path}")

    frame_start = int(cfg.get("FRAME_START", 0))
    frame_end = int(cfg.get("FRAME_END", 300))
    image_cam = str(cfg.get("IMAGE_CAM", "image_00"))
    overwrite = bool(cfg.get("OVERWRITE", True))
    input_stage2 = str(cfg.get("INPUT_STAGE2_RUN", "auto_latest_pass"))
    input_mask_dir = str(cfg.get("INPUT_MASK_DIR", "stage2/merged_masks"))
    qa_frames_path = str(cfg.get("QA_FRAMES_PATH", "qa/qa_frames.json"))

    run_id = now_ts()
    run_dir = Path("runs") / f"world_crosswalk_from_stage2_0010_000_300_{run_id}"
    ensure_overwrite(run_dir if overwrite else run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")

    if input_stage2 == "auto_latest_pass":
        stage2_run = _find_latest_pass_stage2_run()
    else:
        stage2_run = Path(input_stage2)
    if stage2_run is None or not stage2_run.exists():
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "stage2_pass_run_not_found"})
        return 0

    mask_dir = stage2_run / input_mask_dir
    if not mask_dir.exists():
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "mask_dir_missing"})
        return 0

    qa_frames = [0, 100, 250, 290, 300]
    qa_path = stage2_run / qa_frames_path
    if qa_path.exists():
        try:
            qa_json = json.loads(qa_path.read_text(encoding="utf-8"))
            qa_frames = [int(v) for v in qa_json.get("frames") or qa_frames]
        except Exception:
            qa_frames = qa_frames

    data_root = _find_data_root(str(cfg.get("KITTI_ROOT", "")))
    image_dir = _find_image_dir(data_root, DRIVE_ID, image_cam)

    img_size = None
    for fid in range(frame_start, frame_end + 1):
        img_path = _find_frame_path(image_dir, f"{fid:010d}")
        if img_path and img_path.exists():
            img = Image.open(img_path)
            img_size = img.size
            break
    if img_size is None:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "image_frames_missing"})
        return 0

    use_dtm = bool(cfg.get("USE_DTM", True))
    dtm_path_cfg = str(cfg.get("DTM_PATH", "auto_latest_clean_dtm_0010"))
    if not use_dtm:
        dtm_path = None
    elif dtm_path_cfg == "auto_latest_clean_dtm_0010":
        dtm_path = _find_latest_dtm_0010()
    else:
        dtm_path = Path(dtm_path_cfg) if dtm_path_cfg else None
        if dtm_path is not None and not dtm_path.exists():
            dtm_path = None

    dtm_median = _dtm_median(dtm_path)
    ctx = configure_default_context(data_root, DRIVE_ID, cam_id=image_cam, dtm_path=dtm_path, frame_id_for_size=f"{frame_start:010d}")

    output_epsg = int(cfg.get("OUTPUT_EPSG", 32632))
    if output_epsg != 32632:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "unsupported_output_epsg"})
        return 0

    frames_dir = run_dir / "frames"
    tables_dir = run_dir / "tables"
    merged_dir = run_dir / "merged"
    images_dir = run_dir / "images"
    frames_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    per_frame_rows = []
    roundtrip_rows = []
    qa_overlays = []
    reason_counter = Counter()
    bbox_fail = False
    pixel_present_frames: List[str] = []
    raw_items_by_frame: Dict[str, object] = {}
    canonical_items: List[Dict[str, object]] = []

    step_px = int(cfg.get("CONTOUR_SAMPLE_STEP_PX", 2))
    side_crop = tuple(cfg.get("PIXEL_SIDE_CROP", [0.05, 0.95]))
    strict_v_min = float(cfg.get("PIXEL_FILTER_STRICT_V_MIN", 0.45))
    fallback_v_min = float(cfg.get("PIXEL_FILTER_FALLBACK_V_MIN", 0.30))
    min_area_px = int(cfg.get("MIN_AREA_PX_PRESENT", 200))
    min_valid_world_pts = int(cfg.get("MIN_VALID_WORLD_PTS", 60))
    dtm_iter = int(cfg.get("DTM_ITERATIONS", 2))
    camera_height = float(cfg.get("CAMERA_HEIGHT_M", 1.65))
    canonical_margin = float(cfg.get("CANONICAL_MARGIN_M", 0.30))
    canonical_min_area = float(cfg.get("CANONICAL_MIN_AREA_M2", 10.0))
    canonical_max_area = float(cfg.get("CANONICAL_MAX_AREA_M2", 350.0))

    img_w, img_h = img_size
    for frame in range(frame_start, frame_end + 1):
        frame_id = f"{frame:010d}"
        mask_path = mask_dir / f"frame_{frame_id}.png"
        frame_dir = frames_dir / frame_id
        frame_dir.mkdir(parents=True, exist_ok=True)

        if not mask_path.exists():
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": 0.0,
                    "status": "not_crosswalk",
                    "reason": "missing_mask",
                    "valid_world_pts": 0,
                    "dtm_hit_ratio": 0.0,
                    "canonical_area_m2": "",
                }
            )
            reason_counter["not_crosswalk"] += 1
            continue

        mask = _load_mask(mask_path)
        mask = _resize_mask(mask, (img_w, img_h))
        mask_area = float(np.count_nonzero(mask))
        if mask_area < min_area_px:
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": mask_area,
                    "status": "not_crosswalk",
                    "reason": "area_lt_min",
                    "valid_world_pts": 0,
                    "dtm_hit_ratio": 0.0,
                    "canonical_area_m2": "",
                }
            )
            reason_counter["not_crosswalk"] += 1
            continue

        pixel_present_frames.append(frame_id)
        contours = _extract_contours(mask)
        if not contours:
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": mask_area,
                    "status": "backproject_failed",
                    "reason": "empty_contour",
                    "valid_world_pts": 0,
                    "dtm_hit_ratio": 0.0,
                    "canonical_area_m2": "",
                }
            )
            reason_counter["backproject_failed"] += 1
            continue

        sampled_contours, sampled_pts, filter_stage = _filter_with_fallback(
            contours,
            step_px,
            img_w,
            img_h,
            strict_v_min,
            side_crop,
            fallback_v_min,
            side_crop,
        )

        if not sampled_pts:
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": mask_area,
                    "status": "backproject_failed",
                    "reason": "low_valid",
                    "valid_world_pts": 0,
                    "dtm_hit_ratio": 0.0,
                    "canonical_area_m2": "",
                }
            )
            reason_counter["backproject_failed"] += 1
            continue

        try:
            origin_z = float(ctx.pose_provider.get_t_w_c0(frame_id)[:3, 3][2])
        except Exception:
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": mask_area,
                    "status": "backproject_failed",
                    "reason": "missing_pose",
                    "valid_world_pts": 0,
                    "dtm_hit_ratio": 0.0,
                    "canonical_area_m2": "",
                }
            )
            reason_counter["backproject_failed"] += 1
            debug = {
                "status": "backproject_failed",
                "reason": "missing_pose",
                "mask_area_px": mask_area,
            }
            (frame_dir / "landing_debug.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")
            continue
        cam_z0 = origin_z - camera_height
        z0_use = float(dtm_median) if dtm_median is not None and float(dtm_median) < origin_z else cam_z0

        world_contours = []
        world_contours_wk = []
        valid_world_pts = 0
        ray_upwards = 0
        missing_pose = 0
        dtm_hit = 0
        dtm_miss = 0
        z_utm_samples: List[float] = []

        for contour in sampled_contours:
            world_pts = []
            world_pts_wk = []
            for u, v in contour:
                pt, reason, hit = _backproject_point(
                    frame_id,
                    float(u),
                    float(v),
                    ctx,
                    use_dtm,
                    dtm_iter,
                    z0_use,
                )
                if pt is None:
                    if reason == "ray_upwards":
                        ray_upwards += 1
                    elif reason == "missing_pose":
                        missing_pose += 1
                    continue
                valid_world_pts += 1
                if use_dtm and ctx.dtm is not None:
                    if hit:
                        dtm_hit += 1
                    else:
                        dtm_miss += 1
                world_pts_wk.append([float(pt[0]), float(pt[1]), float(pt[2])])
                pts_utm = kitti_world_to_utm32(np.array([pt], dtype=np.float64), data_root, DRIVE_ID, frame_id)
                world_pts.append([float(pts_utm[0, 0]), float(pts_utm[0, 1])])
                z_utm_samples.append(float(pts_utm[0, 2]))
            if len(world_pts) >= 3:
                world_contours.append(np.array(world_pts, dtype=float))
            if len(world_pts_wk) >= 3:
                world_contours_wk.append(np.array(world_pts_wk, dtype=float))

        if valid_world_pts < min_valid_world_pts:
            reason = "low_valid"
            if missing_pose >= len(sampled_pts):
                reason = "missing_pose"
            elif ray_upwards >= len(sampled_pts):
                reason = "ray_upwards"
            elif use_dtm and ctx.dtm is not None and dtm_hit == 0:
                reason = "dtm_nodata"
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": mask_area,
                    "status": "backproject_failed",
                    "reason": reason,
                    "valid_world_pts": valid_world_pts,
                    "dtm_hit_ratio": round(dtm_hit / max(1, valid_world_pts), 4) if use_dtm else 0.0,
                    "canonical_area_m2": "",
                }
            )
            reason_counter["backproject_failed"] += 1
            debug = {
                "status": "backproject_failed",
                "reason": reason,
                "mask_area_px": mask_area,
                "filter_stage_used": filter_stage,
                "valid_world_pts": valid_world_pts,
                "dtm_hit_ratio": round(dtm_hit / max(1, valid_world_pts), 4) if use_dtm else 0.0,
                "z0_used": z0_use,
                "dtm_path": str(dtm_path) if dtm_path else "",
            }
            (frame_dir / "landing_debug.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")
            continue

        z_utm_median = float(np.median(z_utm_samples)) if z_utm_samples else None

        raw_geom = _polygon_from_world_points(world_contours)
        raw_geom = _make_valid(raw_geom)
        if raw_geom is None or raw_geom.is_empty:
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": mask_area,
                    "status": "backproject_failed",
                    "reason": "raw_invalid",
                    "valid_world_pts": valid_world_pts,
                    "dtm_hit_ratio": round(dtm_hit / max(1, valid_world_pts), 4) if use_dtm else 0.0,
                    "canonical_area_m2": "",
                }
            )
            reason_counter["backproject_failed"] += 1
            continue

        bbox_ok, bbox = _check_utm_bbox(raw_geom)
        if not bbox_ok:
            bbox_fail = True
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": mask_area,
                    "status": "backproject_failed",
                    "reason": "bbox_invalid",
                    "valid_world_pts": valid_world_pts,
                    "dtm_hit_ratio": round(dtm_hit / max(1, valid_world_pts), 4) if use_dtm else 0.0,
                    "canonical_area_m2": "",
                }
            )
            reason_counter["backproject_failed"] += 1
            continue

        try:
            rect = raw_geom.minimum_rotated_rectangle
            canonical_geom = rect.buffer(canonical_margin)
            canonical_geom = _make_valid(canonical_geom)
        except Exception:
            canonical_geom = None

        canonical_area = float(canonical_geom.area) if canonical_geom is not None and not canonical_geom.is_empty else 0.0
        if canonical_geom is None or canonical_geom.is_empty:
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": mask_area,
                    "status": "backproject_failed",
                    "reason": "canonical_invalid",
                    "valid_world_pts": valid_world_pts,
                    "dtm_hit_ratio": round(dtm_hit / max(1, valid_world_pts), 4) if use_dtm else 0.0,
                    "canonical_area_m2": "",
                }
            )
            reason_counter["backproject_failed"] += 1
            continue

        if canonical_area < canonical_min_area or canonical_area > canonical_max_area:
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": mask_area,
                    "status": "backproject_failed",
                    "reason": "canonical_area",
                    "valid_world_pts": valid_world_pts,
                    "dtm_hit_ratio": round(dtm_hit / max(1, valid_world_pts), 4) if use_dtm else 0.0,
                    "canonical_area_m2": round(canonical_area, 3),
                }
            )
            reason_counter["backproject_failed"] += 1
            continue

        bbox_ok, bbox = _check_utm_bbox(canonical_geom)
        if not bbox_ok:
            bbox_fail = True
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": mask_area,
                    "status": "backproject_failed",
                    "reason": "bbox_invalid",
                    "valid_world_pts": valid_world_pts,
                    "dtm_hit_ratio": round(dtm_hit / max(1, valid_world_pts), 4) if use_dtm else 0.0,
                    "canonical_area_m2": round(canonical_area, 3),
                }
            )
            reason_counter["backproject_failed"] += 1
            continue

        try:
            import geopandas as gpd

            gdf_raw = gpd.GeoDataFrame([{"frame_id": frame_id, "geometry": raw_geom}], geometry="geometry", crs="EPSG:32632")
            gdf_can = gpd.GeoDataFrame([{"frame_id": frame_id, "geometry": canonical_geom}], geometry="geometry", crs="EPSG:32632")
            gpkg_path = frame_dir / "world_crosswalk_utm32.gpkg"
            gdf_raw.to_file(gpkg_path, layer="raw_utm32", driver="GPKG")
            gdf_can.to_file(gpkg_path, layer="canonical_utm32", driver="GPKG")
        except Exception:
            pass

        raw_items_by_frame[frame_id] = raw_geom
        canonical_items.append({"frame_id": frame_id, "geometry": canonical_geom})

        per_frame_rows.append(
            {
                "frame_id": frame_id,
                "mask_area_px": mask_area,
                "status": "ok",
                "reason": "ok",
                "valid_world_pts": valid_world_pts,
                "dtm_hit_ratio": round(dtm_hit / max(1, valid_world_pts), 4) if use_dtm else 0.0,
                "canonical_area_m2": round(canonical_area, 3),
            }
        )
        reason_counter["ok"] += 1

        debug = {
            "status": "ok",
            "reason": "ok",
            "mask_area_px": mask_area,
            "filter_stage_used": filter_stage,
            "valid_world_pts": valid_world_pts,
            "dtm_hit_ratio": round(dtm_hit / max(1, valid_world_pts), 4) if use_dtm else 0.0,
            "z0_used": z0_use,
            "dtm_path": str(dtm_path) if dtm_path else "",
            "bbox_raw_utm32": bbox,
            "canonical_area_m2": round(canonical_area, 3),
            "z_utm_median": z_utm_median,
        }
        (frame_dir / "landing_debug.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")

        if int(frame_id) in set(qa_frames):
            img_path = _find_frame_path(image_dir, frame_id)
            if img_path and img_path.exists():
                img = Image.open(img_path).convert("RGB")
                overlay = img.copy()
                draw = ImageDraw.Draw(overlay)
                longest = max(contours, key=lambda c: c.shape[0]) if contours else None
                if longest is not None and len(longest) >= 2:
                    pts = [(float(u), float(v)) for u, v in longest]
                    draw.line(pts + [pts[0]], fill=(255, 0, 0), width=2)

                raw_proj = []
                raw_proj_src = None
                if world_contours_wk:
                    raw_proj_src = max(world_contours_wk, key=lambda c: c.shape[0])
                    u, v, valid = world_to_pixel_cam0(frame_id, raw_proj_src, ctx=ctx)
                    raw_proj = [(float(uu), float(vv)) for uu, vv, ok in zip(u, v, valid) if ok]
                    if len(raw_proj) >= 2:
                        draw.line(raw_proj + [raw_proj[0]], fill=(0, 255, 0), width=2)

                can_pts = _sample_boundary_points(canonical_geom, int(cfg.get("ROUNDTRIP_SAMPLE_PTS", 200)))
                can_proj = []
                if can_pts:
                    z_list = []
                    for x, y in can_pts:
                        if z_utm_median is not None:
                            z_val = float(z_utm_median)
                        else:
                            z_val = _sample_dtm_utm(ctx, float(x), float(y)) if use_dtm else None
                            if z_val is None:
                                z_val = float(z0_use)
                        z_list.append(z_val)
                    pts_wu = np.array([[x, y, z] for (x, y), z in zip(can_pts, z_list)], dtype=np.float64)
                    pts_wk = utm32_to_kitti_world(pts_wu, data_root, DRIVE_ID, frame_id)
                    u, v, valid = world_to_pixel_cam0(frame_id, pts_wk, ctx=ctx)
                    can_proj = [(float(uu), float(vv)) for uu, vv, ok in zip(u, v, valid) if ok]
                    if len(can_proj) >= 2:
                        draw.line(can_proj + [can_proj[0]], fill=(0, 120, 255), width=2)

                overlay_path = frame_dir / "overlay_roundtrip.png"
                overlay.save(overlay_path)
                qa_overlays.append((frame_id, overlay_path))

                p50 = ""
                p90 = ""
                valid_ratio = 0.0
                try:
                    from shapely.geometry import LineString, Point

                    if longest is not None and len(longest) >= 2 and raw_proj:
                        line = LineString([(float(u), float(v)) for u, v in longest])
                        if line.length > 0:
                            dists = [float(line.distance(Point(u, v))) for u, v in raw_proj]
                            if dists:
                                p50 = float(np.percentile(dists, 50))
                                p90 = float(np.percentile(dists, 90))
                                denom = len(raw_proj_src) if raw_proj_src is not None else len(raw_proj)
                                valid_ratio = float(len(raw_proj) / max(1, denom))
                except Exception:
                    pass

                roundtrip_rows.append(
                    {
                        "frame_id": frame_id,
                        "p50": p50,
                        "p90": p90,
                        "valid_ratio": round(valid_ratio, 4),
                        "reason": "ok" if p90 != "" else "no_roundtrip",
                    }
                )

    write_csv(
        tables_dir / "per_frame_landing.csv",
        per_frame_rows,
        ["frame_id", "mask_area_px", "status", "reason", "valid_world_pts", "dtm_hit_ratio", "canonical_area_m2"],
    )
    write_csv(
        tables_dir / "roundtrip_px_errors.csv",
        roundtrip_rows,
        ["frame_id", "p50", "p90", "valid_ratio", "reason"],
    )

    _montage(qa_overlays, images_dir / "qa_montage_roundtrip.png")
    _pie_chart(reason_counter, images_dir / "landing_reason_summary.png")

    merge_dist = float(cfg.get("MERGE_DIST_M", 2.0))
    merge_iou = float(cfg.get("MERGE_IOU_MIN", 0.20))
    min_support = int(cfg.get("MIN_SUPPORT_FRAMES", 3))
    merged_items = []
    merged_raw_items = []
    if canonical_items and unary_union is not None:
        merged_items, merged_raw_items = _cluster_candidates(
            canonical_items, raw_items_by_frame, merge_dist, merge_iou, min_support
        )

    try:
        import geopandas as gpd

        gdf_can = (
            gpd.GeoDataFrame(merged_items, geometry="geometry", crs="EPSG:32632")
            if merged_items
            else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:32632")
        )
        gdf_can.to_file(merged_dir / "crosswalk_candidates_canonical_utm32.gpkg", layer="crosswalk_candidates", driver="GPKG")

        gdf_raw = (
            gpd.GeoDataFrame(merged_raw_items, geometry="geometry", crs="EPSG:32632")
            if merged_raw_items
            else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:32632")
        )
        gdf_raw.to_file(merged_dir / "crosswalk_candidates_raw_utm32.gpkg", layer="crosswalk_candidates_raw", driver="GPKG")
    except Exception:
        pass

    merge_rows = [
        {
            "candidate_id": item["candidate_id"],
            "support_frames": item["support_frames"],
            "area_m2": float(item["geometry"].area) if item.get("geometry") is not None else 0.0,
            "center_x": item.get("center_x", 0.0),
            "center_y": item.get("center_y", 0.0),
            "angle_deg": item.get("angle_deg", 0.0),
            "length_m": item.get("length_m", 0.0),
            "width_m": item.get("width_m", 0.0),
        }
        for item in merged_items
    ]
    write_csv(
        tables_dir / "merge_stats.csv",
        merge_rows,
        ["candidate_id", "support_frames", "area_m2", "center_x", "center_y", "angle_deg", "length_m", "width_m"],
    )

    n_pixel_present = len(pixel_present_frames)
    n_ok = sum(1 for r in per_frame_rows if r["status"] == "ok")
    ok_rate = float(n_ok / max(1, n_pixel_present)) if n_pixel_present else 0.0

    qa_p90_vals = []
    qa_set = {f"{f:010d}" for f in qa_frames}
    for row in roundtrip_rows:
        if row.get("frame_id") not in qa_set:
            continue
        try:
            val = float(row.get("p90"))
            qa_p90_vals.append(val)
        except Exception:
            continue
    qa_p90_max = max(qa_p90_vals) if qa_p90_vals else None

    status = "PASS"
    if bbox_fail:
        status = "FAIL"
    elif n_pixel_present > 0 and n_ok == 0:
        status = "FAIL"
    elif qa_p90_max is not None and qa_p90_max > float(cfg.get("ROUNDTRIP_P90_WARN_PX", 15)):
        status = "FAIL"
    elif ok_rate < 0.6 or len(merged_items) == 0:
        status = "WARN"
    elif qa_p90_max is not None and qa_p90_max > float(cfg.get("ROUNDTRIP_P90_PASS_PX", 8)):
        status = "WARN"

    decision = {
        "status": status,
        "n_pixel_present": n_pixel_present,
        "n_ok": n_ok,
        "ok_rate": round(ok_rate, 4) if n_pixel_present else 0.0,
        "merged_candidate_count": len(merged_items),
        "qa_roundtrip_p90_max": qa_p90_max,
    }
    if bbox_fail:
        decision["reason"] = "utm_bbox_invalid"

    write_json(run_dir / "decision.json", decision)

    source_ref = {
        "input_stage2_run": str(stage2_run),
        "stage2_decision_hash": _hash_file(stage2_run / "decision.json") if (stage2_run / "decision.json").exists() else "",
    }
    (run_dir / "source_ref.json").write_text(json.dumps(source_ref, indent=2), encoding="utf-8")

    resolved_cfg = dict(cfg)
    resolved_cfg.update(
        {
            "RESOLVED": {
                "run_id": run_id,
                "input_stage2_run": str(stage2_run),
                "mask_dir": str(mask_dir),
                "dtm_path": str(dtm_path) if dtm_path else "",
            }
        }
    )
    import yaml

    resolved_path = run_dir / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved_cfg, sort_keys=False), encoding="utf-8")
    (run_dir / "params_hash.txt").write_text(_hash_file(resolved_path), encoding="utf-8")

    report_lines = [
        "# World Candidates from Stage2 merged_masks (0010 f0-300)",
        "",
        f"- status: {status}",
        f"- n_pixel_present: {n_pixel_present}",
        f"- n_ok: {n_ok}",
        f"- ok_rate: {ok_rate:.3f}" if n_pixel_present else "- ok_rate: 0.000",
        f"- merged_candidate_count: {len(merged_items)}",
        f"- qa_roundtrip_p90_max: {qa_p90_max if qa_p90_max is not None else 'NA'}",
        "",
        "## backproject_failed_top3",
        *[f"- {k}: {v}" for k, v in reason_counter.most_common(3)],
        "",
        "## outputs",
        "- merged/crosswalk_candidates_canonical_utm32.gpkg",
        "- merged/crosswalk_candidates_raw_utm32.gpkg",
        "- images/qa_montage_roundtrip.png",
        "- images/landing_reason_summary.png",
        "- tables/per_frame_landing.csv",
        "- tables/roundtrip_px_errors.csv",
        "- tables/merge_stats.csv",
    ]
    write_text(run_dir / "report.md", "\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

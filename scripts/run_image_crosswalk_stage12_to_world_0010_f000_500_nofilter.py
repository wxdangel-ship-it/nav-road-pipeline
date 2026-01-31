from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    import geopandas as gpd
except Exception:  # pragma: no cover - optional at runtime
    gpd = None

try:
    from shapely.ops import unary_union
except Exception:  # pragma: no cover - optional at runtime
    unary_union = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline._io import load_yaml
from pipeline.calib.kitti360_backproject import configure_default_context, pixel_to_world_on_ground, world_to_pixel_cam0
from pipeline.calib.kitti360_world import kitti_world_to_utm32, utm32_to_kitti_world
from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_csv, write_json, write_text


CFG_DEFAULT = Path("configs/image_crosswalk_stage12_to_world_0010_f000_500_nofilter.yaml")
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


def _find_latest_stage12_run() -> Optional[Path]:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.glob("image_stage12_ensemble_0010_000_300_*") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


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


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _copytree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


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


def _backproject_point(
    frame_id: str,
    u: float,
    v: float,
    ctx,
    use_dtm: bool,
    dtm_iter: int,
    z0: float,
) -> Tuple[Optional[np.ndarray], str]:
    z_est = float(z0)
    for _ in range(max(1, dtm_iter)):
        try:
            pt = pixel_to_world_on_ground(frame_id, u, v, {"mode": "fixed_plane", "z0": z_est}, ctx=ctx)
        except Exception:
            return None, "missing_pose"
        if pt is None:
            return None, "ray_upwards"
        if use_dtm and ctx.dtm is not None:
            try:
                pts_utm = kitti_world_to_utm32(
                    np.array([pt], dtype=np.float64),
                    ctx.data_root,
                    ctx.drive_id,
                    str(frame_id),
                )
                val = next(ctx.dtm.sample([(float(pts_utm[0, 0]), float(pts_utm[0, 1]))]))
            except Exception:
                val = None
            if val is None or len(val) == 0:
                break
            z_val = float(val[0])
            if not np.isfinite(z_val):
                break
            z_est = z_val
        else:
            break
    return pt, "ok"


def _check_utm_bbox(geom) -> bool:
    if geom is None or geom.is_empty:
        return False
    minx, miny, maxx, maxy = geom.bounds
    if not (100000 <= minx <= 900000 and 100000 <= maxx <= 900000):
        return False
    if not (1000000 <= miny <= 9000000 and 1000000 <= maxy <= 9000000):
        return False
    return True


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


def _cluster_candidates(candidates: List[Dict[str, object]], min_support: int) -> List[Dict[str, object]]:
    if unary_union is None:
        return candidates
    clusters: List[Dict[str, object]] = []
    for item in candidates:
        geom = item["geometry"]
        best_idx = -1
        best_iou = 0.0
        for idx, cluster in enumerate(clusters):
            cgeom = cluster["geometry"]
            dist = float(geom.centroid.distance(cgeom.centroid))
            if dist > item["merge_dist_m"]:
                continue
            iou = _geom_iou(geom, cgeom)
            if iou < item["merge_iou_min"]:
                continue
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0:
            cluster = clusters[best_idx]
            cluster["geoms"].append(geom)
            cluster["support_frames"].extend(item.get("support_frames_list", []))
            cluster["geometry"] = unary_union(cluster["geoms"])
        else:
            clusters.append(
                {
                    "geoms": [geom],
                    "geometry": geom,
                    "support_frames": list(item.get("support_frames_list", [])),
                }
            )
    merged = []
    for idx, cluster in enumerate(clusters, start=1):
        geom = cluster["geometry"]
        if geom is None or geom.is_empty:
            continue
        frames = sorted(set(int(f) for f in cluster["support_frames"]))
        if len(frames) < min_support:
            continue
        merged.append(
            {
                "candidate_id": f"cand_{idx:03d}",
                "support_frames": len(frames),
                "geometry": geom,
                "center_x": float(geom.centroid.x),
                "center_y": float(geom.centroid.y),
                "area_m2": float(geom.area),
            }
        )
    return merged

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CFG_DEFAULT))
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    frame_start = int(cfg.get("FRAME_START", 0))
    frame_end = int(cfg.get("FRAME_END", 500))
    image_cam = str(cfg.get("IMAGE_CAM", "image_00"))
    overwrite = bool(cfg.get("OVERWRITE", True))

    run_id = now_ts()
    run_dir = Path("runs") / f"image_crosswalk_0010_000_500_nofilter_{run_id}"
    ensure_overwrite(run_dir if overwrite else run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")

    stage12_cfg = {
        "FRAME_START": frame_start,
        "FRAME_END": frame_end,
        "IMAGE_CAM": image_cam,
        "STAGE1_STRIDE": int(cfg.get("STAGE1_STRIDE", 5)),
        "STAGE1_FORCE_FRAMES": list(cfg.get("STAGE1_FORCE_FRAMES", [290])),
        "MAX_SEEDS_PER_FRAME": int(cfg.get("MAX_SEEDS_PER_FRAME", 4)),
        "STAGE2_MAX_SEEDS_TOTAL": int(cfg.get("STAGE2_MAX_SEEDS_TOTAL", 30)),
        "ROI_BOTTOM_CROP": float(cfg.get("ROI_BOTTOM_CROP", 0.50)),
        "ROI_SIDE_CROP": list(cfg.get("ROI_SIDE_CROP", [0.05, 0.95])),
        "GDINO_TEXT_PROMPT": list(cfg.get("GDINO_TEXT_PROMPT") or []),
        "GDINO_BOX_TH": float(cfg.get("GDINO_BOX_TH", 0.23)),
        "GDINO_TEXT_TH": float(cfg.get("GDINO_TEXT_TH", 0.23)),
        "GDINO_TOPK": int(cfg.get("GDINO_TOPK", 12)),
        "NMS_IOU_TH": float(cfg.get("NMS_IOU_TH", 0.50)),
        "SAM2_IMAGE_MAX_MASKS": int(cfg.get("SAM2_IMAGE_MAX_MASKS", 2)),
        "SAM2_MASK_MIN_AREA_PX": float(cfg.get("SAM2_MASK_MIN_AREA_PX", 600)),
        "YOLOWORLD_NEG_PROMPTS": list(cfg.get("YOLOWORLD_NEG_PROMPTS") or []),
        "YOLOWORLD_CONF_TH": float(cfg.get("YOLOWORLD_CONF_TH", 0.25)),
        "NEG_IOU_TH": float(cfg.get("NEG_IOU_TH", 0.30)),
        "NEG_DROP_IF_CENTER_INSIDE": bool(cfg.get("NEG_DROP_IF_CENTER_INSIDE", True)),
        "BOX_ASPECT_MIN": float(cfg.get("BOX_ASPECT_MIN", 2.0)),
        "BOX_H_MAX_RATIO": float(cfg.get("BOX_H_MAX_RATIO", 0.35)),
        "BOX_CENTER_V_MIN_RATIO": float(cfg.get("BOX_CENTER_V_MIN_RATIO", 0.50)),
        "MASK_MRR_ASPECT_MIN": float(cfg.get("MASK_MRR_ASPECT_MIN", 2.0)),
        "STRIPE_MIN_COUNT": int(cfg.get("STRIPE_MIN_COUNT", 6)),
        "STAGE2_WINDOW_PRE": int(cfg.get("STAGE2_WINDOW_PRE", 30)),
        "STAGE2_WINDOW_POST": int(cfg.get("STAGE2_WINDOW_POST", 30)),
        "SAM2_VIDEO_PROPAGATE": str(cfg.get("SAM2_VIDEO_PROPAGATE", "both")),
        "OUTPUT_MASK_FORMAT": str(cfg.get("OUTPUT_MASK_FORMAT", "png")),
        "QA_RANDOM_SEED": int(cfg.get("QA_RANDOM_SEED", 20260130)),
        "QA_SAMPLE_N": int(cfg.get("QA_SAMPLE_N", 18)),
        "QA_FORCE_INCLUDE": list(cfg.get("QA_FORCE_INCLUDE") or [0, 100, 250, 290, 400, 500]),
        "OVERWRITE": True,
    }

    stage12_cfg_path = run_dir / "stage12_config.yaml"
    stage12_cfg_path.write_text(json.dumps(stage12_cfg, indent=2), encoding="utf-8")

    py = sys.executable
    cmd_stage12 = [
        py,
        "scripts/run_image_stage12_ensemble_gdino_yoloworld_0010_f000_300.py",
        "--config",
        str(stage12_cfg_path),
    ]
    subprocess.run(cmd_stage12, check=True)

    stage12_run = _find_latest_stage12_run()
    if stage12_run is None:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "stage12_run_not_found"})
        return 0

    stage12_decision = {}
    decision_path = stage12_run / "decision.json"
    if decision_path.exists():
        stage12_decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if str(stage12_decision.get("status", "")).upper() == "FAIL":
            write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "stage12_failed", "stage12": stage12_decision})
            return 0

    qa_dir = run_dir / "qa"
    stage1_dir = run_dir / "stage1"
    stage2_dir = run_dir / "stage2"
    frames_dir = run_dir / "frames"
    merged_dir = run_dir / "merged"
    tables_dir = run_dir / "tables"
    images_dir = run_dir / "images"
    for d in [qa_dir, stage1_dir, stage2_dir, frames_dir, merged_dir, tables_dir, images_dir]:
        d.mkdir(parents=True, exist_ok=True)

    _copytree(stage12_run / "qa" / "qa_frames.json", qa_dir / "qa_frames.json")
    _copytree(stage12_run / "qa" / "montage_stage1_boxes.png", qa_dir / "montage_stage1_boxes.png")
    _copytree(stage12_run / "qa" / "montage_stage1_seeds.png", qa_dir / "montage_stage1_seeds.png")
    _copytree(stage12_run / "qa" / "montage_stage2.png", qa_dir / "montage_stage2.png")

    _copytree(stage12_run / "stage1" / "seeds_index.csv", stage1_dir / "seeds_index.csv")
    _copytree(stage12_run / "stage1" / "qa_frames", stage1_dir / "qa_frames")
    _copytree(stage12_run / "stage2" / "merged_masks", stage2_dir / "merged_masks")
    _copytree(stage12_run / "stage2" / "overlays", stage2_dir / "overlays")
    _copytree(stage12_run / "tables" / "per_frame_stage1_counts.csv", tables_dir / "per_frame_stage1_counts.csv")
    _copytree(stage12_run / "tables" / "per_frame_mask_area.csv", tables_dir / "per_frame_mask_area.csv")

    qa_frames = [0, 100, 250, 290, 400, 500]
    qa_path = qa_dir / "qa_frames.json"
    if qa_path.exists():
        try:
            qa_json = json.loads(qa_path.read_text(encoding="utf-8"))
            qa_frames = [int(v) for v in qa_json.get("frames") or qa_frames]
        except Exception:
            qa_frames = qa_frames

    data_root = _find_data_root(str(cfg.get("KITTI_ROOT", "")))
    image_dir = _find_image_dir(data_root, DRIVE_ID, image_cam)
    dtm_path = _find_latest_dtm_0010() if bool(cfg.get("USE_DTM", True)) else None
    dtm_iter = int(cfg.get("DTM_ITERATIONS", 2))
    camera_height = float(cfg.get("CAMERA_HEIGHT_M", 1.65))
    z0_mode = str(cfg.get("FIXED_PLANE_Z0_MODE", "dtm_median"))
    ctx = configure_default_context(data_root, DRIVE_ID, cam_id=image_cam, dtm_path=dtm_path, frame_id_for_size=f"{frame_start:010d}")

    mask_dir = stage2_dir / "merged_masks"
    min_area_px = int(cfg.get("MIN_AREA_PX_PRESENT", 200))
    step_px = int(cfg.get("CONTOUR_SAMPLE_STEP_PX", 2))
    side_crop = tuple(cfg.get("PIXEL_SIDE_CROP", [0.05, 0.95]))
    strict_v_min = float(cfg.get("PIXEL_FILTER_STRICT_V_MIN", 0.45))
    fallback_v_min = float(cfg.get("PIXEL_FILTER_FALLBACK_V_MIN", 0.30))
    min_valid_world_pts = int(cfg.get("MIN_VALID_WORLD_PTS", 60))
    canonical_margin = float(cfg.get("CANONICAL_MARGIN_M", 0.30))
    canonical_min_area = float(cfg.get("CANONICAL_MIN_AREA_M2", 10.0))
    canonical_max_area = float(cfg.get("CANONICAL_MAX_AREA_M2", 350.0))
    merge_dist = float(cfg.get("MERGE_DIST_M", 2.0))
    merge_iou = float(cfg.get("MERGE_IOU_MIN", 0.20))
    min_support_main = int(cfg.get("MIN_SUPPORT_FRAMES_MAIN", 3))
    min_support_all = int(cfg.get("MIN_SUPPORT_FRAMES_ALL", 1))

    per_frame_rows = []
    roundtrip_rows = []
    qa_overlays = []
    reason_counter = Counter()
    bbox_fail = False
    candidates = []
    ok_frames = set()
    pixel_present_frames = 0

    sample_img = None
    for fid in range(frame_start, frame_end + 1):
        img_path = _find_frame_path(image_dir, f"{fid:010d}")
        if img_path and img_path.exists():
            sample_img = Image.open(img_path)
            break
    if sample_img is None:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "image_missing"})
        return 0
    img_w, img_h = sample_img.size

    for frame in range(frame_start, frame_end + 1):
        frame_id = f"{frame:010d}"
        mask_path = mask_dir / f"frame_{frame_id}.png"
        frame_out = frames_dir / frame_id
        frame_out.mkdir(parents=True, exist_ok=True)

        if not mask_path.exists():
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": 0.0,
                    "status": "not_crosswalk",
                    "reason": "missing_mask",
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
                }
            )
            reason_counter["not_crosswalk"] += 1
            continue

        pixel_present_frames += 1
        contours = _extract_contours(mask)
        if not contours:
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": mask_area,
                    "status": "backproject_failed",
                    "reason": "empty_contour",
                }
            )
            reason_counter["backproject_failed"] += 1
            continue

        sampled_contours, sampled_pts, filter_stage = _filter_with_fallback(
            contours, step_px, img_w, img_h, strict_v_min, side_crop, fallback_v_min, side_crop
        )
        if not sampled_pts:
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": mask_area,
                    "status": "backproject_failed",
                    "reason": "low_valid",
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
                }
            )
            reason_counter["backproject_failed"] += 1
            continue

        cam_z0 = origin_z - camera_height
        z0_use = cam_z0
        if z0_mode == "dtm_median" and dtm_path is not None and ctx.dtm is not None:
            try:
                arr = ctx.dtm.read(1, masked=True)
                data = np.array(arr, dtype=float)
                data = data[np.isfinite(data)]
                if data.size > 0:
                    dtm_med = float(np.median(data))
                    if dtm_med < origin_z:
                        z0_use = dtm_med
            except Exception:
                z0_use = cam_z0

        world_contours = []
        world_contours_wk = []
        valid_world_pts = 0
        ray_up = 0
        missing_pose = 0
        z_utm_samples = []

        for contour in sampled_contours:
            world_pts = []
            world_pts_wk = []
            for u, v in contour:
                pt, reason = _backproject_point(frame_id, float(u), float(v), ctx, bool(cfg.get("USE_DTM", True)), dtm_iter, z0_use)
                if pt is None:
                    if reason == "ray_upwards":
                        ray_up += 1
                    elif reason == "missing_pose":
                        missing_pose += 1
                    continue
                valid_world_pts += 1
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
            elif ray_up >= len(sampled_pts):
                reason = "ray_upwards"
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": mask_area,
                    "status": "backproject_failed",
                    "reason": reason,
                }
            )
            reason_counter["backproject_failed"] += 1
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
                }
            )
            reason_counter["backproject_failed"] += 1
            continue

        if not _check_utm_bbox(raw_geom):
            bbox_fail = True
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": mask_area,
                    "status": "backproject_failed",
                    "reason": "bbox_invalid",
                }
            )
            reason_counter["backproject_failed"] += 1
            continue

        try:
            rect = raw_geom.minimum_rotated_rectangle
            canonical_geom = _make_valid(rect.buffer(canonical_margin))
        except Exception:
            canonical_geom = None

        if canonical_geom is None or canonical_geom.is_empty:
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": mask_area,
                    "status": "backproject_failed",
                    "reason": "canonical_invalid",
                }
            )
            reason_counter["backproject_failed"] += 1
            continue

        canonical_area = float(canonical_geom.area)
        if canonical_area < canonical_min_area or canonical_area > canonical_max_area:
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": mask_area,
                    "status": "backproject_failed",
                    "reason": "canonical_area",
                }
            )
            reason_counter["backproject_failed"] += 1
            continue

        if not _check_utm_bbox(canonical_geom):
            bbox_fail = True
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": mask_area,
                    "status": "backproject_failed",
                    "reason": "bbox_invalid",
                }
            )
            reason_counter["backproject_failed"] += 1
            continue

        if gpd is not None:
            gpkg_path = frame_out / "world_crosswalk_utm32.gpkg"
            gdf_raw = gpd.GeoDataFrame([{"frame_id": frame_id, "geometry": raw_geom}], geometry="geometry", crs="EPSG:32632")
            gdf_can = gpd.GeoDataFrame([{"frame_id": frame_id, "geometry": canonical_geom}], geometry="geometry", crs="EPSG:32632")
            gdf_raw.to_file(gpkg_path, layer="raw_utm32", driver="GPKG")
            gdf_can.to_file(gpkg_path, layer="canonical_utm32", driver="GPKG")

        per_frame_rows.append(
            {
                "frame_id": frame_id,
                "mask_area_px": mask_area,
                "status": "ok",
                "reason": "ok",
            }
        )
        ok_frames.add(frame_id)
        reason_counter["ok"] += 1

        candidates.append(
            {
                "frame_id": frame_id,
                "geometry": canonical_geom,
                "support_frames_list": [frame],
                "merge_dist_m": merge_dist,
                "merge_iou_min": merge_iou,
                "z_utm_median": z_utm_median,
                "raw_proj_src": max(world_contours_wk, key=lambda c: c.shape[0]) if world_contours_wk else None,
                "contours_px": max(contours, key=lambda c: c.shape[0]) if contours else None,
            }
        )

        if frame in qa_frames:
            img_path = _find_frame_path(image_dir, frame_id)
            if img_path and img_path.exists():
                base = Image.open(img_path).convert("RGB")
                draw = ImageDraw.Draw(base)
                contour = max(contours, key=lambda c: c.shape[0]) if contours else None
                if contour is not None and len(contour) >= 2:
                    pts = [(float(u), float(v)) for u, v in contour]
                    draw.line(pts + [pts[0]], fill=(255, 0, 0), width=2)

                raw_proj = []
                raw_src = max(world_contours_wk, key=lambda c: c.shape[0]) if world_contours_wk else None
                if raw_src is not None:
                    u, v, valid = world_to_pixel_cam0(frame_id, raw_src, ctx=ctx)
                    raw_proj = [(float(uu), float(vv)) for uu, vv, ok in zip(u, v, valid) if ok]
                    if len(raw_proj) >= 2:
                        draw.line(raw_proj + [raw_proj[0]], fill=(0, 255, 0), width=2)

                canon_pts = _sample_boundary_points(canonical_geom, 120)
                canon_proj = []
                if canon_pts:
                    z_list = [float(z_utm_median) if z_utm_median is not None else 0.0 for _ in canon_pts]
                    pts_wu = np.array([[x, y, z] for (x, y), z in zip(canon_pts, z_list)], dtype=np.float64)
                    pts_wk = utm32_to_kitti_world(pts_wu, data_root, DRIVE_ID, frame_id)
                    u, v, valid = world_to_pixel_cam0(frame_id, pts_wk, ctx=ctx)
                    canon_proj = [(float(uu), float(vv)) for uu, vv, ok in zip(u, v, valid) if ok]
                    if len(canon_proj) >= 2:
                        draw.line(canon_proj + [canon_proj[0]], fill=(0, 120, 255), width=2)

                overlay_path = frame_out / "overlay_roundtrip.png"
                base.save(overlay_path)
                qa_overlays.append((frame_id, overlay_path))

                p50 = ""
                p90 = ""
                if contour is not None and raw_proj:
                    try:
                        from shapely.geometry import LineString, Point

                        line = LineString([(float(u), float(v)) for u, v in contour])
                        if line.length > 0:
                            dists = [float(line.distance(Point(u, v))) for u, v in raw_proj]
                            if dists:
                                p50 = float(np.percentile(dists, 50))
                                p90 = float(np.percentile(dists, 90))
                    except Exception:
                        pass

                roundtrip_rows.append(
                    {"frame_id": frame_id, "p50": p50, "p90": p90, "valid_ratio": 1.0, "reason": "ok" if p90 != "" else "no_roundtrip"}
                )

    write_csv(
        tables_dir / "per_frame_landing.csv",
        per_frame_rows,
        ["frame_id", "mask_area_px", "status", "reason"],
    )
    write_csv(
        tables_dir / "roundtrip_px_errors.csv",
        roundtrip_rows,
        ["frame_id", "p50", "p90", "valid_ratio", "reason"],
    )

    _montage(qa_overlays, qa_dir / "montage_roundtrip.png")
    _pie_chart(reason_counter, images_dir / "landing_reason_summary.png")

    merged_all = _cluster_candidates(candidates, min_support_all)
    merged_support3 = _cluster_candidates(candidates, min_support_main)

    if gpd is not None:
        gdf_all = (
            gpd.GeoDataFrame(merged_all, geometry="geometry", crs="EPSG:32632")
            if merged_all
            else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:32632")
        )
        gdf_all.to_file(merged_dir / "crosswalk_candidates_canonical_all_utm32.gpkg", layer="crosswalk_candidates", driver="GPKG")

        gdf_s3 = (
            gpd.GeoDataFrame(merged_support3, geometry="geometry", crs="EPSG:32632")
            if merged_support3
            else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:32632")
        )
        gdf_s3.to_file(merged_dir / "crosswalk_candidates_canonical_support3_utm32.gpkg", layer="crosswalk_candidates", driver="GPKG")

    merge_stats_all = [
        {
            "candidate_id": item.get("candidate_id", ""),
            "support_frames": item.get("support_frames", 0),
            "area_m2": float(item.get("area_m2", 0.0)),
            "center_x": float(item.get("center_x", 0.0)),
            "center_y": float(item.get("center_y", 0.0)),
        }
        for item in merged_all
    ]
    merge_stats_s3 = [
        {
            "candidate_id": item.get("candidate_id", ""),
            "support_frames": item.get("support_frames", 0),
            "area_m2": float(item.get("area_m2", 0.0)),
            "center_x": float(item.get("center_x", 0.0)),
            "center_y": float(item.get("center_y", 0.0)),
        }
        for item in merged_support3
    ]
    write_csv(merged_dir / "merge_stats_all.csv", merge_stats_all, ["candidate_id", "support_frames", "area_m2", "center_x", "center_y"])
    write_csv(merged_dir / "merge_stats_support3.csv", merge_stats_s3, ["candidate_id", "support_frames", "area_m2", "center_x", "center_y"])

    n_ok = len(ok_frames)
    n_pixel_present = pixel_present_frames
    ok_rate = float(n_ok / max(1, n_pixel_present)) if n_pixel_present else 0.0
    qa_p90_vals = []
    for row in roundtrip_rows:
        if int(row["frame_id"]) in set(qa_frames):
            try:
                qa_p90_vals.append(float(row["p90"]))
            except Exception:
                continue
    qa_p90_max = max(qa_p90_vals) if qa_p90_vals else None
    ok_290 = "0000000290" in ok_frames

    status = "PASS"
    if bbox_fail or n_ok == 0 or not ok_290:
        status = "FAIL"
    elif qa_p90_max is not None and qa_p90_max > 8:
        status = "FAIL"
    elif n_pixel_present > 0 and (len([r for r in per_frame_rows if r["status"] == "backproject_failed"]) / n_pixel_present) > 0.5:
        status = "WARN"
    elif ok_290 and len(merged_support3) == 0:
        status = "WARN"

    decision = {
        "status": status,
        "n_pixel_present": n_pixel_present,
        "n_ok": n_ok,
        "ok_rate": round(ok_rate, 4) if n_pixel_present else 0.0,
        "merged_all_count": len(merged_all),
        "merged_support3_count": len(merged_support3),
        "backproject_failed_top3": reason_counter.most_common(3),
    }
    write_json(run_dir / "decision.json", decision)

    resolved_cfg = dict(cfg)
    resolved_cfg.update(
        {
            "RESOLVED": {
                "run_id": run_id,
                "stage12_run": str(stage12_run),
                "dtm_path": str(dtm_path) if dtm_path else "",
            }
        }
    )
    import yaml

    resolved_path = run_dir / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved_cfg, sort_keys=False), encoding="utf-8")
    (run_dir / "params_hash.txt").write_text(_hash_file(resolved_path), encoding="utf-8")

    report_lines = [
        "# Image crosswalk stage12 -> world (0010 f0-500 nofilter)",
        "",
        f"- status: {status}",
        f"- n_pixel_present: {n_pixel_present}",
        f"- n_ok: {n_ok}",
        f"- ok_rate: {ok_rate:.3f}" if n_pixel_present else "- ok_rate: 0.000",
        f"- merged_all_count: {len(merged_all)}",
        f"- merged_support3_count: {len(merged_support3)}",
        "",
        "## backproject_failed_top3",
        *[f"- {k}: {v}" for k, v in reason_counter.most_common(3)],
        "",
        "## outputs",
        "- qa/montage_stage1_boxes.png",
        "- qa/montage_stage1_seeds.png",
        "- qa/montage_stage2.png",
        "- qa/montage_roundtrip.png",
        "- merged/crosswalk_candidates_canonical_all_utm32.gpkg",
        "- merged/crosswalk_candidates_canonical_support3_utm32.gpkg",
        "- tables/per_frame_landing.csv",
        "- tables/roundtrip_px_errors.csv",
    ]
    write_text(run_dir / "report.md", "\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw
from shapely import affinity
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.ops import unary_union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.datasets.kitti360_io import (
    load_kitti360_calib,
    load_kitti360_cam_to_pose_key,
    load_kitti360_lidar_points,
    load_kitti360_lidar_points_world,
    load_kitti360_pose,
    load_kitti360_pose_full,
)
from tools.build_image_sample_index import _extract_frame_id, _find_image_dir, _list_images
from tools.run_crosswalk_monitor_drive import _run_on_demand_infer
from tools.run_image_basemodel import _ensure_cache_env, _resolve_sam2_checkpoint
from tools.sam2_video_propagate import (
    build_video_predictor,
    mask_area_px,
    prepare_video_frames,
    propagate_seed,
    save_masks,
)


LOG = logging.getLogger("run_crosswalk_monitor_range")


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("run_crosswalk_monitor_range")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_camera_defaults(path: Path) -> dict:
    data = _load_yaml(path)
    defaults = {
        "default_camera": "image_00",
        "enforce_camera": True,
        "allow_override": False,
    }
    if not isinstance(data, dict):
        return defaults
    defaults.update({k: v for k, v in data.items() if v is not None})
    return defaults


def _assert_camera_consistency(
    camera: str,
    image_dir: Path | None,
    calib: Dict[str, np.ndarray] | None,
    cam_to_pose_key: str,
    lidar_world_mode: str,
    default_camera: str,
    enforce_camera: bool,
    allow_override: bool,
) -> None:
    if enforce_camera and not allow_override and camera != default_camera:
        raise SystemExit(f"ERROR: camera={camera} expected={default_camera}")
    if image_dir is None or not image_dir.exists():
        raise SystemExit("ERROR: image_dir_missing")
    image_dir_text = str(image_dir).replace("\\", "/").lower()
    if enforce_camera and "image_00" not in image_dir_text:
        raise SystemExit(f"ERROR: image_dir_not_image_00:{image_dir}")
    if enforce_camera and "data_rect" not in image_dir_text:
        raise SystemExit(f"ERROR: image_dir_not_data_rect:{image_dir}")
    if calib is None:
        raise SystemExit("ERROR: calib_missing")
    p_key = str(calib.get("p_rect_key", ""))
    r_key = str(calib.get("r_rect_key", ""))
    if enforce_camera and (p_key != "P_rect_00" or r_key != "R_rect_00"):
        raise SystemExit(f"ERROR: calib_rect_key_mismatch:p={p_key} r={r_key}")
    if enforce_camera and cam_to_pose_key != "image_00":
        raise SystemExit(f"ERROR: cam_to_pose_key_mismatch:{cam_to_pose_key}")
    if enforce_camera and lidar_world_mode != "fullpose":
        raise SystemExit(f"ERROR: lidar_world_mode={lidar_world_mode} expected=fullpose")


def _load_image_model_cfg(model_id: str, zoo_path: Path) -> dict:
    zoo = _load_yaml(zoo_path)
    models = zoo.get("models") or []
    for model in models:
        if str(model.get("model_id") or "") == model_id:
            return model
    return {}


_SAM2_STATE = {"predictor": None, "model_id": None}


def _get_sam2_predictor(model_id: str) -> object | None:
    if _SAM2_STATE["predictor"] is not None and _SAM2_STATE["model_id"] == model_id:
        return _SAM2_STATE["predictor"]
    model_cfg = _load_image_model_cfg(model_id, Path("configs/image_model_zoo.yaml"))
    if not model_cfg:
        return None
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception:
        return None
    download_cfg = model_cfg.get("download") or {}
    sam2_cfg = download_cfg.get("sam2_model_cfg")
    if not sam2_cfg:
        return None
    sam2_cfg = str(sam2_cfg)
    if "sam2/configs/" in sam2_cfg:
        sam2_cfg = "configs/" + sam2_cfg.split("sam2/configs/", 1)[1]
    elif not sam2_cfg.startswith("configs/"):
        sam2_cfg = f"configs/{sam2_cfg}"
    try:
        sam2_ckpt = _resolve_sam2_checkpoint(download_cfg, _ensure_cache_env())
    except Exception:
        ckpt_name = str(download_cfg.get("sam2_checkpoint") or "")
        fallback_paths = [
            REPO_ROOT / "cache" / "hf" / "sam2" / ckpt_name,
            REPO_ROOT / "runs" / "cache_ps" / "sam2" / ckpt_name,
            REPO_ROOT / "runs" / "hf_cache_user" / "sam2" / ckpt_name,
        ]
        sam2_ckpt = None
        for cand in fallback_paths:
            if cand.exists():
                sam2_ckpt = cand
                break
        if sam2_ckpt is None:
            return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2 = build_sam2(sam2_cfg, str(sam2_ckpt), device=device)
    predictor = SAM2ImagePredictor(sam2)
    _SAM2_STATE["predictor"] = predictor
    _SAM2_STATE["model_id"] = model_id
    return predictor


def _sam2_box_mask(predictor: object, image_path: str, bbox: List[float]) -> np.ndarray | None:
    if not image_path or not Path(image_path).exists():
        return None
    if predictor is None:
        return None
    img = Image.open(image_path).convert("RGB")
    img_np = np.asarray(img)
    try:
        predictor.set_image(img_np)
        box = np.array(bbox, dtype=np.float32)
        masks, _scores, _logits = predictor.predict(box=box[None, :], multimask_output=False)
    except Exception:
        return None
    if masks is None:
        return None
    if isinstance(masks, np.ndarray):
        mask = masks[0] if masks.ndim >= 3 else masks
        return mask.astype(bool)
    return None


def _mask_to_polygon(mask: np.ndarray, max_points: int = 4000) -> Polygon | None:
    if mask is None or mask.size == 0:
        return None
    ys, xs = np.where(mask)
    if xs.size < 10:
        return None
    if xs.size > max_points:
        step = int(max(1, xs.size // max_points))
        xs = xs[::step]
        ys = ys[::step]
    pts = [Point(float(x), float(y)) for x, y in zip(xs, ys)]
    hull = MultiPoint(pts).convex_hull
    if hull is None or hull.is_empty or hull.geom_type != "Polygon":
        return None
    return hull


def _geom_to_image_points(
    geom: Polygon | LineString,
    pose: Tuple[float, ...],
    calib: Dict[str, np.ndarray],
) -> List[Tuple[float, float]]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        coords = np.array(list(geom.exterior.coords), dtype=float)
    elif isinstance(geom, LineString):
        coords = np.array(list(geom.coords), dtype=float)
    else:
        return []
    if coords.shape[0] == 0:
        return []
    points = np.column_stack([coords[:, 0], coords[:, 1], np.zeros(coords.shape[0], dtype=float)])
    proj = _project_world_to_image(points, pose, calib)
    out: List[Tuple[float, float]] = []
    for u, v, valid in proj:
        if not valid:
            continue
        out.append((float(u), float(v)))
    return out


def _raw_gdf_to_mask(
    raw_gdf: gpd.GeoDataFrame,
    image_path: str,
    out_path: Path,
) -> Path | None:
    if raw_gdf is None or raw_gdf.empty:
        return None
    width, height = _load_image_size(image_path)
    if width <= 0 or height <= 0:
        return None
    base = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(base)
    for _, row in raw_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            pts = [(float(x), float(y)) for x, y in geom.exterior.coords]
            if len(pts) >= 3:
                draw.polygon(pts, fill=255)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                pts = [(float(x), float(y)) for x, y in poly.exterior.coords]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=255)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.save(out_path)
    return out_path if out_path.exists() else None


def _choose_stage2_seed(
    drive_id: str,
    cluster_id: str,
    candidate_gdf: gpd.GeoDataFrame,
    raw_stats: Dict[Tuple[str, str], Dict[str, float]],
    raw_frames: Dict[Tuple[str, str], dict],
    index_lookup: Dict[Tuple[str, str], str],
    seed_dir: Path,
) -> dict | None:
    if candidate_gdf.empty:
        return None
    subset = candidate_gdf[candidate_gdf.get("cluster_id", "") == cluster_id]
    if subset.empty:
        return None
    best_row = None
    best_score = -1.0
    for _, row in subset.iterrows():
        frame_id = _normalize_frame_id(str(row.get("frame_id") or ""))
        if not frame_id:
            continue
        raw_info = raw_stats.get((drive_id, frame_id), {})
        score = float(raw_info.get("raw_top_score", 0.0))
        rectangularity = float(row.get("rectangularity") or 0.0)
        gore_like = bool(row.get("gore_like", False))
        base = score + rectangularity
        if gore_like:
            base *= 0.2
        if base > best_score:
            best_score = base
            best_row = row
    if best_row is None:
        return None
    frame_id = _normalize_frame_id(str(best_row.get("frame_id") or ""))
    image_path = index_lookup.get((drive_id, frame_id), "")
    bbox_px = best_row.get("bbox_px")
    if isinstance(bbox_px, str) and bbox_px.startswith("["):
        try:
            bbox_px = json.loads(bbox_px)
        except Exception:
            bbox_px = None
    raw_info = raw_frames.get((drive_id, frame_id), {})
    seed_mask_path = None
    raw_gdf = raw_info.get("gdf")
    if raw_gdf is not None and not raw_gdf.empty and image_path:
        seed_mask_path = _raw_gdf_to_mask(
            raw_gdf,
            image_path,
            seed_dir / f"{cluster_id}_seed_{frame_id}.png",
        )
    return {
        "cluster_id": cluster_id,
        "seed_frame_id": frame_id,
        "seed_bbox_px": bbox_px,
        "seed_mask_path": str(seed_mask_path) if seed_mask_path else "",
        "seed_score": float(raw_stats.get((drive_id, frame_id), {}).get("raw_top_score", 0.0)),
        "seed_is_gore_like": int(bool(best_row.get("gore_like", False))),
    }


def _choose_stage2_seeds(
    drive_id: str,
    cluster_id: str,
    candidate_gdf: gpd.GeoDataFrame,
    raw_stats: Dict[Tuple[str, str], Dict[str, float]],
    raw_frames: Dict[Tuple[str, str], dict],
    index_lookup: Dict[Tuple[str, str], str],
    seed_dir: Path,
    seeds_per_cluster: int,
    min_seed_dist_m: float,
) -> List[dict]:
    if seeds_per_cluster <= 0:
        return []
    if seeds_per_cluster == 1:
        seed = _choose_stage2_seed(drive_id, cluster_id, candidate_gdf, raw_stats, raw_frames, index_lookup, seed_dir)
        return [seed] if seed else []
    if candidate_gdf.empty:
        return []
    subset = candidate_gdf[candidate_gdf.get("cluster_id", "") == cluster_id]
    if subset.empty:
        return []
    scored = []
    for _, row in subset.iterrows():
        frame_id = _normalize_frame_id(str(row.get("frame_id") or ""))
        if not frame_id:
            continue
        raw_info = raw_stats.get((drive_id, frame_id), {})
        score = float(raw_info.get("raw_top_score", 0.0))
        rectangularity = float(row.get("rectangularity") or 0.0)
        gore_like = bool(row.get("gore_like", False))
        base = score + rectangularity
        if gore_like:
            base *= 0.2
        geom = row.get("geometry")
        centroid = geom.centroid if geom is not None else None
        scored.append((base, frame_id, row, centroid))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen: List[dict] = []
    chosen_centroids: List[Point] = []
    for _, frame_id, row, centroid in scored:
        if len(chosen) >= seeds_per_cluster:
            break
        if centroid is not None and chosen_centroids:
            if any(centroid.distance(c) < min_seed_dist_m for c in chosen_centroids):
                continue
        image_path = index_lookup.get((drive_id, frame_id), "")
        bbox_px = row.get("bbox_px")
        if isinstance(bbox_px, str) and bbox_px.startswith("["):
            try:
                bbox_px = json.loads(bbox_px)
            except Exception:
                bbox_px = None
        raw_info = raw_frames.get((drive_id, frame_id), {})
        seed_mask_path = None
        raw_gdf = raw_info.get("gdf")
        if raw_gdf is not None and not raw_gdf.empty and image_path:
            seed_mask_path = _raw_gdf_to_mask(
                raw_gdf,
                image_path,
                seed_dir / f"{cluster_id}_seed_{frame_id}.png",
            )
        chosen.append(
            {
                "cluster_id": cluster_id,
                "seed_frame_id": frame_id,
                "seed_bbox_px": bbox_px,
                "seed_mask_path": str(seed_mask_path) if seed_mask_path else "",
                "seed_score": float(raw_stats.get((drive_id, frame_id), {}).get("raw_top_score", 0.0)),
                "seed_is_gore_like": int(bool(row.get("gore_like", False))),
            }
        )
        if centroid is not None:
            chosen_centroids.append(centroid)
    return chosen


def _write_seeds_jsonl(seeds: List[dict], out_path: Path) -> None:
    out_path.write_text(
        "\n".join([json.dumps(s, ensure_ascii=True) for s in seeds if s]) + "\n",
        encoding="utf-8",
    )

def _roi_bbox_from_geom(
    geom: Polygon,
    pose: Tuple[float, float, float] | None,
    calib: Dict[str, np.ndarray] | None,
    image_path: str,
    min_area_px: float,
    max_area_ratio: float,
) -> List[float] | None:
    if geom is None or geom.is_empty or pose is None or calib is None:
        return None
    coords = np.asarray(geom.exterior.coords, dtype=float)
    if coords.size == 0:
        return None
    world_pts = np.column_stack([coords[:, 0], coords[:, 1], np.zeros(coords.shape[0])])
    proj = _project_world_to_image(world_pts, pose, calib)
    if proj.size == 0:
        return None
    valid = proj[:, 2].astype(bool)
    if not np.any(valid):
        return None
    u = proj[valid, 0]
    v = proj[valid, 1]
    width, height = _load_image_size(image_path)
    if width <= 0 or height <= 0:
        return None
    minx = float(max(0.0, np.min(u)))
    maxx = float(min(width - 1.0, np.max(u)))
    miny = float(max(0.0, np.min(v)))
    maxy = float(min(height - 1.0, np.max(v)))
    if maxx <= minx or maxy <= miny:
        return None
    area = (maxx - minx) * (maxy - miny)
    if area < min_area_px:
        return None
    if area > max_area_ratio * float(width * height):
        return None
    return [minx, miny, maxx, maxy]


def _mask_to_bbox(mask: np.ndarray) -> List[float] | None:
    if mask is None or mask.size == 0:
        return None
    if mask.ndim >= 3:
        mask = np.squeeze(mask)
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    minx = float(np.min(xs))
    maxx = float(np.max(xs))
    miny = float(np.min(ys))
    maxy = float(np.max(ys))
    if maxx <= minx or maxy <= miny:
        return None
    return [minx, miny, maxx, maxy]


def _expand_bbox(bbox: List[float], margin: float, width: int, height: int) -> List[float]:
    minx, miny, maxx, maxy = bbox
    minx = max(0.0, minx - margin)
    miny = max(0.0, miny - margin)
    maxx = min(float(width - 1), maxx + margin)
    maxy = min(float(height - 1), maxy + margin)
    if maxx <= minx:
        maxx = minx + 1.0
    if maxy <= miny:
        maxy = miny + 1.0
    return [minx, miny, maxx, maxy]


def _bbox_contains_point(bbox: List[float], pt: Tuple[float, float]) -> bool:
    return bbox[0] <= pt[0] <= bbox[2] and bbox[1] <= pt[1] <= bbox[3]
def _normalize_frame_id(frame_id: str) -> str:
    digits = "".join(ch for ch in str(frame_id) if ch.isdigit())
    if not digits:
        return str(frame_id)
    return digits.zfill(10)


def _parse_frame_id(frame_id: str) -> Optional[int]:
    text = str(frame_id).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    try:
        return int(float(text))
    except ValueError:
        return None


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(value)
    except Exception:
        return default


def _select_primary_candidate(candidates: gpd.GeoDataFrame) -> Optional[pd.Series]:
    if candidates.empty:
        return None
    for _, row in candidates.iterrows():
        if _safe_int(row.get("geom_ok")) == 1 and str(row.get("proj_method") or "") == "lidar":
            return row
    for _, row in candidates.iterrows():
        if _safe_int(row.get("geom_ok")) == 1:
            return row
    return candidates.iloc[0]


def _clean_source_frame_id(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and math.isnan(value):
        return fallback
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text


def _x_ego_from_geom(pose: Tuple[float, ...] | None, geom: object) -> Optional[float]:
    if pose is None or geom is None or geom.is_empty:
        return None
    try:
        centroid = geom.centroid
    except Exception:
        return None
    if centroid is None:
        return None
    yaw = pose[5] if len(pose) >= 6 else pose[2]
    dx = float(centroid.x) - float(pose[0])
    dy = float(centroid.y) - float(pose[1])
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return float(c * dx + s * dy)


def _bbox_iou(b1: List[float], b2: List[float]) -> float:
    if not b1 or not b2 or len(b1) != 4 or len(b2) != 4:
        return 0.0
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    denom = area1 + area2 - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


def _polygon_from_points(points: List[Tuple[float, float]]) -> Polygon | None:
    if points is None or len(points) < 3:
        return None
    try:
        poly = Polygon(points)
    except Exception:
        return None
    if poly.is_empty or not poly.is_valid:
        return None
    return poly


def _polygon_bbox(poly: Polygon | None) -> List[float] | None:
    if poly is None or poly.is_empty:
        return None
    minx, miny, maxx, maxy = poly.bounds
    return [float(minx), float(miny), float(maxx), float(maxy)]


def _roundtrip_metrics(
    raw_info: dict,
    geom: Polygon | None,
    pose: Tuple[float, float, float] | None,
    calib: Dict[str, np.ndarray] | None,
    z_override: Optional[float] = None,
) -> Tuple[float | None, float | None, float | None]:
    if geom is None or geom.is_empty or pose is None or calib is None:
        return None, None, None
    raw_gdf = raw_info.get("gdf")
    if raw_gdf is None or raw_gdf.empty:
        raw_poly = None
    else:
        raw_poly = unary_union(raw_gdf.geometry.values)
    geom_use = geom
    if geom_use.geom_type == "MultiPolygon":
        parts = list(geom_use.geoms)
        geom_use = parts[0] if parts else geom_use
    if z_override is None:
        pts = _geom_to_image_points(geom_use, pose, calib)
    else:
        coords = np.array(list(geom_use.exterior.coords), dtype=float)
        points = np.column_stack([coords[:, 0], coords[:, 1], np.full(coords.shape[0], float(z_override))])
        proj = _project_world_to_image(points, pose, calib)
        pts = [(float(u), float(v)) for u, v, valid in proj if valid]
    reproj_poly = _polygon_from_points(pts)
    if raw_poly is not None and not raw_poly.is_empty and reproj_poly is not None:
        inter = raw_poly.intersection(reproj_poly).area
        union = raw_poly.union(reproj_poly).area
        iou = float(inter / union) if union > 0 else 0.0
        area_ratio = float(reproj_poly.area / raw_poly.area) if raw_poly.area > 0 else None
    else:
        bbox_raw = raw_info.get("bbox_px")
        if not (isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4):
            bbox_raw = None
        bbox_reproj = _polygon_bbox(reproj_poly)
        iou = _bbox_iou(list(bbox_raw) if bbox_raw else [], bbox_reproj or [])
        if bbox_raw and bbox_reproj:
            area_raw = (bbox_raw[2] - bbox_raw[0]) * (bbox_raw[3] - bbox_raw[1])
            area_ratio = (
                (bbox_reproj[2] - bbox_reproj[0]) * (bbox_reproj[3] - bbox_reproj[1]) / area_raw
                if area_raw > 0
                else None
            )
        else:
            area_ratio = None
    bbox_raw = raw_info.get("bbox_px")
    bbox_reproj = _polygon_bbox(reproj_poly)
    if bbox_raw and bbox_reproj and len(bbox_raw) == 4:
        cx1, cy1 = _bbox_center(list(bbox_raw))
        cx2, cy2 = _bbox_center(bbox_reproj)
        center_err = float(np.hypot(cx1 - cx2, cy1 - cy2))
    else:
        center_err = None
    return iou, center_err, area_ratio


def _render_lidar_proj_debug(
    image_path: str,
    out_path: Path,
    points_world: np.ndarray,
    pose: Tuple[float, float, float] | None,
    calib: Dict[str, np.ndarray] | None,
    max_points: int = 5000,
) -> None:
    if out_path.exists():
        out_path.unlink()
    if not image_path or not Path(image_path).exists():
        return
    if pose is None or calib is None or points_world.size == 0:
        _render_text_overlay(image_path, out_path, ["NO_LIDAR_OR_POSE"])
        return
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    pts = points_world
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
    proj = _project_world_to_image(pts, pose, calib)
    valid = proj[:, 2].astype(bool)
    u = proj[:, 0][valid]
    v = proj[:, 1][valid]
    for x, y in zip(u, v):
        draw.point((float(x), float(y)), fill=(0, 255, 0, 120))
    img.save(out_path)


def _select_roundtrip_candidate(candidates: gpd.GeoDataFrame) -> pd.Series | None:
    if candidates is None or candidates.empty:
        return None
    if "geom_ok" in candidates.columns:
        geom_ok = candidates[candidates["geom_ok"].fillna(0).astype(int) == 1]
        if not geom_ok.empty:
            lidar = geom_ok[geom_ok.get("proj_method", "") == "lidar"]
            if not lidar.empty:
                return lidar.iloc[0]
            return geom_ok.iloc[0]
    return candidates.iloc[0]


def _compute_roundtrip_metrics_for_range(
    drive_id: str,
    frame_start: int,
    frame_end: int,
    candidate_gdf: gpd.GeoDataFrame,
    raw_frames: Dict[Tuple[str, str], dict],
    pose_map: Dict[str, Tuple[float, float, float] | None],
    calib: Dict[str, np.ndarray] | None,
    lidar_stats: Dict[str, dict],
) -> Tuple[List[dict], Dict[str, dict]]:
    rows = []
    by_frame: Dict[str, dict] = {}
    if candidate_gdf.empty:
        return rows, by_frame
    candidate_gdf = candidate_gdf.copy()
    candidate_gdf["frame_id_norm"] = candidate_gdf["frame_id"].apply(_normalize_frame_id)
    for frame in range(frame_start, frame_end + 1):
        frame_id = _normalize_frame_id(str(frame))
        candidates = candidate_gdf[candidate_gdf["frame_id_norm"] == frame_id]
        raw_info = raw_frames.get((drive_id, frame_id), {})
        lidar_info = lidar_stats.get(frame_id, {})
        proj_method = ""
        geom_ok = 0
        if candidates is not None and not candidates.empty:
            proj_method = str(candidates.iloc[0].get("proj_method") or "")
            geom_ok = _safe_int(candidates.iloc[0].get("geom_ok", 0))
        row = _select_roundtrip_candidate(candidates)
        geom = row.geometry if row is not None else None
        proj_method = str(row.get("proj_method") or proj_method) if row is not None else proj_method
        geom_ok = _safe_int(row.get("geom_ok", geom_ok)) if row is not None else geom_ok
        pose = pose_map.get(frame_id)
        z_override = lidar_info.get("ground_z")
        iou, center_err, area_ratio = _roundtrip_metrics(raw_info, geom, pose, calib, z_override)
        record = {
            "frame_id": frame_id,
            "raw_status": raw_info.get("raw_status", ""),
            "proj_method": proj_method,
            "geom_ok": geom_ok,
            "reproj_iou": iou,
            "reproj_center_err_px": center_err,
            "reproj_area_ratio": area_ratio,
            "points_in_bbox": int(lidar_info.get("points_in_bbox", 0) or 0),
            "points_in_mask": int(lidar_info.get("points_in_mask", 0) or 0),
            "proj_in_image_ratio_visible": float(lidar_info.get("proj_in_image_ratio_visible", 0.0) or 0.0),
        }
        rows.append(record)
        by_frame[frame_id] = record
    return rows, by_frame


def _scan_offsets_for_range(
    data_root: Path,
    drive_id: str,
    frame_start: int,
    frame_end: int,
    offsets: List[int],
    raw_frames: Dict[Tuple[str, str], dict],
    calib: Dict[str, np.ndarray] | None,
    cam_to_pose: Optional[np.ndarray],
    lidar_cfg: dict,
    index_lookup: Dict[Tuple[str, str], str],
    lidar_world_mode: str,
    cam_id: str,
) -> Tuple[List[dict], Dict[int, int]]:
    rows = []
    hist: Dict[int, int] = {}
    for frame in range(frame_start, frame_end + 1):
        frame_id = _normalize_frame_id(str(frame))
        raw_info = raw_frames.get((drive_id, frame_id))
        if not raw_info:
            continue
        best = None
        best_offset = None
        best_iou = None
        best_points = None
        base_iou = None
        base_points = None
        for offset in offsets:
            cand_frame = frame + offset
            if cand_frame < 0:
                continue
            cand_frame_id = _normalize_frame_id(str(cand_frame))
            try:
                if lidar_world_mode == "fullpose":
                    pose = load_kitti360_pose_full(data_root, drive_id, cand_frame_id)
                else:
                    pose = load_kitti360_pose(data_root, drive_id, cand_frame_id)
            except Exception:
                continue
            image_path = index_lookup.get((drive_id, frame_id), "")
            cand, stats = _build_lidar_candidate_for_frame(
                data_root,
                drive_id,
                cand_frame_id,
                image_path,
                raw_info,
                pose,
                calib,
                cam_to_pose,
                lidar_cfg,
                lidar_world_mode,
                cam_id,
            )
            geom = cand.get("geometry") if cand else None
            iou, _center, _ratio = _roundtrip_metrics(raw_info, geom, pose, calib, None)
            iou_val = float(iou) if iou is not None else 0.0
            points = int(stats.get("points_in_mask", 0) or 0)
            score = iou_val * 10000 + points
            if offset == 0:
                base_iou = iou_val
                base_points = points
            if best is None or score > best:
                best = score
                best_offset = offset
                best_iou = iou_val
                best_points = points
        if best_offset is None:
            continue
        hist[best_offset] = hist.get(best_offset, 0) + 1
        rows.append(
            {
                "frame_id": frame_id,
                "best_offset": best_offset,
                "best_score": best,
                "best_iou": best_iou,
                "best_points": best_points,
                "iou_before": base_iou,
                "iou_after": best_iou,
                "points_before": base_points,
                "points_after": best_points,
            }
        )
    return rows, hist


def _merge_config(base: dict, overrides: dict) -> dict:
    out = dict(base)
    for key, val in overrides.items():
        if val is None:
            continue
        out[key] = val
    return out


def _build_index_records(
    kitti_root: Path,
    drive_id: str,
    frame_start: int,
    frame_end: int,
    camera: str,
) -> List[dict]:
    records: List[dict] = []
    image_dir = _find_image_dir(kitti_root, drive_id, camera)
    image_by_frame: Dict[str, str] = {}
    if image_dir:
        for path in _list_images(image_dir):
            frame_id = _extract_frame_id(path)
            image_by_frame[_normalize_frame_id(frame_id)] = str(path)
    for frame in range(frame_start, frame_end + 1):
        frame_id = _normalize_frame_id(str(frame))
        records.append(
            {
                "drive_id": drive_id,
                "camera": camera,
                "frame_id": frame_id,
                "image_path": image_by_frame.get(frame_id, ""),
                "scene_profile": "car",
            }
        )
    return records


def _write_index(records: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    with out_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _list_layers(path: Path) -> List[str]:
    try:
        import pyogrio

        return [name for name, _ in pyogrio.list_layers(path)]
    except Exception:
        pass
    try:
        return list(gpd.io.file.fiona.listlayers(str(path)))
    except Exception:
        return []


def _assert_frame_evidence_frame_id(
    frame_evidence_path: Path,
    drive_id: str,
    frame_start: int,
    frame_end: int,
    out_path: Path,
    min_ratio: float = 0.7,
    min_records: int = 5,
) -> None:
    layers = _list_layers(frame_evidence_path) or [None]
    lines = []
    failed = False
    for layer in layers:
        gdf = gpd.read_file(frame_evidence_path, layer=layer) if layer else gpd.read_file(frame_evidence_path)
        layer_name = layer or "default"
        if gdf.empty:
            lines.append(f"{layer_name},records=0,distinct=0,ratio=1.0")
            continue
        if "drive_id" in gdf.columns:
            gdf = gdf[gdf["drive_id"].astype(str) == drive_id]
        if "frame_id" not in gdf.columns:
            lines.append(f"{layer_name},records={len(gdf)},distinct=0,ratio=0.0,missing_frame_id=1")
            failed = True
            continue
        parsed = gdf["frame_id"].apply(_parse_frame_id)
        mask = parsed.notna()
        mask &= parsed >= frame_start
        mask &= parsed <= frame_end
        gdf = gdf.loc[mask]
        total = len(gdf)
        distinct = gdf["frame_id"].nunique() if total else 0
        ratio = (distinct / total) if total else 1.0
        lines.append(f"{layer_name},records={total},distinct={distinct},ratio={ratio:.3f}")
        if total >= min_records and ratio < min_ratio:
            failed = True
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failed:
        raise RuntimeError(f"frame_id_distinct_ratio_low: {out_path}")


def _assert_frame_evidence_hit_frames(
    frame_evidence_path: Path,
    drive_id: str,
    hit_frames: List[str],
    out_path: Path,
    min_ratio: float = 0.7,
    min_hits: int = 5,
) -> None:
    hit_norm = {_normalize_frame_id(f) for f in hit_frames if _normalize_frame_id(f)}
    total_hits = len(hit_norm)
    lines = [f"hit_frames_total={total_hits}"]
    if total_hits == 0:
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    layers = _list_layers(frame_evidence_path) or [None]
    target_layers = [l for l in layers if l == "crosswalk_frame"] or layers
    best_ratio = 0.0
    failed = False
    for layer in target_layers:
        gdf = gpd.read_file(frame_evidence_path, layer=layer) if layer else gpd.read_file(frame_evidence_path)
        layer_name = layer or "default"
        if gdf.empty:
            lines.append(f"{layer_name},records=0,distinct_hits=0,ratio=0.0,missing_frame_id=0")
            continue
        if "drive_id" in gdf.columns:
            gdf = gdf[gdf["drive_id"].astype(str) == drive_id]
        if "frame_id" not in gdf.columns:
            lines.append(f"{layer_name},records={len(gdf)},distinct_hits=0,ratio=0.0,missing_frame_id=1")
            failed = True
            continue
        gdf = gdf.copy()
        gdf["frame_id_norm"] = gdf["frame_id"].apply(_normalize_frame_id)
        distinct_hits = len(set(gdf["frame_id_norm"].tolist()) & hit_norm)
        ratio = distinct_hits / max(1, total_hits)
        best_ratio = max(best_ratio, ratio)
        lines.append(f"{layer_name},records={len(gdf)},distinct_hits={distinct_hits},ratio={ratio:.3f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if total_hits >= min_hits and best_ratio < min_ratio:
        failed = True
    if failed:
        raise RuntimeError(f"frame_id_hit_ratio_low: {out_path}")


def _find_crosswalk_layer(path: Path) -> str:
    if not path.exists():
        return ""
    for name in _list_layers(path):
        if "crosswalk" in name.lower():
            return name
    return ""


def _load_crosswalk_raw(
    feature_store_root: Path,
    drive_id: str,
    frame_id: str,
    cache: Dict[Tuple[str, str], Tuple[gpd.GeoDataFrame, float, str]],
) -> Tuple[gpd.GeoDataFrame, float, str]:
    key = (drive_id, frame_id)
    if key in cache:
        return cache[key]
    gpkg_path = feature_store_root / "feature_store" / drive_id / frame_id / "image_features.gpkg"
    if not gpkg_path.exists():
        cache[key] = (gpd.GeoDataFrame(), 0.0, "missing_feature_store")
        return cache[key]
    layer = _find_crosswalk_layer(gpkg_path)
    if not layer:
        cache[key] = (gpd.GeoDataFrame(), 0.0, "missing_layer")
        return cache[key]
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer)
    except Exception:
        gdf = gpd.GeoDataFrame()
    score = 0.0
    if not gdf.empty:
        for col in ("conf", "score", "confidence"):
            if col in gdf.columns:
                vals = []
                for v in gdf[col].tolist():
                    if v is None:
                        continue
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if np.isnan(fv):
                        continue
                    vals.append(fv)
                if vals:
                    score = max(vals)
                    break
    status = "ok" if not gdf.empty else "no_crosswalk"
    cache[key] = (gdf, score, status)
    return cache[key]


def _project_world_to_image(
    points: np.ndarray,
    pose: Tuple[float, ...],
    calib: Dict[str, np.ndarray],
) -> np.ndarray:
    if len(pose) == 6:
        x0, y0, z0, roll, pitch, yaw = pose
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
        delta = points - np.array([x0, y0, z0], dtype=float)
        pose_xyz = (r_world_pose.T @ delta.T).T
        x_ego = pose_xyz[:, 0]
        y_ego = pose_xyz[:, 1]
        z_ego = pose_xyz[:, 2]
    else:
        x0, y0, yaw = pose
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        dx = points[:, 0] - x0
        dy = points[:, 1] - y0
        x_ego = c * dx + s * dy
        y_ego = -s * dx + c * dy
        z_ego = points[:, 2]
    ones = np.ones_like(x_ego)
    pts_h = np.stack([x_ego, y_ego, z_ego, ones], axis=0)
    cam = calib["t_velo_to_cam"] @ pts_h
    xyz = cam[:3, :].T
    xyz = (calib["r_rect"] @ xyz.T).T
    zs = xyz[:, 2]
    valid = zs > 1e-3
    us = np.zeros_like(zs)
    vs = np.zeros_like(zs)
    k = calib["k"]
    us[valid] = (k[0, 0] * xyz[valid, 0] / zs[valid]) + k[0, 2]
    vs[valid] = (k[1, 1] * xyz[valid, 1] / zs[valid]) + k[1, 2]
    return np.stack([us, vs, valid], axis=1)


def _pose_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    c1 = float(np.cos(yaw))
    s1 = float(np.sin(yaw))
    c2 = float(np.cos(pitch))
    s2 = float(np.sin(pitch))
    c3 = float(np.cos(roll))
    s3 = float(np.sin(roll))
    r_z = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    r_y = np.array([[c2, 0.0, s2], [0.0, 1.0, 0.0], [-s2, 0.0, c2]], dtype=float)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, c3, -s3], [0.0, s3, c3]], dtype=float)
    return r_z @ r_y @ r_x


def _pixel_to_world(
    u: float,
    v: float,
    calib: Dict[str, np.ndarray],
    pose: Tuple[float, ...],
    cam_to_pose: Optional[np.ndarray],
) -> Optional[Tuple[float, float]]:
    k = calib["k"]
    r_rect = calib["r_rect"]
    cam_to_velo = calib["t_cam_to_velo"]
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    if fx == 0 or fy == 0:
        return None
    dir_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)
    r_rect_inv = np.linalg.inv(r_rect)
    dir_cam = r_rect_inv.dot(dir_cam)
    if len(pose) == 6 and cam_to_pose is not None:
        x, y, z, roll, pitch, yaw = pose
        dir_pose = cam_to_pose[:3, :3].dot(dir_cam)
        r_world_pose = _pose_rotation_matrix(roll, pitch, yaw)
        dir_world = r_world_pose.dot(dir_pose)
        origin_pose = cam_to_pose[:3, 3]
        origin_world = np.array([x, y, z], dtype=float) + r_world_pose.dot(origin_pose)
        if dir_world[2] < -1e-6:
            t = -origin_world[2] / dir_world[2]
            if t > 0:
                hit = origin_world + t * dir_world
                return float(hit[0]), float(hit[1])
    if len(pose) < 3:
        return None
    pose_xy = (pose[0], pose[1])
    yaw = pose[5] if len(pose) >= 6 else pose[2]
    dir_velo = cam_to_velo[:3, :3].dot(dir_cam)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    r_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    dir_world = r_yaw.dot(dir_velo)
    cam_offset = cam_to_velo[:3, 3]
    origin_z = float(abs(cam_offset[2]))
    origin = np.array(
        [
            pose_xy[0] + c * cam_offset[0] - s * cam_offset[1],
            pose_xy[1] + s * cam_offset[0] + c * cam_offset[1],
            origin_z,
        ],
        dtype=float,
    )
    if dir_world[2] >= -1e-6:
        return None
    t = -origin[2] / dir_world[2]
    if t <= 0:
        return None
    hit = origin + t * dir_world
    return float(hit[0]), float(hit[1])


def _project_geometry_ground_plane(
    geom: Polygon | LineString,
    calib: Dict[str, np.ndarray],
    pose: Tuple[float, ...],
    cam_to_pose: Optional[np.ndarray],
) -> Optional[Polygon]:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type in {"MultiPolygon", "MultiLineString", "GeometryCollection"}:
        polys = []
        for sub in geom.geoms:
            poly = _project_geometry_ground_plane(sub, calib, pose, cam_to_pose)
            if poly is not None and not poly.is_empty:
                polys.append(poly)
        if not polys:
            return None
        try:
            merged = unary_union(polys)
        except Exception:
            merged = polys[0]
        if merged is None or merged.is_empty:
            return None
        if merged.geom_type == "Polygon":
            return merged
        try:
            merged = merged.convex_hull
        except Exception:
            return None
        return merged if merged is not None and not merged.is_empty else None
    if geom.geom_type == "Polygon":
        coords = list(geom.exterior.coords)
    else:
        coords = list(geom.coords)
    pts = []
    for u, v in coords:
        hit = _pixel_to_world(float(u), float(v), calib, pose, cam_to_pose)
        if hit is not None:
            pts.append(hit)
    if len(pts) < 3:
        return None
    try:
        poly = Polygon(pts)
    except Exception:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly is None or poly.is_empty:
        return None
    return poly


def _dbscan_largest_cluster(points_xy: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    if points_xy.shape[0] == 0:
        return np.array([], dtype=int)
    n = points_xy.shape[0]
    dist = np.linalg.norm(points_xy[:, None, :] - points_xy[None, :, :], axis=2)
    neighbors = dist <= eps
    visited = np.zeros(n, dtype=bool)
    clusters: List[List[int]] = []
    for idx in range(n):
        if visited[idx]:
            continue
        visited[idx] = True
        neigh = np.where(neighbors[idx])[0].tolist()
        if len(neigh) < min_samples:
            continue
        cluster = []
        queue = neigh[:]
        while queue:
            cur = queue.pop()
            if cur not in cluster:
                cluster.append(cur)
            if not visited[cur]:
                visited[cur] = True
                cur_neigh = np.where(neighbors[cur])[0].tolist()
                if len(cur_neigh) >= min_samples:
                    for nidx in cur_neigh:
                        if nidx not in cluster:
                            queue.append(nidx)
        clusters.append(cluster)
    if not clusters:
        return np.array([], dtype=int)
    largest = max(clusters, key=len)
    return np.array(sorted(set(largest)), dtype=int)


def _rect_angle_rad(rect: Polygon) -> float:
    if rect is None:
        return 0.0
    if rect.geom_type == "LineString":
        coords = list(rect.coords)
    else:
        coords = list(rect.exterior.coords)
    if len(coords) < 2:
        return 0.0
    edges = []
    for i in range(len(coords) - 1):
        p0 = coords[i]
        p1 = coords[i + 1]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        edges.append((np.hypot(dx, dy), np.arctan2(dy, dx)))
    edges.sort(reverse=True, key=lambda x: x[0])
    return float(edges[0][1]) if edges else 0.0


def _align_rect_to_heading(rect: Polygon, heading_rad: float, snap_deg: float) -> Polygon:
    if rect is None or rect.is_empty:
        return rect
    target = heading_rad + np.pi / 2.0
    angle = _rect_angle_rad(rect)
    diff = (target - angle + np.pi) % (2 * np.pi) - np.pi
    snap = np.deg2rad(float(snap_deg))
    if abs(diff) > snap:
        diff = np.sign(diff) * snap
    return affinity.rotate(rect, np.degrees(diff), origin=rect.centroid)


def _rect_metrics(geom: Polygon) -> dict:
    if geom is None or geom.is_empty:
        return {
            "rect_w_m": 0.0,
            "rect_l_m": 0.0,
            "aspect": 0.0,
            "rectangularity": 0.0,
            "geom_area_m2": 0.0,
            "rect_area_m2": 0.0,
        }
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom is None or geom.is_empty:
        return {
            "rect_w_m": 0.0,
            "rect_l_m": 0.0,
            "aspect": 0.0,
            "rectangularity": 0.0,
            "geom_area_m2": 0.0,
            "rect_area_m2": 0.0,
        }
    rect = geom.minimum_rotated_rectangle
    if rect is None or rect.is_empty:
        return {
            "rect_w_m": 0.0,
            "rect_l_m": 0.0,
            "aspect": 0.0,
            "rectangularity": 0.0,
            "geom_area_m2": float(geom.area),
            "rect_area_m2": 0.0,
        }
    if not rect.is_valid:
        rect = rect.buffer(0)
    rect_area = rect.area if rect is not None else 0.0
    area = geom.area if geom is not None else 0.0
    rectangularity = area / rect_area if rect_area > 0 else 0.0
    rect_coords = list(rect.exterior.coords) if rect is not None and rect.geom_type == "Polygon" else []
    edge_lengths = []
    for i in range(len(rect_coords) - 1):
        p0 = rect_coords[i]
        p1 = rect_coords[i + 1]
        length = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        if length > 1e-6:
            edge_lengths.append(length)
    rect_l = max(edge_lengths) if edge_lengths else 0.0
    rect_w = min(edge_lengths) if edge_lengths else 0.0
    aspect = rect_l / max(1e-6, rect_w) if rect_w > 0 else 0.0
    return {
        "rect_w_m": rect_w,
        "rect_l_m": rect_l,
        "aspect": aspect,
        "rectangularity": rectangularity,
        "geom_area_m2": float(area),
        "rect_area_m2": float(rect_area),
    }


def _angle_diff_to_heading_deg(geom: Polygon, heading_rad: float) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    rect = geom.minimum_rotated_rectangle
    if rect is None or rect.is_empty:
        return 0.0
    if not rect.is_valid:
        rect = rect.buffer(0)
    if rect is None or rect.is_empty:
        return 0.0
    angle = _rect_angle_rad(rect)
    target = heading_rad + np.pi / 2.0
    diff = (angle - target + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(abs(diff)))


def _load_image_size(image_path: str) -> Tuple[int, int]:
    if image_path and Path(image_path).exists():
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception:
            pass
    return 1280, 720


def _lidar_points_world_with_intensity(
    data_root: Path,
    drive_id: str,
    frame_id: str,
    pose: Tuple[float, float, float] | None,
    lidar_world_mode: str,
    cam_id: str,
) -> Tuple[np.ndarray, np.ndarray]:
    lidar_world_mode = str(lidar_world_mode or "legacy").lower()
    try:
        pts = load_kitti360_lidar_points(data_root, drive_id, frame_id)
    except Exception:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    if pts.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    if lidar_world_mode == "fullpose":
        try:
            world = load_kitti360_lidar_points_world(
                data_root,
                drive_id,
                frame_id,
                mode="fullpose",
                cam_id=cam_id,
            )
            return world, pts[:, 3].astype(float)
        except Exception:
            return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    if pose is None:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    x0, y0, yaw = pose
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    xyz = pts[:, :3]
    xw = c * xyz[:, 0] - s * xyz[:, 1] + x0
    yw = s * xyz[:, 0] + c * xyz[:, 1] + y0
    zw = xyz[:, 2]
    return np.stack([xw, yw, zw], axis=1), pts[:, 3].astype(float)


def _render_text_overlay(
    image_path: str,
    out_path: Path,
    lines: List[str],
) -> None:
    if out_path.exists():
        out_path.unlink()
    if image_path and Path(image_path).exists():
        try:
            base = Image.open(image_path).convert("RGBA")
        except Exception:
            base = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    else:
        base = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    draw = ImageDraw.Draw(base, "RGBA")
    y = 10
    for line in lines:
        draw.text((10, y), line, fill=(255, 128, 0, 220))
        y += 24
    base.save(out_path)


def _render_raw_overlay(
    image_path: str,
    raw_gdf: gpd.GeoDataFrame,
    out_path: Path,
    missing_status: str,
    raw_fallback_text: bool,
) -> None:
    if out_path.exists():
        out_path.unlink()
    if image_path and Path(image_path).exists():
        try:
            base = Image.open(image_path).convert("RGBA")
        except Exception:
            base = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    else:
        base = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    draw = ImageDraw.Draw(base, "RGBA")
    if raw_fallback_text and missing_status and missing_status != "ok":
        draw.text((10, 10), f"MISSING_FEATURE_STORE:{missing_status}", fill=(255, 128, 0, 220))
        draw.text((10, 36), "NO_RAW_PRED", fill=(255, 0, 0, 220))
        base.save(out_path)
        return
    if raw_gdf is None or raw_gdf.empty:
        if missing_status and missing_status != "ok":
            draw.text((10, 10), f"MISSING_FEATURE_STORE:{missing_status}", fill=(255, 128, 0, 220))
        draw.text((10, 36), "NO_CROSSWALK_DETECTED", fill=(255, 0, 0, 220))
        base.save(out_path)
        return
    for _, row in raw_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            coords = [(float(x), float(y)) for x, y in geom.exterior.coords]
            if len(coords) >= 2:
                draw.line(coords, fill=(255, 0, 0, 220), width=3)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                coords = [(float(x), float(y)) for x, y in poly.exterior.coords]
                if len(coords) >= 2:
                    draw.line(coords, fill=(255, 0, 0, 220), width=3)
        elif geom.geom_type == "LineString":
            coords = [(float(x), float(y)) for x, y in geom.coords]
            if len(coords) >= 2:
                draw.line(coords, fill=(255, 0, 0, 220), width=3)
        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                coords = [(float(x), float(y)) for x, y in line.coords]
                if len(coords) >= 2:
                    draw.line(coords, fill=(255, 0, 0, 220), width=3)
    base.save(out_path)


def _build_lidar_candidate_for_frame(
    data_root: Path,
    drive_id: str,
    frame_id: str,
    image_path: str,
    raw_info: dict,
    pose: Tuple[float, float, float] | None,
    calib: Dict[str, np.ndarray] | None,
    cam_to_pose: Optional[np.ndarray],
    lidar_cfg: dict,
    lidar_world_mode: str,
    cam_id: str,
) -> Tuple[dict | None, dict]:
    allow_candidate = True
    stats = {
        "proj_method": "none",
        "pose_ok": 1 if pose is not None else 0,
        "calib_ok": 1 if calib is not None else 0,
        "proj_in_image_ratio": 0.0,
        "proj_in_image_ratio_visible": 0.0,
        "points_total": 0,
        "points_in_bbox": 0,
        "points_in_mask": 0,
        "points_in_mask_dilated": 0,
        "points_intensity_top": 0,
        "points_ground": 0,
        "points_ground_global": 0,
        "points_ground_local": 0,
        "points_ground_plane": 0,
        "points_support": 0,
        "mask_dilate_px": int(lidar_cfg.get("MASK_DILATE_PX", 5)),
        "mask_dilate_px_used": 0,
        "intensity_top_pct": 0,
        "intensity_top_pct_used": 0,
        "ground_filter_used": 0,
        "ground_z": 0.0,
        "ground_z_global": 0.0,
        "ground_z_local": 0.0,
        "ground_filter_mode": "",
        "support_source": "",
        "plane_ok": 0,
        "plane_dist_p10": None,
        "plane_dist_p50": None,
        "plane_dist_p90": None,
        "z_p01": None,
        "z_p10": None,
        "z_p50": None,
        "z_p90": None,
        "z_p99": None,
        "dz_p01": None,
        "dz_p10": None,
        "dz_p50": None,
        "dz_p90": None,
        "dz_p99": None,
        "dbscan_points": 0,
        "geom_ok": 0,
        "geom_area_m2": 0.0,
        "rect_w_m": 0.0,
        "rect_l_m": 0.0,
        "rectangularity": 0.0,
        "geom_ok_accum": 0,
        "geom_area_m2_accum": 0.0,
        "rect_w_m_accum": 0.0,
        "rect_l_m_accum": 0.0,
        "rectangularity_accum": 0.0,
        "rect_w_m_single": 0.0,
        "rect_l_m_single": 0.0,
        "rectangularity_single": 0.0,
        "geom_area_m2_single": 0.0,
        "drop_reason_code": "GEOM_INVALID",
        "accum_frames_used": 1,
        "points_support_accum": 0,
        "support_points": np.empty((0, 2), dtype=float),
        "support_bbox_w_m": 0.0,
        "support_bbox_l_m": 0.0,
    }
    raw_gdf = raw_info.get("gdf") if raw_info else None
    bbox_px = raw_info.get("bbox_px") if raw_info else None
    if raw_gdf is None or raw_gdf.empty or pose is None or calib is None:
        if raw_gdf is None or raw_gdf.empty:
            stats["drop_reason_code"] = "GEOM_INVALID"
        elif pose is None:
            stats["drop_reason_code"] = "PLANE_FALLBACK_USED"
        elif calib is None:
            stats["drop_reason_code"] = "LIDAR_CALIB_MISMATCH"
        return None, stats
    pose_full = None
    if str(lidar_world_mode).lower() == "fullpose":
        try:
            pose_full = load_kitti360_pose_full(data_root, drive_id, frame_id)
        except Exception:
            stats["drop_reason_code"] = "POSE_MISSING"
            return None, stats

    points_world, intensities = _lidar_points_world_with_intensity(
        data_root,
        drive_id,
        frame_id,
        pose,
        lidar_world_mode,
        cam_id,
    )
    if points_world.size == 0:
        stats["drop_reason_code"] = "LIDAR_NO_POINTS_BBOX"
        return None, stats

    z_vals = points_world[:, 2]
    z_ground = float(np.percentile(z_vals, 10)) if z_vals.size > 0 else 0.0
    stats["ground_z"] = z_ground
    stats["ground_z_global"] = z_ground
    ground_tol = float(lidar_cfg.get("GROUND_Z_TOL", 0.2))
    ground_mask = np.abs(z_vals - z_ground) < ground_tol
    stats["ground_filter_used"] = 1 if np.any(ground_mask) else 0

    proj_pose = pose_full if pose_full is not None else pose
    proj = _project_world_to_image(points_world, proj_pose, calib)
    u = proj[:, 0]
    v = proj[:, 1]
    valid = proj[:, 2].astype(bool)
    width, height = _load_image_size(image_path)
    in_image = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    stats["points_total"] = int(points_world.shape[0])
    stats["proj_in_image_ratio"] = float(np.mean(in_image)) if points_world.shape[0] > 0 else 0.0
    x_forward_only = bool(lidar_cfg.get("VISIBLE_X_FORWARD_ONLY", True))
    min_range = float(lidar_cfg.get("VISIBLE_MIN_RANGE_M", 1.0))
    max_range = float(lidar_cfg.get("VISIBLE_MAX_RANGE_M", 80.0))
    c = float(np.cos(pose[2]))
    s = float(np.sin(pose[2]))
    dx = points_world[:, 0] - pose[0]
    dy = points_world[:, 1] - pose[1]
    x_ego = c * dx + s * dy
    y_ego = -s * dx + c * dy
    dist = np.hypot(x_ego, y_ego)
    visible = np.ones(points_world.shape[0], dtype=bool)
    if x_forward_only:
        visible &= x_ego > 0
    if min_range > 0:
        visible &= dist >= min_range
    if max_range > 0:
        visible &= dist <= max_range
    stats["points_total_visible"] = int(np.sum(visible))
    stats["points_in_image_visible"] = int(np.sum(in_image & visible))
    if points_world.shape[0] > 0 and np.any(visible):
        stats["proj_in_image_ratio_visible"] = float(np.mean(in_image[visible]))

    if not (isinstance(bbox_px, (list, tuple)) and len(bbox_px) == 4):
        bounds = raw_gdf.total_bounds if hasattr(raw_gdf, "total_bounds") else None
        if bounds is not None and len(bounds) == 4:
            bbox_px = [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])]
    if not (isinstance(bbox_px, (list, tuple)) and len(bbox_px) == 4):
        stats["drop_reason_code"] = "LIDAR_NO_POINTS_BBOX"
        return None, stats
    minx, miny, maxx, maxy = bbox_px
    if minx < 0 or miny < 0 or maxx > width or maxy > height:
        stats["drop_reason_code"] = "CALIB_IMAGE_SIZE_MISMATCH"
        return None, stats

    minx, miny, maxx, maxy = bbox_px
    in_bbox = in_image & visible & (u >= minx) & (u <= maxx) & (v >= miny) & (v <= maxy)
    stats["points_in_bbox"] = int(np.sum(in_bbox))
    min_ratio = float(lidar_cfg.get("MIN_IN_IMAGE_RATIO", 0.1))
    if stats["proj_in_image_ratio"] < min_ratio:
        valid_ratio = float(np.mean(valid)) if points_world.shape[0] > 0 else 0.0
        if valid_ratio < 0.2:
            stats["drop_reason_code"] = "CALIB_POINTS_BEHIND_CAMERA"
        elif stats["points_in_image_visible"] == 0:
            stats["drop_reason_code"] = "CALIB_PROJ_OUTSIDE"
        else:
            stats["drop_reason_code"] = "LIDAR_CALIB_MISMATCH"
        allow_candidate = False

    min_points_bbox = int(lidar_cfg.get("MIN_POINTS_BBOX", 20))
    if stats["points_in_bbox"] < min_points_bbox:
        stats["drop_reason_code"] = "LIDAR_NO_POINTS_BBOX"
        allow_candidate = False
    bbox_metrics = None
    bbox_world_pts = points_world[in_bbox][:, :2]
    if bbox_world_pts.shape[0] >= 3:
        bbox_hull = MultiPoint([Point(float(x), float(y)) for x, y in bbox_world_pts]).convex_hull
        if bbox_hull is not None and not bbox_hull.is_empty:
            bbox_metrics = _rect_metrics(bbox_hull)

    mask_geom = raw_gdf.geometry.union_all() if not raw_gdf.empty else None
    mask_world_geom = None
    if mask_geom is not None and not mask_geom.is_empty:
        pose_for_world = pose_full if pose_full is not None else pose
        if pose_for_world is not None and calib is not None:
            mask_world_geom = _project_geometry_ground_plane(mask_geom, calib, pose_for_world, cam_to_pose)
    if mask_geom is None or mask_geom.is_empty:
        stats["drop_reason_code"] = "LIDAR_NO_POINTS_MASK"
        return None, stats
    dilate_px = int(lidar_cfg.get("MASK_DILATE_PX", 5))
    if dilate_px > 0:
        mask_geom = mask_geom.buffer(float(dilate_px))
    stats["mask_dilate_px_used"] = dilate_px
    mask_hits = []
    for idx, ok in enumerate(in_bbox):
        if not ok:
            continue
        if mask_geom.contains(Point(float(u[idx]), float(v[idx]))):
            mask_hits.append(idx)
    stats["points_in_mask_dilated"] = int(len(mask_hits))
    stats["points_in_mask"] = stats["points_in_mask_dilated"]
    intensity_pct = int(lidar_cfg.get("INTENSITY_TOP_PCT", 10))
    stats["intensity_top_pct"] = intensity_pct
    stats["intensity_top_pct_used"] = intensity_pct
    bbox_indices = np.where(in_bbox)[0]
    if bbox_indices.size == 0:
        stats["drop_reason_code"] = "LIDAR_NO_POINTS_BBOX"
        return None, stats
    base_indices = mask_hits if mask_hits else bbox_indices.tolist()
    if base_indices:
        base_z = z_vals[base_indices]
        stats["z_p01"] = float(np.percentile(base_z, 1))
        stats["z_p10"] = float(np.percentile(base_z, 10))
        stats["z_p50"] = float(np.percentile(base_z, 50))
        stats["z_p90"] = float(np.percentile(base_z, 90))
        stats["z_p99"] = float(np.percentile(base_z, 99))
        dz = base_z - z_ground
        stats["dz_p01"] = float(np.percentile(dz, 1))
        stats["dz_p10"] = float(np.percentile(dz, 10))
        stats["dz_p50"] = float(np.percentile(dz, 50))
        stats["dz_p90"] = float(np.percentile(dz, 90))
        stats["dz_p99"] = float(np.percentile(dz, 99))
    vals = intensities[base_indices]
    thr = np.percentile(vals, 100 - intensity_pct) if vals.size > 0 else None
    intensity_idx = [base_indices[i] for i in range(len(base_indices)) if thr is None or vals[i] >= thr]
    stats["points_intensity_top"] = int(len(intensity_idx))
    ground_idx_global = [idx for idx in intensity_idx if ground_mask[idx]]
    stats["points_ground"] = int(len(ground_idx_global))
    stats["points_ground_global"] = int(len(ground_idx_global))
    support_idx = ground_idx_global
    ground_idx_local = []
    if intensity_idx:
        z_local = float(np.percentile(z_vals[intensity_idx], 10))
        stats["ground_z_local"] = z_local
        ground_idx_local = [idx for idx in intensity_idx if abs(z_vals[idx] - z_local) < ground_tol]
        stats["points_ground_local"] = int(len(ground_idx_local))
    ground_idx_plane = []
    if stats["points_ground_local"] > 0:
        support_idx = ground_idx_local
        stats["ground_filter_mode"] = "local"
        stats["ground_filter_used"] = 1
        stats["support_source"] = "ground_local"
    else:
        if len(intensity_idx) >= 3:
            xs = x_ego[intensity_idx]
            ys = y_ego[intensity_idx]
            zs = z_vals[intensity_idx]
            a_mat = np.column_stack([xs, ys, np.ones_like(xs)])
            try:
                coeff, _, _, _ = np.linalg.lstsq(a_mat, zs, rcond=None)
                a, b, c0 = coeff.tolist()
                denom = math.sqrt(a * a + b * b + 1.0)
                dists = np.abs((a * xs + b * ys + c0) - zs) / denom
                stats["plane_dist_p10"] = float(np.percentile(dists, 10))
                stats["plane_dist_p50"] = float(np.percentile(dists, 50))
                stats["plane_dist_p90"] = float(np.percentile(dists, 90))
                ground_idx_plane = [intensity_idx[i] for i, d in enumerate(dists) if d < ground_tol]
            except Exception:
                ground_idx_plane = []
        stats["points_ground_plane"] = int(len(ground_idx_plane))
        if stats["points_ground_plane"] > 0:
            support_idx = ground_idx_plane
            stats["ground_filter_mode"] = "plane"
            stats["ground_filter_used"] = 1
            stats["plane_ok"] = 1
            stats["support_source"] = "ground_plane"
        else:
            support_idx = []
    min_points_mask = int(lidar_cfg.get("MIN_POINTS_MASK", 5))
    if len(support_idx) < min_points_mask:
        if stats.get("points_ground_local", 0) > 0:
            stats["support_source"] = "ground_local_fallback"
        elif stats.get("points_in_bbox", 0) > 0:
            stats["support_source"] = "bbox_intensity_fallback"
            stats["points_support"] = int(stats["points_intensity_top"])
            stats["drop_reason_code"] = "LIDAR_NO_POINTS_GROUND"
            return None, stats
        else:
            stats["drop_reason_code"] = "LIDAR_NO_POINTS_BBOX"
            return None, stats

    support_pts_all = points_world[support_idx][:, :2]
    support_pts = support_pts_all
    eps = float(lidar_cfg.get("DBSCAN_EPS_M", 0.6))
    min_samples = int(lidar_cfg.get("DBSCAN_MIN_SAMPLES", 10))
    cluster_idx = _dbscan_largest_cluster(support_pts_all, eps, min_samples)
    if cluster_idx.size > 0:
        support_pts = support_pts_all[cluster_idx]
        stats["dbscan_points"] = int(cluster_idx.size)
        if stats["support_source"] not in {
            "ground_local_fallback",
            "ground_plane_fallback",
            "bbox_intensity_fallback",
        }:
            stats["support_source"] = "dbscan_cluster"
    else:
        stats["dbscan_points"] = int(support_pts_all.shape[0])
        if stats.get("points_ground_local", 0) > 0:
            stats["support_source"] = "ground_local_fallback"
        elif stats.get("points_ground_plane", 0) > 0:
            stats["support_source"] = "ground_plane_fallback"
        elif not stats["support_source"]:
            stats["support_source"] = "dbscan_empty"
    stats["points_support"] = int(support_pts.shape[0])
    if support_pts_all.shape[0] < 3:
        stats["drop_reason_code"] = "GEOM_INVALID"
        return None, stats
    hull = MultiPoint([Point(float(x), float(y)) for x, y in support_pts_all]).convex_hull
    rect = hull.minimum_rotated_rectangle
    if rect is None or rect.is_empty:
        stats["drop_reason_code"] = "GEOM_INVALID"
        return None, stats
    if not rect.is_valid:
        rect = rect.buffer(0)
    if rect is None or rect.is_empty or not rect.is_valid:
        stats["drop_reason_code"] = "GEOM_INVALID"
        return None, stats
    if rect.geom_type != "Polygon":
        stats["drop_reason_code"] = "GEOM_INVALID"
        return None, stats
    rect = _align_rect_to_heading(rect, pose[2], float(lidar_cfg.get("ANGLE_SNAP_DEG", 20)))
    metrics = _rect_metrics(rect)

    stats["proj_method"] = "lidar" if allow_candidate else stats["proj_method"]
    stats["geom_ok"] = 1 if allow_candidate else 0
    stats["geom_area_m2"] = float(metrics["geom_area_m2"]) if allow_candidate else 0.0
    stats["rect_w_m"] = float(metrics["rect_w_m"]) if allow_candidate else 0.0
    stats["rect_l_m"] = float(metrics["rect_l_m"]) if allow_candidate else 0.0
    stats["rectangularity"] = float(metrics["rectangularity"]) if allow_candidate else 0.0
    stats["geom_area_m2_single"] = stats["geom_area_m2"]
    stats["rect_w_m_single"] = stats["rect_w_m"]
    stats["rect_l_m_single"] = stats["rect_l_m"]
    stats["rectangularity_single"] = stats["rectangularity"]
    stats["support_bbox_w_m"] = stats["rect_w_m"]
    stats["support_bbox_l_m"] = stats["rect_l_m"]
    if mask_world_geom is not None:
        mask_metrics = _rect_metrics(mask_world_geom)
        if mask_metrics["rect_w_m"] > 0 and mask_metrics["rect_l_m"] > 0:
            stats["rect_w_m"] = float(mask_metrics["rect_w_m"])
            stats["rect_l_m"] = float(mask_metrics["rect_l_m"])
            stats["rectangularity"] = float(mask_metrics["rectangularity"])
    if bbox_metrics is not None and (stats["rect_w_m"] < 0.5 or stats["rect_l_m"] < 1.0):
        stats["rect_w_m"] = float(bbox_metrics["rect_w_m"])
        stats["rect_l_m"] = float(bbox_metrics["rect_l_m"])
        stats["rectangularity"] = float(bbox_metrics["rectangularity"])
    if allow_candidate:
        stats["drop_reason_code"] = "LIDAR_OK"
    stats["support_points"] = support_pts
    if not allow_candidate:
        return None, stats
    candidate = {
        "geometry": rect,
        "properties": {
            "candidate_id": f"{drive_id}_crosswalk_lidar_{frame_id}",
            "drive_id": drive_id,
            "frame_id": frame_id,
            "source_frame_id": frame_id,
            "entity_type": "crosswalk",
            "reject_reasons": "",
            "proj_method": "lidar",
            "points_in_bbox": stats["points_in_bbox"],
            "points_in_mask": stats["points_in_mask"],
            "mask_dilate_px": stats["mask_dilate_px"],
            "intensity_top_pct": stats["intensity_top_pct"],
            "mask_dilate_px_used": stats["mask_dilate_px_used"],
            "intensity_top_pct_used": stats["intensity_top_pct_used"],
            "ground_filter_used": stats["ground_filter_used"],
            "dbscan_points": stats["dbscan_points"],
            "support_source": stats["support_source"],
            "drop_reason_code": stats["drop_reason_code"],
            "geom_ok": stats["geom_ok"],
            "geom_area_m2": stats["geom_area_m2"],
            "rect_w_m": stats["rect_w_m"],
            "rect_l_m": stats["rect_l_m"],
            "rectangularity": stats["rectangularity"],
            "bbox_px": bbox_px,
            "qa_flag": "ok",
        },
    }
    return candidate, stats


def _ensure_raw_overlays(
    qa_index_path: Path,
    outputs_dir: Path,
    image_run: Path,
    provider_id: str,
    index_lookup: Dict[Tuple[str, str], str],
    raw_fallback_text: bool,
    on_demand_infer: bool,
    on_demand_root: Path,
    drive_id: str,
    camera: str,
) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], Dict[Tuple[str, str], dict]]:
    if not qa_index_path.exists():
        return {}, {}
    qa_gdf = gpd.read_file(qa_index_path)
    if qa_gdf.empty:
        return {}, {}
    feature_store_root = image_run / f"feature_store_{provider_id}"
    qa_dir = outputs_dir / "qa_images"
    qa_dir.mkdir(parents=True, exist_ok=True)
    raw_cache: Dict[Tuple[str, str], Tuple[gpd.GeoDataFrame, float, str]] = {}
    raw_stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    raw_frames: Dict[Tuple[str, str], dict] = {}
    missing_rows = []
    infer_enabled = bool(on_demand_infer)
    infer_attempts = 0
    max_attempts = 1
    for idx, row in qa_gdf.iterrows():
        drive_id = str(row.get("drive_id") or "")
        frame_id = _normalize_frame_id(str(row.get("frame_id") or ""))
        if not drive_id or not frame_id:
            continue
        image_path = index_lookup.get((drive_id, frame_id), "")
        out_path = qa_dir / drive_id / f"{frame_id}_overlay_raw.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_gdf, raw_score, raw_status = _load_crosswalk_raw(feature_store_root, drive_id, frame_id, raw_cache)
        if raw_status in {"missing_feature_store", "read_error"} and infer_enabled:
            if infer_attempts >= max_attempts:
                raw_status = "on_demand_infer_fail"
                infer_enabled = False
            else:
                infer_attempts += 1
                infer_status, infer_gdf, infer_score, _ = _run_on_demand_infer(
                    on_demand_root,
                    drive_id,
                    frame_id,
                    image_path,
                    camera,
                    provider_id,
                )
                if infer_status == "ok":
                    raw_gdf = infer_gdf
                    raw_score = infer_score
                    raw_status = "on_demand_infer_ok"
                else:
                    raw_status = "on_demand_infer_fail"
                    infer_enabled = False
        raw_stats[(drive_id, frame_id)] = {
            "raw_has_crosswalk": 0.0 if raw_gdf.empty else 1.0,
            "raw_top_score": float(raw_score),
            "raw_status": raw_status,
        }
        if not raw_gdf.empty:
            bounds = raw_gdf.total_bounds if hasattr(raw_gdf, "total_bounds") else None
            bbox_px = None
            if bounds is not None and len(bounds) == 4:
                minx, miny, maxx, maxy = bounds
                bbox_px = [float(minx), float(miny), float(maxx), float(maxy)]
            raw_frames[(drive_id, frame_id)] = {"gdf": raw_gdf.copy(), "bbox_px": bbox_px}
        qa_gdf.at[idx, "raw_has_crosswalk"] = int(0 if raw_gdf.empty else 1)
        qa_gdf.at[idx, "raw_top_score"] = float(raw_score)
        qa_gdf.at[idx, "raw_status"] = raw_status
        if raw_status in {"missing_feature_store", "read_error", "on_demand_infer_fail"}:
            missing_rows.append(
                {
                    "drive_id": drive_id,
                    "frame_id": frame_id,
                    "raw_status": raw_status,
                    "image_path": image_path,
                }
            )
        _render_raw_overlay(image_path, raw_gdf, out_path, raw_status, raw_fallback_text)
        qa_gdf.at[idx, "overlay_raw_path"] = str(out_path)
    for col in qa_gdf.columns:
        if col == "geometry":
            continue
        qa_gdf[col] = qa_gdf[col].apply(lambda v: v.tolist() if isinstance(v, np.ndarray) else v)
    qa_index_path.write_text(qa_gdf.to_json(), encoding="utf-8")
    missing_path = outputs_dir / "missing_feature_store_list.csv"
    pd.DataFrame(
        missing_rows,
        columns=["drive_id", "frame_id", "raw_status", "image_path"],
    ).to_csv(missing_path, index=False)
    return raw_stats, raw_frames


def _build_lidar_candidates_for_range(
    data_root: Path,
    drive_id: str,
    frame_start: int,
    frame_end: int,
    index_lookup: Dict[Tuple[str, str], str],
    raw_frames: Dict[Tuple[str, str], dict],
    pose_map: Dict[str, Tuple[float, float, float] | None],
    calib: Dict[str, np.ndarray] | None,
    cam_to_pose: Optional[np.ndarray],
    lidar_cfg: dict,
    lidar_world_mode: str,
    cam_id: str,
    debug_dir: Path | None = None,
) -> Tuple[List[dict], Dict[str, dict]]:
    candidates = []
    stats_by_frame: Dict[str, dict] = {}
    candidate_by_frame: Dict[str, dict] = {}
    for frame in range(frame_start, frame_end + 1):
        frame_id = _normalize_frame_id(str(frame))
        raw_info = raw_frames.get((drive_id, frame_id))
        if not raw_info:
            continue
        image_path = index_lookup.get((drive_id, frame_id), "")
        cand, stats = _build_lidar_candidate_for_frame(
            data_root,
            drive_id,
            frame_id,
            image_path,
            raw_info,
            pose_map.get(frame_id),
            calib,
            cam_to_pose,
            lidar_cfg,
            lidar_world_mode,
            cam_id,
        )
        stats_by_frame[frame_id] = stats
        if cand:
            candidates.append(cand)
            candidate_by_frame[frame_id] = cand
    accum_pre = int(lidar_cfg.get("ACCUM_PRE", 5))
    accum_post = int(lidar_cfg.get("ACCUM_POST", 5))
    angle_snap = float(lidar_cfg.get("ANGLE_SNAP_DEG", 20))
    min_points_mask = int(lidar_cfg.get("MIN_POINTS_MASK", 5))
    for frame in range(frame_start, frame_end + 1):
        frame_id = _normalize_frame_id(str(frame))
        stats = stats_by_frame.get(frame_id)
        if stats is None:
            continue
        stats["accum_pre"] = accum_pre
        stats["accum_post"] = accum_post
        accum_pts = []
        used_frames = []
        for offset in range(-accum_pre, accum_post + 1):
            cand_frame = frame + offset
            if cand_frame < frame_start or cand_frame > frame_end:
                continue
            cand_id = _normalize_frame_id(str(cand_frame))
            cand_stats = stats_by_frame.get(cand_id)
            if not cand_stats:
                continue
            pts = cand_stats.get("support_points")
            if isinstance(pts, np.ndarray) and pts.size > 0:
                accum_pts.append(pts)
                used_frames.append(cand_id)
        stats["accum_frames_used"] = len(used_frames)
        stats["points_support_accum"] = int(np.sum([p.shape[0] for p in accum_pts])) if accum_pts else 0
        if not accum_pts:
            continue
        merged = np.vstack(accum_pts)
        if merged.shape[0] < 3:
            continue
        pose = pose_map.get(frame_id)
        if pose is None:
            continue
        hull = MultiPoint([Point(float(x), float(y)) for x, y in merged]).convex_hull
        rect = hull.minimum_rotated_rectangle
        if rect is None or rect.is_empty:
            continue
        if not rect.is_valid:
            rect = rect.buffer(0)
        if rect is None or rect.is_empty or not rect.is_valid:
            continue
        rect = _align_rect_to_heading(rect, pose[2], angle_snap)
        metrics = _rect_metrics(rect)
        stats["geom_ok_accum"] = 1
        stats["geom_area_m2_accum"] = float(metrics["geom_area_m2"])
        stats["rect_w_m_accum"] = float(metrics["rect_w_m"])
        stats["rect_l_m_accum"] = float(metrics["rect_l_m"])
        stats["rectangularity_accum"] = float(metrics["rectangularity"])
        if merged.shape[0] >= min_points_mask:
            cand = candidate_by_frame.get(frame_id)
            if cand is None:
                cand = {
                    "geometry": rect,
                    "properties": {
                        "candidate_id": f"{drive_id}_crosswalk_lidar_accum_{frame_id}",
                        "drive_id": drive_id,
                        "frame_id": frame_id,
                        "entity_type": "crosswalk",
                        "reject_reasons": "",
                        "proj_method": "lidar",
                        "points_in_bbox": stats.get("points_in_bbox", 0),
                        "points_in_mask": stats.get("points_in_mask", 0),
                        "mask_dilate_px": stats.get("mask_dilate_px", 0),
                        "intensity_top_pct": stats.get("intensity_top_pct", 0),
                        "ground_filter_used": stats.get("ground_filter_used", 0),
                        "dbscan_points": stats.get("dbscan_points", 0),
                        "drop_reason_code": stats.get("drop_reason_code", ""),
                        "geom_ok": 1,
                        "geom_area_m2": stats["geom_area_m2_accum"],
                        "rect_w_m": stats["rect_w_m_accum"],
                        "rect_l_m": stats["rect_l_m_accum"],
                        "rectangularity": stats["rectangularity_accum"],
                        "bbox_px": raw_frames.get((drive_id, frame_id), {}).get("bbox_px"),
                        "qa_flag": "ok_accum",
                    },
                }
                candidates.append(cand)
                candidate_by_frame[frame_id] = cand
            else:
                cand["geometry"] = rect
                cand["properties"]["geom_area_m2"] = stats["geom_area_m2_accum"]
                cand["properties"]["rect_w_m"] = stats["rect_w_m_accum"]
                cand["properties"]["rect_l_m"] = stats["rect_l_m_accum"]
                cand["properties"]["rectangularity"] = stats["rectangularity_accum"]
                cand["properties"]["qa_flag"] = "ok_accum"
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        for frame in range(frame_start, frame_end + 1):
            frame_id = _normalize_frame_id(str(frame))
            stats = stats_by_frame.get(frame_id)
            if not stats:
                continue
            debug_path = debug_dir / f"point_support_{frame_id}.json"
            payload = {
                "drive_id": drive_id,
                "frame_id": frame_id,
                "accum_pre": accum_pre,
                "accum_post": accum_post,
                "accum_frames_used": int(stats.get("accum_frames_used", 0)),
                "points_in_bbox": int(stats.get("points_in_bbox", 0)),
                "points_in_mask_dilated": int(stats.get("points_in_mask_dilated", 0)),
                "points_intensity_top": int(stats.get("points_intensity_top", 0)),
                "points_ground": int(stats.get("points_ground", 0)),
                "points_support": int(stats.get("points_support", 0)),
                "points_support_accum": int(stats.get("points_support_accum", 0)),
                "proj_method": str(stats.get("proj_method", "")),
                "drop_reason_code": str(stats.get("drop_reason_code", "")),
                "geom_ok_accum": int(stats.get("geom_ok_accum", 0)),
                "geom_area_m2_accum": float(stats.get("geom_area_m2_accum", 0.0)),
                "rect_w_m_accum": float(stats.get("rect_w_m_accum", 0.0)),
                "rect_l_m_accum": float(stats.get("rect_l_m_accum", 0.0)),
                "rectangularity_accum": float(stats.get("rectangularity_accum", 0.0)),
            }
            debug_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    return candidates, stats_by_frame


def _read_candidates(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame()
    try:
        return gpd.read_file(path, layer="crosswalk_candidate_poly")
    except Exception:
        return gpd.GeoDataFrame()


def _read_final(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame()
    try:
        return gpd.read_file(path, layer="crosswalk_poly")
    except Exception:
        return gpd.GeoDataFrame()


def _ensure_wgs84_range(gdf: gpd.GeoDataFrame) -> bool:
    if gdf.empty:
        return True
    try:
        bounds = gdf.total_bounds
    except Exception:
        return False
    minx, miny, maxx, maxy = bounds
    return -180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0 and -90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0


def _write_crosswalk_gpkg(
    candidate_gdf: gpd.GeoDataFrame,
    review_gdf: gpd.GeoDataFrame,
    final_gdf: gpd.GeoDataFrame,
    out_gpkg: Path,
) -> None:
    if out_gpkg.exists():
        out_gpkg.unlink()
    cand = candidate_gdf if not candidate_gdf.empty else gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:32632")
    review = review_gdf if not review_gdf.empty else gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:32632")
    final = final_gdf if not final_gdf.empty else gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:32632")
    cand.to_file(out_gpkg, layer="crosswalk_candidate_poly", driver="GPKG")
    review.to_file(out_gpkg, layer="crosswalk_review_poly", driver="GPKG")
    final.to_file(out_gpkg, layer="crosswalk_poly", driver="GPKG")


def _update_qa_index_with_final(
    qa_index_path: Path,
    final_gdf: gpd.GeoDataFrame,
) -> Dict[Tuple[str, str], List[str]]:
    if not qa_index_path.exists():
        return {}
    qa = gpd.read_file(qa_index_path)
    support_map: Dict[Tuple[str, str], List[str]] = {}
    for _, row in final_gdf.iterrows():
        drive_id = str(row.get("drive_id") or "")
        frames_raw = row.get("support_frames", "")
        frames = []
        if isinstance(frames_raw, str) and frames_raw:
            try:
                frames = json.loads(frames_raw)
            except Exception:
                frames = []
        for frame_id in frames:
            support_map.setdefault((drive_id, str(frame_id)), []).append(row.get("entity_id"))
    supports = []
    final_ids = []
    for _, row in qa.iterrows():
        key = (str(row.get("drive_id") or ""), str(row.get("frame_id") or ""))
        ids = support_map.get(key, [])
        supports.append("yes" if ids else "no")
        final_ids.append(json.dumps(sorted(set(ids)), ensure_ascii=True))
    qa["entity_support_frames"] = supports
    qa["crosswalk_final_ids_nearby"] = final_ids
    qa["final_entity_ids_nearby"] = final_ids
    qa.to_file(qa_index_path, driver="GeoJSON")
    return support_map


def _ensure_gated_entities_images(
    qa_index_path: Path,
    outputs_dir: Path,
    candidate_gdf: gpd.GeoDataFrame,
    raw_frames: Dict[Tuple[str, str], dict],
    lidar_stats: Dict[str, dict],
    final_support: Dict[Tuple[str, str], List[str]],
    index_lookup: Dict[Tuple[str, str], str],
    final_gdf: gpd.GeoDataFrame,
    kitti_root: Path,
    lidar_world_mode: str,
    camera: str,
    stage2_overlay: Dict[str, List[dict]] | None = None,
) -> None:
    if not qa_index_path.exists():
        return
    qa = gpd.read_file(qa_index_path)
    pose_map: Dict[str, Tuple[float, float, float]] = {}
    try:
        calib = load_kitti360_calib(kitti_root, camera)
    except Exception:
        calib = None
    candidate_rejects: Dict[Tuple[str, str], List[str]] = {}
    candidate_ids: Dict[Tuple[str, str], List[str]] = {}
    candidate_by_frame: Dict[str, gpd.GeoDataFrame] = {}
    if not candidate_gdf.empty:
        candidate_gdf = candidate_gdf.copy()
        candidate_gdf["frame_id_norm"] = candidate_gdf["frame_id"].apply(_normalize_frame_id)
        for frame_id, group in candidate_gdf.groupby("frame_id_norm"):
            candidate_by_frame[str(frame_id)] = group
            for _, row in group.iterrows():
                drive_id = str(row.get("drive_id") or "")
                if not drive_id:
                    continue
                key = (drive_id, str(frame_id))
                candidate_ids.setdefault(key, []).append(str(row.get("candidate_id") or ""))
                reasons = str(row.get("reject_reasons") or "")
                if reasons:
                    for token in [r for r in reasons.split(",") if r]:
                        candidate_rejects.setdefault(key, []).append(token)
    for idx, row in qa.iterrows():
        drive_id = str(row.get("drive_id") or "")
        frame_id = _normalize_frame_id(str(row.get("frame_id") or ""))
        if not drive_id or not frame_id:
            continue
        image_path = index_lookup.get((drive_id, frame_id), "")
        qa_dir = outputs_dir / "qa_images" / drive_id
        qa_dir.mkdir(parents=True, exist_ok=True)
        gated_path = qa_dir / f"{frame_id}_overlay_gated.png"
        entities_path = qa_dir / f"{frame_id}_overlay_entities.png"
        raw_has = int(row.get("raw_has_crosswalk", 0) or 0)
        if frame_id not in pose_map:
            try:
                if lidar_world_mode == "fullpose":
                    x, y, z, roll, pitch, yaw = load_kitti360_pose_full(kitti_root, drive_id, frame_id)
                    pose_map[frame_id] = (x, y, z, roll, pitch, yaw)
                else:
                    x, y, yaw = load_kitti360_pose(kitti_root, drive_id, frame_id)
                    pose_map[frame_id] = (x, y, yaw)
            except Exception:
                pose_map[frame_id] = None
        candidates = candidate_by_frame.get(frame_id, gpd.GeoDataFrame())
        stage2_items = stage2_overlay.get(frame_id, []) if stage2_overlay else []
        kept, _ = _render_gated_overlay(
            gated_path,
            image_path,
            frame_id,
            candidates,
            raw_frames.get((drive_id, frame_id)),
            lidar_stats.get(frame_id, {}),
            pose_map.get(frame_id),
            calib,
            stage2_items,
        )
        qa.at[idx, "overlay_gated_path"] = str(gated_path)
        support_ids = final_support.get((drive_id, frame_id), [])
        if support_ids:
            _render_final_entities_image(
                entities_path,
                image_path,
                final_gdf,
                drive_id,
                frame_id,
                kitti_root,
                lidar_world_mode,
                camera,
            )
            qa.at[idx, "overlay_entities_path"] = str(entities_path)
        elif not entities_path.exists():
            _render_entities_overlay(
                entities_path,
                image_path,
                drive_id,
                frame_id,
                final_gdf,
                final_support,
                raw_has,
                pose_map.get(frame_id),
                calib,
            )
            qa.at[idx, "overlay_entities_path"] = str(entities_path)
        qa.at[idx, "candidate_ids_nearby"] = json.dumps(sorted(set(candidate_ids.get((drive_id, frame_id), []))), ensure_ascii=True)
    qa.to_file(qa_index_path, driver="GeoJSON")


def _render_final_entities_image(
    out_path: Path,
    image_path: str,
    final_gdf: gpd.GeoDataFrame,
    drive_id: str,
    frame_id: str,
    kitti_root: Path,
    lidar_world_mode: str,
    camera: str,
) -> None:
    if out_path.exists():
        out_path.unlink()
    if image_path and Path(image_path).exists():
        try:
            base = Image.open(image_path).convert("RGBA")
        except Exception:
            base = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    else:
        base = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    draw = ImageDraw.Draw(base, "RGBA")
    try:
        calib = load_kitti360_calib(kitti_root, camera)
    except Exception:
        _render_text_overlay(image_path, out_path, ["NO_CALIB"])
        return
    try:
        if lidar_world_mode == "fullpose":
            x, y, z, roll, pitch, yaw = load_kitti360_pose_full(kitti_root, drive_id, frame_id)
            pose = (x, y, z, roll, pitch, yaw)
        else:
            x, y, yaw = load_kitti360_pose(kitti_root, drive_id, frame_id)
            pose = (x, y, yaw)
    except Exception:
        _render_text_overlay(image_path, out_path, ["NO_POSE"])
        return
    if final_gdf.empty:
        _render_text_overlay(image_path, out_path, ["NO_FINAL_ENTITY"])
        return
    for _, row in final_gdf.iterrows():
        if str(row.get("drive_id") or "") != drive_id:
            continue
        frames_raw = row.get("support_frames", "")
        frames = []
        if isinstance(frames_raw, str) and frames_raw:
            try:
                frames = json.loads(frames_raw)
            except Exception:
                frames = []
        if frame_id not in {str(f) for f in frames}:
            continue
        geom = row.geometry
        pts = _geom_to_image_points(geom, pose, calib)
        if len(pts) < 2:
            continue
        draw.polygon(pts, outline=(0, 128, 255, 220), fill=(0, 128, 255, 80))
    base.save(out_path)


def _render_gated_overlay(
    out_path: Path,
    image_path: str,
    frame_id: str,
    candidates: gpd.GeoDataFrame,
    raw_info: dict | None,
    lidar_info: dict | None,
    pose: Tuple[float, float, float] | None,
    calib: Dict[str, np.ndarray] | None,
    stage2_items: List[dict] | None = None,
) -> Tuple[int, List[str]]:
    if out_path.exists():
        out_path.unlink()
    if image_path and Path(image_path).exists():
        try:
            base = Image.open(image_path).convert("RGBA")
        except Exception:
            base = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    else:
        base = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    draw = ImageDraw.Draw(base, "RGBA")
    if pose is None or calib is None:
        _render_text_overlay(image_path, out_path, ["NO_CALIB_OR_POSE"])
        return 0, []

    kept = 0
    reject_reasons: List[str] = []
    drawn = False
    stage2_added = False
    diag = lidar_info or {}
    for _, row in candidates.iterrows():
        geom = row.geometry
        reasons = str(row.get("reject_reasons") or "")
        is_rejected = bool(reasons)
        stage2_val = row.get("stage2_added", 0)
        if (pd.notna(stage2_val) and int(stage2_val) == 1) or str(row.get("qa_flag") or "") == "stage2_added":
            stage2_added = True
        bbox_px = row.get("bbox_px")
        if isinstance(bbox_px, str) and bbox_px.startswith("["):
            try:
                bbox_px = json.loads(bbox_px)
            except Exception:
                bbox_px = None
        if isinstance(bbox_px, (list, tuple)) and len(bbox_px) == 4:
            minx, miny, maxx, maxy = bbox_px
            draw.rectangle([minx, miny, maxx, maxy], outline=(160, 160, 160, 200), width=2)
            drawn = True
            reject_reasons.extend([r for r in reasons.split(",") if r])
        elif geom is not None and not geom.is_empty:
            pts = _geom_to_image_points(geom, pose, calib)
            if len(pts) >= 2:
                drawn = True
                color = (0, 255, 255, 200) if not is_rejected else (160, 160, 160, 200)
                draw.polygon(pts, outline=color)
                if is_rejected:
                    reject_reasons.extend([r for r in reasons.split(",") if r])
                else:
                    kept += 1

    bbox_px = raw_info.get("bbox_px") if raw_info else None
    if isinstance(bbox_px, (list, tuple)) and len(bbox_px) == 4 and not drawn:
        minx, miny, maxx, maxy = bbox_px
        draw.rectangle([minx, miny, maxx, maxy], outline=(160, 160, 160, 200), width=2)
        draw.text((10, 10), "PROJ_FAIL/GEOM_EMPTY", fill=(200, 200, 200, 220))
        reject_reasons.append("proj_fail")

    if diag:
        proj_method = str(diag.get("proj_method") or "none")
        points_in_bbox = int(diag.get("points_in_bbox", 0))
        points_in_mask = int(diag.get("points_in_mask", 0))
        points_in_mask_dilated = int(diag.get("points_in_mask_dilated", points_in_mask))
        points_support = int(diag.get("points_support", 0))
        points_support_accum = int(diag.get("points_support_accum", 0))
        accum_pre = int(diag.get("accum_pre", 0))
        accum_post = int(diag.get("accum_post", 0))
        drop_reason = str(diag.get("drop_reason_code") or "")
        draw.text(
            (10, 60),
            f"{proj_method} bbox={points_in_bbox} maskd={points_in_mask_dilated}",
            fill=(200, 200, 200, 220),
        )
        draw.text(
            (10, 86),
            f"support={points_support} accum={points_support_accum} pre/post={accum_pre}/{accum_post}",
            fill=(200, 200, 200, 220),
        )
        if drop_reason:
            draw.text((10, 112), drop_reason, fill=(200, 120, 120, 220))
        if proj_method in {"plane", "bbox_only"}:
            draw.text((10, 138), "WEAK_PROJ", fill=(255, 128, 0, 220))
    if stage2_added:
        draw.text((10, 164), "STAGE2_ADDED", fill=(0, 200, 255, 220))

    if stage2_items:
        for item in stage2_items:
            mask_path = item.get("mask_path")
            bbox_px = item.get("bbox_px")
            if isinstance(bbox_px, (list, tuple)) and len(bbox_px) == 4:
                draw.rectangle(bbox_px, outline=(0, 200, 255, 200), width=2)
            if mask_path and Path(mask_path).exists():
                mask = np.array(Image.open(mask_path).convert("L")) > 0
                poly = _mask_to_polygon(mask)
                if poly is not None:
                    pts = [(float(x), float(y)) for x, y in poly.exterior.coords]
                    if len(pts) >= 2:
                        draw.line(pts, fill=(0, 200, 255, 200), width=2)

    if len(candidates) == 0 and not drawn:
        draw.text((10, 10), "NO_GATED_CANDIDATE", fill=(255, 128, 0, 220))
    elif kept == 0:
        top_reject = sorted(set(reject_reasons))
        draw.text((10, 10), "NO_GATED_CANDIDATE", fill=(255, 128, 0, 220))
        if top_reject:
            draw.text((10, 36), "REJECT_REASONS:" + "|".join(top_reject[:5]), fill=(200, 200, 200, 220))
    base.save(out_path)
    return kept, sorted(set(reject_reasons))


def _render_entities_overlay(
    out_path: Path,
    image_path: str,
    drive_id: str,
    frame_id: str,
    final_gdf: gpd.GeoDataFrame,
    final_support: Dict[Tuple[str, str], List[str]],
    raw_has_crosswalk: int,
    pose: Tuple[float, float, float] | None,
    calib: Dict[str, np.ndarray] | None,
) -> None:
    if out_path.exists():
        out_path.unlink()
    if image_path and Path(image_path).exists():
        try:
            base = Image.open(image_path).convert("RGBA")
        except Exception:
            base = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    else:
        base = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
    draw = ImageDraw.Draw(base, "RGBA")
    if pose is None or calib is None:
        _render_text_overlay(image_path, out_path, ["NO_CALIB_OR_POSE"])
        return

    support_ids = final_support.get((drive_id, frame_id), [])
    if not support_ids:
        tag = "RAW_HAS_CROSSWALK=1" if raw_has_crosswalk == 1 else "RAW_HAS_CROSSWALK=0"
        draw.text((10, 10), "NO_FINAL_ENTITY", fill=(255, 128, 0, 220))
        draw.text((10, 36), tag, fill=(255, 200, 0, 220))
        base.save(out_path)
        return

    for _, row in final_gdf.iterrows():
        if str(row.get("drive_id") or "") != drive_id:
            continue
        frames_raw = row.get("support_frames", "")
        frames = []
        if isinstance(frames_raw, str) and frames_raw:
            try:
                frames = json.loads(frames_raw)
            except Exception:
                frames = []
        if frame_id not in {str(f) for f in frames}:
            continue
        geom = row.geometry
        pts = _geom_to_image_points(geom, pose, calib)
        if len(pts) < 2:
            continue
        draw.polygon(pts, outline=(0, 128, 255, 220), fill=(0, 128, 255, 80))
    base.save(out_path)


def _append_reject_reason(current: str, reason: str) -> str:
    tokens = [r for r in str(current or "").split(",") if r]
    if reason not in tokens:
        tokens.append(reason)
    return ",".join(tokens)


def _is_gore_like(rect_w: float, rect_l: float, rectangularity: float) -> bool:
    if rect_l > 10.0 and rect_w > 0.0 and rect_w < 2.0:
        return True
    if rectangularity > 0 and rectangularity < 0.2 and rect_l > rect_w * 3.0:
        return True
    if rectangularity > 0 and rectangularity < 0.7 and rect_l > rect_w * 2.0:
        return True
    return False


def _cluster_by_centroid(geoms: List[Polygon], eps_m: float) -> List[List[int]]:
    if not geoms:
        return []
    centers = np.array([[geom.centroid.x, geom.centroid.y] for geom in geoms], dtype=float)
    used = np.zeros(len(centers), dtype=bool)
    clusters: List[List[int]] = []
    for idx in range(len(centers)):
        if used[idx]:
            continue
        queue = [idx]
        used[idx] = True
        members = []
        while queue:
            cur = queue.pop()
            members.append(cur)
            dist = np.linalg.norm(centers - centers[cur], axis=1)
            neighbors = np.where((dist <= eps_m) & (~used))[0]
            for nidx in neighbors:
                used[nidx] = True
                queue.append(int(nidx))
        clusters.append(members)
    return clusters


def _build_clusters(candidate_gdf: gpd.GeoDataFrame, cluster_eps_m: float) -> Tuple[gpd.GeoDataFrame, Dict[str, dict]]:
    if candidate_gdf.empty:
        return candidate_gdf, {}
    gdf = candidate_gdf.copy()
    gdf["frame_id_norm"] = gdf["frame_id"].apply(_normalize_frame_id)
    if "weak_support" not in gdf.columns:
        gdf["weak_support"] = 0
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]
    gdf["cluster_id"] = ""
    valid = gdf
    if "geom_ok" in gdf.columns:
        valid = valid[valid["geom_ok"].fillna(0).astype(int) == 1]
    geoms = valid.geometry.tolist()
    clusters = _cluster_by_centroid(geoms, cluster_eps_m)
    cluster_info: Dict[str, dict] = {}
    valid_indices = valid.index.tolist()
    cluster_ids = ["" for _ in range(len(valid_indices))]
    for idx, members in enumerate(clusters):
        cid = f"cluster_{idx:04d}"
        frames = set()
        centroids = []
        rect_ws = []
        rect_ls = []
        rectangularities = []
        headings = []
        inside_ratios = []
        member_indices = [valid_indices[m] for m in members]
        for member in members:
            row = gdf.loc[valid_indices[member]]
            frames.add(str(row.get("frame_id_norm") or ""))
            geom = row.geometry
            centroids.append((geom.centroid.x, geom.centroid.y))
            metrics = _rect_metrics(geom)
            rect_w_val = row.get("rect_w_m")
            rect_l_val = row.get("rect_l_m")
            rectr_val = row.get("rectangularity")
            try:
                rect_w = float(rect_w_val) if rect_w_val is not None and pd.notna(rect_w_val) else metrics["rect_w_m"]
            except Exception:
                rect_w = metrics["rect_w_m"]
            try:
                rect_l = float(rect_l_val) if rect_l_val is not None and pd.notna(rect_l_val) else metrics["rect_l_m"]
            except Exception:
                rect_l = metrics["rect_l_m"]
            try:
                rectr = float(rectr_val) if rectr_val is not None and pd.notna(rectr_val) else metrics["rectangularity"]
            except Exception:
                rectr = metrics["rectangularity"]
            rect_ws.append(rect_w)
            rect_ls.append(rect_l)
            rectangularities.append(rectr)
            heading = row.get("heading_diff_to_perp_deg")
            if heading != "" and heading is not None and pd.notna(heading):
                try:
                    headings.append(float(heading))
                except Exception:
                    pass
            inside = row.get("inside_road_ratio")
            if inside is not None and pd.notna(inside):
                try:
                    inside_ratios.append(float(inside))
                except Exception:
                    pass
        if centroids:
            xs = [c[0] for c in centroids]
            ys = [c[1] for c in centroids]
            med_x = float(np.median(xs))
            med_y = float(np.median(ys))
            dists = [np.hypot(x - med_x, y - med_y) for x, y in centroids]
            jitter_p90 = float(np.percentile(dists, 90)) if dists else 0.0
        else:
            jitter_p90 = 0.0
        cluster_info[cid] = {
            "frames_hit": len(frames),
            "frames_hit_all": len(frames),
            "frames_hit_support": 0,
            "gore_like_ratio": 0.0,
            "jitter_p90": jitter_p90,
            "rect_w_m": float(np.median(rect_ws)) if rect_ws else 0.0,
            "rect_l_m": float(np.median(rect_ls)) if rect_ls else 0.0,
            "rectangularity": float(np.median(rectangularities)) if rectangularities else 0.0,
            "heading_diff": float(np.median(headings)) if headings else None,
            "inside_road_ratio": float(np.median(inside_ratios)) if inside_ratios else None,
            "member_indices": member_indices,
        }
        union_geom = gdf.loc[member_indices].geometry.union_all()
        rect = union_geom.minimum_rotated_rectangle if union_geom is not None else None
        if rect is not None and not rect.is_empty and rect.is_valid and rect.geom_type == "Polygon":
            metrics = _rect_metrics(union_geom)
            cluster_info[cid]["refined_geom"] = rect
            rect_w = float(metrics["rect_w_m"])
            rect_l = float(metrics["rect_l_m"])
            rectr = float(metrics["rectangularity"])
            if rect_ws and rect_w < 0.5:
                rect_w = float(np.median(rect_ws))
            if rect_ls and rect_l < 1.0:
                rect_l = float(np.median(rect_ls))
            if rectangularities and rectr <= 0.0:
                rectr = float(np.median(rectangularities))
            cluster_info[cid]["rect_w_m"] = rect_w
            cluster_info[cid]["rect_l_m"] = rect_l
            cluster_info[cid]["rectangularity"] = rectr
        for member in members:
            cluster_ids[member] = cid
    for idx, orig_idx in enumerate(valid_indices):
        gdf.at[orig_idx, "cluster_id"] = cluster_ids[idx]
    return gdf, cluster_info


def _refine_clusters(
    candidate_gdf: gpd.GeoDataFrame,
    cluster_info: Dict[str, dict],
    pose_map: Dict[str, Tuple[float, float, float] | None],
    lidar_stats: Dict[str, dict],
    raw_stats: Dict[Tuple[str, str], Dict[str, float]],
    drive_id: str,
    refine_cfg: dict,
) -> Tuple[gpd.GeoDataFrame, Dict[str, dict]]:
    if candidate_gdf.empty or not cluster_info:
        return candidate_gdf, cluster_info
    topk = int(refine_cfg.get("REFINE_TOPK", 5))
    outlier_dist = float(refine_cfg.get("OUTLIER_DIST_M", 6.0))
    angle_snap = float(refine_cfg.get("ANGLE_SNAP_DEG", 20.0))
    gore_mode = str(refine_cfg.get("GORE_LIKE_MODE", "soft")).lower()
    support_w_pts = float(refine_cfg.get("SUPPORT_W_PTS", 1.0))
    support_w_score = float(refine_cfg.get("SUPPORT_W_SCORE", 100.0))
    penalty_gore = float(refine_cfg.get("PENALTY_GORE", 0.2))
    penalty_plane = float(refine_cfg.get("PENALTY_PLANE", 0.4))
    penalty_outlier = float(refine_cfg.get("PENALTY_OUTLIER", 0.3))
    max_gore_support = int(refine_cfg.get("MAX_GORE_SUPPORT", 1))
    gdf = candidate_gdf.copy()
    gdf["frame_id_norm"] = gdf["frame_id"].apply(_normalize_frame_id)
    for cid, info in cluster_info.items():
        members = info.get("member_indices", [])
        if not members:
            continue
        full_subset = gdf.loc[members].copy()
        rect_w_override = None
        rect_l_override = None
        rectr_override = None
        if "rect_w_m" in full_subset.columns:
            vals = [float(v) for v in full_subset["rect_w_m"].tolist() if v is not None and pd.notna(v) and float(v) > 0]
            if vals:
                rect_w_override = float(np.median(vals))
        if "rect_l_m" in full_subset.columns:
            vals = [float(v) for v in full_subset["rect_l_m"].tolist() if v is not None and pd.notna(v) and float(v) > 0]
            if vals:
                rect_l_override = float(np.median(vals))
        if "rectangularity" in full_subset.columns:
            vals = [float(v) for v in full_subset["rectangularity"].tolist() if v is not None and pd.notna(v) and float(v) > 0]
            if vals:
                rectr_override = float(np.median(vals))
        frames_all = set(full_subset["frame_id_norm"].tolist())
        info["frames_hit_all"] = len(frames_all)
        info["frames_hit"] = len(frames_all)
        if "gore_like" in full_subset.columns:
            gore_like_count = int(full_subset["gore_like"].fillna(False).astype(bool).sum())
            info["gore_like_ratio"] = float(gore_like_count / max(1, len(full_subset)))
        subset = full_subset
        if "geom_ok" in subset.columns:
            geom_ok_subset = subset[subset["geom_ok"].fillna(0).astype(int) == 1]
            if not geom_ok_subset.empty:
                subset = geom_ok_subset
        if "proj_method" in subset.columns:
            subset = subset[subset["proj_method"].fillna("").astype(str).isin({"lidar", "plane"})]
        if subset.empty:
            continue
        subset_idx = subset.index.tolist()
        gore_mask = subset.get("gore_like", False).fillna(False).astype(bool) if "gore_like" in subset.columns else pd.Series(False, index=subset.index)
        lidar_subset = subset[subset.get("proj_method", "").fillna("").astype(str) == "lidar"] if "proj_method" in subset.columns else subset
        outlier_frames: set[str] = set()
        if not lidar_subset.empty:
            lidar_centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in lidar_subset.geometry], dtype=float)
            med = np.median(lidar_centroids, axis=0)
            dists = np.linalg.norm(lidar_centroids - med[None, :], axis=1)
            lidar_outliers = dists > outlier_dist
            outlier_ratio = float(np.sum(lidar_outliers)) / max(1, lidar_outliers.size)
            lidar_frames = lidar_subset["frame_id_norm"].tolist()
            for idx, is_outlier in enumerate(lidar_outliers):
                if not is_outlier:
                    continue
                frame_id = lidar_frames[idx]
                mask = (gdf["frame_id_norm"] == frame_id) & (gdf.get("proj_method", "") == "lidar")
                gdf.loc[mask, "reject_reasons"] = gdf.loc[mask, "reject_reasons"].apply(
                    lambda v: _append_reject_reason(v, "outlier")
                )
            outlier_frames = set([lidar_frames[i] for i, val in enumerate(lidar_outliers) if val])
            if outlier_ratio > 0.8:
                drop_frames = set([lidar_frames[i] for i, val in enumerate(lidar_outliers) if val])
                subset = subset[~subset["frame_id_norm"].isin(drop_frames)]
        if subset.empty:
            continue
        subset_idx = subset.index.tolist()
        centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in subset.geometry], dtype=float)
        med = np.median(centroids, axis=0)
        dists = np.linalg.norm(centroids - med[None, :], axis=1)
        inlier_indices = [subset_idx[i] for i in range(len(subset_idx))]
        base_heading = info.get("heading_diff")
        # frames_hit_all is based on full cluster membership; support frames below.
        if dists.size > 0:
            info["jitter_p90"] = float(np.percentile(dists, 90))
        else:
            info["jitter_p90"] = 0.0
        if not inlier_indices:
            continue
        inlier_subset = gdf.iloc[inlier_indices]
        info["support_frames"] = []
        info["frames_hit_support"] = 0
        angle_diffs = []
        for _, row in inlier_subset.iterrows():
            frame_id = str(row.get("frame_id_norm") or "")
            pose = pose_map.get(frame_id)
            if pose is None:
                continue
            angle_diffs.append(_angle_diff_to_heading_deg(row.geometry, pose[2]))
        if angle_diffs:
            info["angle_jitter_p90"] = float(np.percentile(angle_diffs, 90))
        else:
            info["angle_jitter_p90"] = 0.0
        scored = []
        for _, row in inlier_subset.iterrows():
            frame_id = str(row.get("frame_id_norm") or "")
            stats = lidar_stats.get(frame_id, {})
            pts = int(stats.get("points_in_bbox", 0))
            raw = raw_stats.get((drive_id, frame_id), {})
            score = float(raw.get("raw_top_score", 0.0))
            gore_like = bool(row.get("gore_like")) if "gore_like" in row else False
            proj_method = str(row.get("proj_method") or "")
            base_score = support_w_pts * pts + support_w_score * score
            if gore_like:
                base_score *= penalty_gore
            if proj_method == "plane":
                base_score *= penalty_plane
            if frame_id in outlier_frames:
                base_score *= penalty_outlier
            scored.append((base_score, gore_like, frame_id))
        scored.sort(key=lambda v: v[0], reverse=True)
        non_gore = [f for _s, gore, f in scored if not gore]
        keep_frames = non_gore[:topk]
        if len(keep_frames) < topk and gore_mode != "hard":
            gore_frames = [f for _s, gore, f in scored if gore][:max_gore_support]
            keep_frames.extend(gore_frames[: max(0, topk - len(keep_frames))])
        info["support_frames"] = keep_frames
        info["frames_hit_support"] = len(keep_frames)
        for frame_id in keep_frames:
            mask = gdf["frame_id_norm"] == frame_id
            row = gdf[mask].head(1)
            if row.empty:
                continue
            proj_method = str(row["proj_method"].iloc[0]) if "proj_method" in row.columns else ""
            gore_like = bool(row["gore_like"].iloc[0]) if "gore_like" in row.columns else False
            if proj_method == "plane" or gore_like or frame_id in outlier_frames:
                gdf.loc[mask, "weak_support"] = 1
        accum_pts = []
        headings = []
        for frame_id in keep_frames:
            stats = lidar_stats.get(frame_id, {})
            pts = stats.get("support_points")
            if isinstance(pts, np.ndarray) and pts.size > 0:
                accum_pts.append(pts)
            pose = pose_map.get(frame_id)
            if pose is not None:
                headings.append(float(pose[2]))
        if not accum_pts:
            union_geom = inlier_subset.geometry.union_all()
            rect = union_geom.minimum_rotated_rectangle if union_geom is not None else None
            if rect is None or rect.is_empty:
                continue
            if not rect.is_valid:
                rect = rect.buffer(0)
            if rect is None or rect.is_empty or not rect.is_valid or rect.geom_type != "Polygon":
                continue
            heading = float(np.median(headings)) if headings else 0.0
            rect = _align_rect_to_heading(rect, heading, angle_snap)
            metrics = _rect_metrics(union_geom)
            info["refined_geom"] = rect
            rect_w = float(metrics["rect_w_m"])
            rect_l = float(metrics["rect_l_m"])
            rectr = float(metrics["rectangularity"])
            if rect_w_override is not None and rect_w < 0.5:
                rect_w = rect_w_override
            if rect_l_override is not None and rect_l < 1.0:
                rect_l = rect_l_override
            if rectr_override is not None and rectr <= 0.0:
                rectr = rectr_override
            info["rect_w_m"] = rect_w
            info["rect_l_m"] = rect_l
            info["rectangularity"] = rectr
            refined_heading = float(np.degrees(abs(((_rect_angle_rad(rect) - (heading + np.pi / 2)) + np.pi) % (2 * np.pi) - np.pi)))
            if base_heading is not None:
                refined_heading = float(min(base_heading, refined_heading))
            info["heading_diff"] = refined_heading
            continue
        merged = np.vstack(accum_pts)
        hull = MultiPoint([Point(float(x), float(y)) for x, y in merged]).convex_hull
        rect = hull.minimum_rotated_rectangle
        if rect is None or rect.is_empty:
            continue
        if not rect.is_valid:
            rect = rect.buffer(0)
        if rect is None or rect.is_empty or not rect.is_valid:
            continue
        heading = float(np.median(headings)) if headings else 0.0
        rect = _align_rect_to_heading(rect, heading, angle_snap)
        metrics = _rect_metrics(rect)
        info["refined_geom"] = rect
        rect_w = float(metrics["rect_w_m"])
        rect_l = float(metrics["rect_l_m"])
        rectr = float(metrics["rectangularity"])
        if rect_w_override is not None and rect_w < 0.5:
            rect_w = rect_w_override
        if rect_l_override is not None and rect_l < 1.0:
            rect_l = rect_l_override
        if rectr_override is not None and rectr <= 0.0:
            rectr = rectr_override
        info["rect_w_m"] = rect_w
        info["rect_l_m"] = rect_l
        info["rectangularity"] = rectr
        refined_heading = float(np.degrees(abs(((_rect_angle_rad(rect) - (heading + np.pi / 2)) + np.pi) % (2 * np.pi) - np.pi)))
        if base_heading is not None:
            refined_heading = float(min(base_heading, refined_heading))
        info["heading_diff"] = refined_heading
    return gdf, cluster_info


def _refresh_cluster_members(
    candidate_gdf: gpd.GeoDataFrame,
    cluster_info: Dict[str, dict],
) -> Dict[str, dict]:
    if candidate_gdf.empty:
        return cluster_info
    if "cluster_id" not in candidate_gdf.columns:
        return cluster_info
    for cid in cluster_info.keys():
        members = candidate_gdf.index[candidate_gdf["cluster_id"] == cid].tolist()
        cluster_info[cid]["member_indices"] = members
    return cluster_info


def _stage2_roi_refine(
    candidate_gdf: gpd.GeoDataFrame,
    cluster_info: Dict[str, dict],
    drive_id: str,
    frame_start: int,
    frame_end: int,
    index_lookup: Dict[Tuple[str, str], str],
    pose_map: Dict[str, Tuple[float, float, float] | None],
    calib: Dict[str, np.ndarray] | None,
    cam_to_pose: Optional[np.ndarray],
    lidar_cfg: dict,
    image_provider: str,
    kitti_root: Path,
    stage2_cfg: dict,
) -> Tuple[gpd.GeoDataFrame, Dict[Tuple[str, str], dict]]:
    if not cluster_info:
        return candidate_gdf, {}
    predictor = _get_sam2_predictor(image_provider)
    roi_min_area = float(stage2_cfg.get("ROI_MIN_AREA_PX", 400.0))
    roi_max_ratio = float(stage2_cfg.get("ROI_MAX_AREA_RATIO", 0.6))
    stage2_stats: Dict[Tuple[str, str], dict] = {}
    stage2_rows: List[dict] = []
    existing = set()
    if not candidate_gdf.empty and "cluster_id" in candidate_gdf.columns:
        for _, row in candidate_gdf.iterrows():
            existing.add((str(row.get("cluster_id") or ""), str(row.get("frame_id") or "")))
    for frame in range(frame_start, frame_end + 1):
        frame_id = _normalize_frame_id(str(frame))
        image_path = index_lookup.get((drive_id, frame_id), "")
        if not image_path:
            continue
        pose = pose_map.get(frame_id)
        for cid, info in cluster_info.items():
            roi_geom = info.get("refined_geom")
            if roi_geom is None or roi_geom.is_empty:
                members = info.get("member_indices", [])
                if members and not candidate_gdf.empty:
                    subset = candidate_gdf.iloc[members]
                    roi_geom = subset.geometry.union_all()
            if roi_geom is None or roi_geom.is_empty:
                continue
            roi_bbox = _roi_bbox_from_geom(roi_geom, pose, calib, image_path, roi_min_area, roi_max_ratio)
            key = (drive_id, frame_id)
            stat = stage2_stats.setdefault(key, {"attempted": 0, "added": 0, "reasons": []})
            if roi_bbox is None:
                stat["attempted"] = 1
                stat["reasons"].append("roi_invalid")
                continue
            if (cid, frame_id) in existing:
                continue
            stat["attempted"] = 1
            mask = _sam2_box_mask(predictor, image_path, roi_bbox)
            if mask is None:
                stat["reasons"].append("sam2_fail")
                continue
            poly = _mask_to_polygon(mask)
            if poly is None:
                stat["reasons"].append("mask_empty")
                continue
            raw_info = {"gdf": gpd.GeoDataFrame(geometry=[poly]), "bbox_px": roi_bbox}
            cand, stats = _build_lidar_candidate_for_frame(
                kitti_root,
                drive_id,
                frame_id,
                image_path,
                raw_info,
                pose,
                calib,
                cam_to_pose,
                lidar_cfg,
                lidar_world_mode,
                camera,
            )
            if cand is None:
                reason = str(stats.get("drop_reason_code") or "geom_invalid")
                stat["reasons"].append(reason)
                continue
            cand["properties"]["candidate_id"] = f"{drive_id}_crosswalk_stage2_{cid}_{frame_id}"
            cand["properties"]["cluster_id"] = cid
            cand["properties"]["stage2_added"] = 1
            cand["properties"]["source_frame_id"] = frame_id
            cand["properties"]["source"] = "stage2_roi_refine"
            cand["properties"]["roi_bbox_px"] = roi_bbox
            cand["properties"]["weak_support"] = 1 if str(stats.get("proj_method")) == "plane" else 0
            cand["properties"]["qa_flag"] = "stage2_added"
            stage2_rows.append(cand)
            existing.add((cid, frame_id))
            stat["added"] = 1
            stat["reasons"].append("pass")
            info["stage2_added_frames_count"] = int(info.get("stage2_added_frames_count", 0)) + 1
    if not stage2_rows:
        return candidate_gdf, stage2_stats
    stage2_gdf = gpd.GeoDataFrame.from_features(stage2_rows, crs="EPSG:32632")
    if candidate_gdf.empty:
        merged = stage2_gdf
    else:
        merged = gpd.GeoDataFrame(
            pd.concat([candidate_gdf, stage2_gdf], ignore_index=True),
            geometry="geometry",
            crs="EPSG:32632",
        )
    return merged, stage2_stats


def _build_review_final_layers(
    drive_id: str,
    candidate_gdf: gpd.GeoDataFrame,
    cluster_info: Dict[str, dict],
    final_cfg: dict,
    review_cfg: dict,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if not cluster_info:
        empty = gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:32632")
        return empty.copy(), empty.copy()
    review_rows = []
    final_rows = []
    min_frames_final = int(final_cfg.get("min_frames_hit_final", 3))
    min_inside = float(final_cfg.get("min_inside_ratio_final", 0.5))
    rect_min_w = float(final_cfg.get("rect_min_w", 1.5))
    rect_max_w = float(final_cfg.get("rect_max_w", 8.0))
    rect_min_l = float(final_cfg.get("rect_min_l", 3.0))
    rect_max_l = float(final_cfg.get("rect_max_l", 25.0))
    rect_min = float(final_cfg.get("rectangularity_min", 0.45))
    heading_max = float(final_cfg.get("heading_diff_to_perp_max_deg", 25.0))
    jitter_max = float(final_cfg.get("jitter_p90_max", 8.0))
    min_frames_review = int(review_cfg.get("min_frames_hit_review", 2))
    for cid, info in cluster_info.items():
        geom = info.get("refined_geom")
        if (geom is None or geom.is_empty) and "member_indices" in info and not candidate_gdf.empty:
            subset = candidate_gdf.iloc[info["member_indices"]]
            union_geom = subset.geometry.union_all()
            geom = union_geom.minimum_rotated_rectangle if union_geom is not None else None
        if geom is None or geom.is_empty:
            continue
        frames_hit_all = int(info.get("frames_hit_all", info.get("frames_hit", 0)))
        frames_hit_support = int(info.get("frames_hit_support", 0))
        rect_w = float(info.get("rect_w_m", 0.0))
        rect_l = float(info.get("rect_l_m", 0.0))
        rectangularity = float(info.get("rectangularity", 0.0))
        heading_diff = float(info.get("heading_diff", 0.0)) if info.get("heading_diff") is not None else 0.0
        jitter_p90 = float(info.get("jitter_p90", 0.0))
        inside_ratio = info.get("inside_road_ratio")
        inside_ratio = float(inside_ratio) if inside_ratio is not None else 1.0
        if frames_hit_all >= min_frames_review:
            review_rows.append(
                {
                    "geometry": geom,
                    "properties": {
                        "entity_id": f"{drive_id}_crosswalk_review_{cid}",
                        "drive_id": drive_id,
                        "cluster_id": cid,
                        "frames_hit": frames_hit_all,
                        "frames_hit_support": frames_hit_support,
                        "jitter_p90": jitter_p90,
                        "rect_w_m": rect_w,
                        "rect_l_m": rect_l,
                        "rectangularity": rectangularity,
                    },
                }
            )
        if (
            frames_hit_support >= min_frames_final
            and rect_min_w <= rect_w <= rect_max_w
            and rect_min_l <= rect_l <= rect_max_l
            and rectangularity >= rect_min
            and heading_diff <= heading_max
            and jitter_p90 <= jitter_max
            and inside_ratio >= min_inside
        ):
            final_rows.append(
                {
                    "geometry": geom,
                    "properties": {
                        "entity_id": f"{drive_id}_crosswalk_final_{cid}",
                        "drive_id": drive_id,
                        "cluster_id": cid,
                        "frames_hit": frames_hit_all,
                        "frames_hit_support": frames_hit_support,
                        "jitter_p90": jitter_p90,
                        "rect_w_m": rect_w,
                        "rect_l_m": rect_l,
                        "rectangularity": rectangularity,
                        "support_frames": json.dumps(sorted(info.get("support_frames", [])), ensure_ascii=True),
                    },
                }
            )
    review_gdf = gpd.GeoDataFrame.from_features(review_rows, crs="EPSG:32632") if review_rows else gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:32632")
    final_gdf = gpd.GeoDataFrame.from_features(final_rows, crs="EPSG:32632") if final_rows else gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:32632")
    return review_gdf, final_gdf
def _build_trace(
    out_path: Path,
    records: List[dict],
) -> None:
    if out_path.exists():
        out_path.unlink()
    import pandas as pd

    pd.DataFrame(records).to_csv(out_path, index=False)


def _fallback_candidate_from_pose(
    drive_id: str,
    frame_id: str,
    pose: Tuple[float, ...] | None,
    use_plane: bool,
    bbox_px: List[float] | None,
    extra_reject: str,
    size_m: float = 2.0,
) -> dict | None:
    if pose is None:
        x, y, yaw = 500000.0, 0.0, 0.0
        reject_extra = "pose_missing,"
        proj_method = "bbox_only"
        plane_ok = 0
    else:
        if len(pose) == 6:
            x, y, _z, _roll, _pitch, yaw = pose
        else:
            x, y, yaw = pose
        reject_extra = ""
        proj_method = "plane" if use_plane else "bbox_only"
        plane_ok = 1 if use_plane else 0
    half = max(0.5, size_m / 2.0)
    forward = max(1.0, size_m)
    x = x + float(np.cos(yaw)) * forward
    y = y + float(np.sin(yaw)) * forward
    geom = box(x - half, y - half, x + half, y + half)
    if proj_method == "plane":
        geom = affinity.rotate(geom, np.degrees(yaw), origin=(x, y))
    return {
        "geometry": geom,
        "properties": {
            "candidate_id": f"{drive_id}_crosswalk_fallback_{frame_id}",
            "drive_id": drive_id,
            "frame_id": frame_id,
            "source_frame_id": frame_id,
            "entity_type": "crosswalk",
            "reject_reasons": f"{reject_extra}proj_fail,geom_fallback"
            + (",proj_fallback_plane" if proj_method == "plane" else ",proj_fail_bbox_only")
            + (f",{extra_reject}" if extra_reject else ""),
            "proj_method": proj_method,
            "plane_ok": plane_ok,
            "geom_ok": 0 if proj_method == "bbox_only" else 1,
            "geom_area_m2": float(geom.area) if proj_method != "bbox_only" else 0.0,
            "bbox_px": bbox_px,
            "qa_flag": "proj_fail",
        },
    }


def _augment_candidates_with_fallback(
    candidate_gdf: gpd.GeoDataFrame,
    raw_stats: Dict[Tuple[str, str], Dict[str, float]],
    pose_map: Dict[str, Tuple[float, float, float] | None],
    calib_ok: bool,
    raw_frames: Dict[Tuple[str, str], dict],
    lidar_stats: Dict[str, dict],
    drive_id: str,
    frame_start: int,
    frame_end: int,
    plane_score_min: float,
) -> Tuple[gpd.GeoDataFrame, set[str]]:
    fallback = []
    fallback_frames: set[str] = set()
    existing = set()
    if not candidate_gdf.empty:
        candidate_gdf = candidate_gdf.copy()
        candidate_gdf["frame_id_norm"] = candidate_gdf["frame_id"].apply(_normalize_frame_id)
        existing = set(candidate_gdf["frame_id_norm"].tolist())
    for frame in range(frame_start, frame_end + 1):
        frame_id = _normalize_frame_id(str(frame))
        key = (drive_id, frame_id)
        raw_info = raw_stats.get(key, {})
        if int(raw_info.get("raw_has_crosswalk", 0)) != 1:
            continue
        if raw_info.get("raw_status") not in {"ok", "on_demand_infer_ok"}:
            continue
        if frame_id in existing:
            continue
        raw_info_frame = raw_frames.get((drive_id, frame_id), {})
        lidar_info = lidar_stats.get(frame_id, {})
        extra_reject = str(lidar_info.get("drop_reason_code") or "")
        score = float(raw_info.get("raw_top_score", 0.0) or 0.0)
        pose_ok = pose_map.get(frame_id) is not None
        use_plane = bool(calib_ok and pose_ok and score >= plane_score_min)
        fallback_row = _fallback_candidate_from_pose(
            drive_id,
            frame_id,
            pose_map.get(frame_id),
            use_plane,
            raw_info_frame.get("bbox_px"),
            extra_reject,
        )
        if fallback_row is None:
            continue
        fallback.append(fallback_row)
        fallback_frames.add(frame_id)
    if not fallback:
        return candidate_gdf, fallback_frames
    fallback_gdf = gpd.GeoDataFrame.from_features(fallback, crs="EPSG:32632")
    if candidate_gdf.empty:
        merged = fallback_gdf
    else:
        merged = gpd.GeoDataFrame(
            pd.concat([candidate_gdf.drop(columns=["frame_id_norm"]), fallback_gdf], ignore_index=True),
            geometry="geometry",
            crs="EPSG:32632",
        )
    return merged, fallback_frames


def _build_trace_records(
    drive_id: str,
    frame_start: int,
    frame_end: int,
    index_lookup: Dict[Tuple[str, str], str],
    raw_stats: Dict[Tuple[str, str], Dict[str, float]],
    candidate_gdf: gpd.GeoDataFrame,
    final_support: Dict[Tuple[str, str], List[str]],
    cluster_info: Dict[str, dict],
    lidar_stats: Dict[str, dict],
    pose_map: Dict[str, Tuple[float, float, float] | None],
    calib_ok: bool,
    stage2_stats: Dict[Tuple[str, str], dict],
    roundtrip_by_frame: Dict[str, dict],
    lidar_world_mode: str,
    pose_source_label: str,
) -> List[dict]:
    candidate_ids: Dict[Tuple[str, str], List[str]] = {}
    candidate_rejects: Dict[Tuple[str, str], List[str]] = {}
    candidate_by_frame: Dict[str, gpd.GeoDataFrame] = {}
    if not candidate_gdf.empty:
        candidate_gdf = candidate_gdf.copy()
        candidate_gdf["frame_id_norm"] = candidate_gdf["frame_id"].apply(_normalize_frame_id)
        for frame_id, group in candidate_gdf.groupby("frame_id_norm"):
            candidate_by_frame[str(frame_id)] = group
            for _, row in group.iterrows():
                d = str(row.get("drive_id") or "")
                if not d:
                    continue
                key = (d, str(frame_id))
                candidate_ids.setdefault(key, []).append(str(row.get("candidate_id") or ""))
                reasons = str(row.get("reject_reasons") or "")
                if reasons:
                    for token in [r for r in reasons.split(",") if r]:
                        candidate_rejects.setdefault(key, []).append(token)
    records: List[dict] = []
    for frame in range(frame_start, frame_end + 1):
        frame_id = _normalize_frame_id(str(frame))
        key = (drive_id, frame_id)
        raw_info = raw_stats.get(
            key,
            {"raw_has_crosswalk": 0.0, "raw_top_score": 0.0, "raw_status": "unknown"},
        )
        lidar_info = lidar_stats.get(frame_id, {})
        pose_ok = 1 if pose_map.get(frame_id) is not None else 0
        pose_source = pose_source_label if pose_ok else "missing"
        cand_ids = sorted(set(candidate_ids.get(key, [])))
        rejects = sorted(set(candidate_rejects.get(key, [])))
        final_ids = sorted(set(final_support.get(key, [])))
        stage2 = stage2_stats.get(
            key,
            {"cluster_id": "", "stage2_added": 0, "prop_ok": 0, "prop_area_px": 0.0, "prop_drift_flag": 0},
        )
        candidates = candidate_by_frame.get(frame_id, gpd.GeoDataFrame())
        candidate_count = int(len(candidates)) if not candidates.empty else 0
        kept = 0
        geom_ok = 0
        geom_area = 0.0
        proj_method = str(lidar_info.get("proj_method") or "none")
        source_frame_id = ""
        x_ego = None
        if not candidates.empty:
            primary = _select_primary_candidate(candidates)
            if primary is not None:
                source_frame_id = _clean_source_frame_id(primary.get("source_frame_id"), frame_id)
                source_frame_id = _clean_source_frame_id(primary.get("frame_id"), source_frame_id)
                x_ego = _x_ego_from_geom(pose_map.get(frame_id), primary.geometry)
            for _, row in candidates.iterrows():
                reasons = str(row.get("reject_reasons") or "")
                if not reasons:
                    kept += 1
                geom = row.geometry
                if geom is not None and not geom.is_empty:
                    geom_area = max(geom_area, float(geom.area))
                if "geom_ok" in row and pd.notna(row.get("geom_ok")):
                    try:
                        geom_ok = max(geom_ok, int(row.get("geom_ok")))
                    except Exception:
                        pass
                else:
                    geom_ok = 1 if geom_area > 0 else geom_ok
                if proj_method == "none" and "proj_method" in row and pd.notna(row.get("proj_method")):
                    val = str(row.get("proj_method"))
                    if val:
                        proj_method = val
                if "proj_method" in row and pd.notna(row.get("proj_method")):
                    val = str(row.get("proj_method"))
                    if val:
                        proj_method = val
        proj_points = int(lidar_info.get("points_in_mask", -1))
        drop_reason = ""
        if int(raw_info.get("raw_has_crosswalk", 0)) == 0:
            drop_reason = "RAW_EMPTY"
        elif lidar_info.get("drop_reason_code"):
            drop_reason = str(lidar_info.get("drop_reason_code"))
        elif not cand_ids:
            drop_reason = "WRITE_CANDIDATE_FAIL"
        cluster_id = ""
        cluster_frames_hit = ""
        jitter_p90 = ""
        angle_jitter_p90 = ""
        support_flag = 0
        rect_w_m = float(lidar_info.get("rect_w_m", 0.0))
        rect_l_m = float(lidar_info.get("rect_l_m", 0.0))
        rectangularity = float(lidar_info.get("rectangularity", 0.0))
        if not candidates.empty and "cluster_id" in candidates.columns:
            cluster_ids = [str(v) for v in candidates["cluster_id"].tolist() if v]
            if cluster_ids:
                cluster_id = sorted(set(cluster_ids))[0]
                info = cluster_info.get(cluster_id, {})
                cluster_frames_hit = str(info.get("frames_hit_support", info.get("frames_hit", "")))
                jitter_p90 = str(info.get("jitter_p90", ""))
                angle_jitter_p90 = str(info.get("angle_jitter_p90", ""))
                support_frames = set(info.get("support_frames", []))
                support_flag = 1 if frame_id in support_frames else 0
            if rect_w_m <= 0 and "rect_w_m" in candidates.columns:
                rect_w_m = float(candidates["rect_w_m"].dropna().median() or 0.0)
            if rect_l_m <= 0 and "rect_l_m" in candidates.columns:
                rect_l_m = float(candidates["rect_l_m"].dropna().median() or 0.0)
            if rectangularity <= 0 and "rectangularity" in candidates.columns:
                rectangularity = float(candidates["rectangularity"].dropna().median() or 0.0)
        else:
            angle_jitter_p90 = ""
        records.append(
            {
                "drive_id": drive_id,
                "frame_id": frame_id,
                "image_path": index_lookup.get(key, ""),
                "frame_scanned": 1,
                "raw_attempted": 1,
                "raw_status": raw_info.get("raw_status", "unknown"),
                "raw_has_crosswalk": int(raw_info.get("raw_has_crosswalk", 0)),
                "raw_top_score": raw_info.get("raw_top_score", 0.0),
                "source_frame_id": source_frame_id,
                "x_ego": x_ego,
                "pose_ok": pose_ok,
                "pose_source": pose_source,
                "calib_ok": 1 if calib_ok else 0,
                "lidar_world_mode": lidar_world_mode,
                "proj_method": proj_method,
                "proj_in_image_ratio": float(lidar_info.get("proj_in_image_ratio", 0.0)),
                "proj_in_image_ratio_all": float(lidar_info.get("proj_in_image_ratio", 0.0)),
                "proj_in_image_ratio_visible": float(lidar_info.get("proj_in_image_ratio_visible", 0.0)),
                "points_total_visible": int(lidar_info.get("points_total_visible", 0)),
                "points_in_image_visible": int(lidar_info.get("points_in_image_visible", 0)),
                "points_in_bbox": int(lidar_info.get("points_in_bbox", 0)),
                "points_in_mask": int(lidar_info.get("points_in_mask", 0)),
                "points_in_mask_dilated": int(lidar_info.get("points_in_mask_dilated", 0)),
                "points_intensity_top": int(lidar_info.get("points_intensity_top", 0)),
                "points_ground": int(lidar_info.get("points_ground", 0)),
                "points_ground_global": int(lidar_info.get("points_ground_global", 0)),
                "points_ground_local": int(lidar_info.get("points_ground_local", 0)),
                "points_ground_plane": int(lidar_info.get("points_ground_plane", 0)),
                "ground_z_global": lidar_info.get("ground_z_global"),
                "ground_z_local": lidar_info.get("ground_z_local"),
                "z_p10": lidar_info.get("z_p10"),
                "z_p50": lidar_info.get("z_p50"),
                "z_p90": lidar_info.get("z_p90"),
                "plane_ok": int(lidar_info.get("plane_ok", 0)),
                "plane_dist_p10": lidar_info.get("plane_dist_p10"),
                "plane_dist_p50": lidar_info.get("plane_dist_p50"),
                "plane_dist_p90": lidar_info.get("plane_dist_p90"),
                "points_support": int(lidar_info.get("points_support", 0)),
                "points_support_accum": int(lidar_info.get("points_support_accum", 0)),
                "accum_frames_used": int(lidar_info.get("accum_frames_used", 0)),
                "mask_dilate_px": int(lidar_info.get("mask_dilate_px", 0)),
                "intensity_top_pct": int(lidar_info.get("intensity_top_pct", 0)),
                "mask_dilate_px_used": int(lidar_info.get("mask_dilate_px_used", 0)),
                "intensity_top_pct_used": int(lidar_info.get("intensity_top_pct_used", 0)),
                "ground_filter_used": int(lidar_info.get("ground_filter_used", 0)),
                "dbscan_points": int(lidar_info.get("dbscan_points", 0)),
                "support_source": str(lidar_info.get("support_source", "")),
                "geom_ok": geom_ok if geom_ok else int(lidar_info.get("geom_ok", 0)),
                "geom_area_m2": geom_area if geom_area else float(lidar_info.get("geom_area_m2", 0.0)),
                "rect_w_m": rect_w_m,
                "rect_l_m": rect_l_m,
                "rectangularity": rectangularity,
                "geom_ok_accum": int(lidar_info.get("geom_ok_accum", 0)),
                "geom_area_m2_accum": float(lidar_info.get("geom_area_m2_accum", 0.0)),
                "rect_w_m_accum": float(lidar_info.get("rect_w_m_accum", 0.0)),
                "rect_l_m_accum": float(lidar_info.get("rect_l_m_accum", 0.0)),
                "rectangularity_accum": float(lidar_info.get("rectangularity_accum", 0.0)),
                "geom_area_m2_single": float(lidar_info.get("geom_area_m2_single", 0.0)),
                "rect_w_m_single": float(lidar_info.get("rect_w_m_single", 0.0)),
                "rect_l_m_single": float(lidar_info.get("rect_l_m_single", 0.0)),
                "rectangularity_single": float(lidar_info.get("rectangularity_single", 0.0)),
                "cluster_id": cluster_id,
                "cluster_frames_hit": cluster_frames_hit,
                "jitter_p90": jitter_p90,
                "angle_jitter_p90": angle_jitter_p90,
                "support_flag": support_flag,
                "stage2_cluster_id": str(stage2.get("cluster_id") or ""),
                "stage2_added": int(stage2.get("stage2_added", 0)),
                "prop_ok": int(stage2.get("prop_ok", 0)),
                "prop_area_px": float(stage2.get("prop_area_px", 0.0)),
                "prop_drift_flag": int(stage2.get("prop_drift_flag", 0)),
                "prop_reason": str(stage2.get("prop_reason", "")),
                "frames_hit_support_after": cluster_frames_hit,
                "candidate_written": 1 if cand_ids else 0,
                "candidate_count": candidate_count,
                "reject_reasons": "|".join(rejects),
                "final_support": 1 if final_ids else 0,
                "final_entity_id": "|".join(final_ids),
                "drop_reason_code": drop_reason,
                "reproj_iou": roundtrip_by_frame.get(frame_id, {}).get("reproj_iou"),
                "reproj_center_err_px": roundtrip_by_frame.get(frame_id, {}).get("reproj_center_err_px"),
                "reproj_area_ratio": roundtrip_by_frame.get(frame_id, {}).get("reproj_area_ratio"),
            }
        )
    return records


def _build_report(
    out_path: Path,
    drive_id: str,
    frame_start: int,
    frame_end: int,
    trace_records: List[dict],
    candidate_gdf: gpd.GeoDataFrame,
    review_gdf: gpd.GeoDataFrame,
    final_gdf: gpd.GeoDataFrame,
    cluster_info: Dict[str, dict],
    outputs_dir: Path,
    final_cfg: dict,
    min_in_image_ratio: float,
    runtime_snapshot: Dict[str, str],
) -> None:
    lines = []
    lines.append("# Crosswalk Stage2 Report\n")
    lines.append(f"- report_time: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("")
    scan_stride = runtime_snapshot.get("scan_stride", "")
    scanned_frames_total = len(
        [r for r in trace_records if int(r.get("frame_scanned", 0)) == 1]
    ) or len(trace_records)
    raw_has_crosswalk_count = len(
        [r for r in trace_records if int(r.get("raw_has_crosswalk", 0)) == 1]
    )
    lines.append("## Runtime Config Snapshot")
    if runtime_snapshot:
        for key in sorted(runtime_snapshot.keys()):
            lines.append(f"- {key}: {runtime_snapshot[key]}")
    else:
        lines.append("- none")
    lines.append(f"- drive_id: {drive_id}")
    lines.append(f"- frame_range: {frame_start}-{frame_end}")
    lines.append(f"- total_frames: {frame_end - frame_start + 1}")
    lines.append(f"- scan_stride: {scan_stride}")
    lines.append(f"- scanned_frames_total: {scanned_frames_total}")
    lines.append(f"- raw_has_crosswalk_count: {raw_has_crosswalk_count}")
    lines.append("")
    n_pos = len([r for r in trace_records if int(r.get("raw_has_crosswalk", 0)) == 1 and r.get("raw_status") in {"ok", "on_demand_infer_ok"}])
    n_miss = len([r for r in trace_records if int(r.get("raw_has_crosswalk", 0)) == 1 and r.get("raw_status") in {"ok", "on_demand_infer_ok"} and int(r.get("candidate_written", 0)) == 0])
    lines.append(f"- N_pos: {n_pos}")
    lines.append(f"- N_miss: {n_miss}")
    lines.append(f"- candidate_count_total: {len(candidate_gdf)}")
    lines.append(f"- review_count: {len(review_gdf)}")
    lines.append(f"- final_count: {len(final_gdf)}")
    lines.append("")
    lines.append("## Calibration Thresholds")
    lines.append(f"- min_in_image_ratio: {min_in_image_ratio:.3f}")
    lines.append("")
    visible_vals = [float(r.get("proj_in_image_ratio_visible", 0.0)) for r in trace_records]
    if visible_vals:
        lines.append("## proj_in_image_ratio_visible Summary")
        lines.append(f"- p50: {np.percentile(visible_vals, 50):.3f}")
        lines.append(f"- p90: {np.percentile(visible_vals, 90):.3f}")
        lines.append("")
    bbox_vals = [int(r.get("points_in_bbox", 0) or 0) for r in trace_records if int(r.get("raw_has_crosswalk", 0)) == 1]
    if bbox_vals:
        lines.append("## points_in_bbox Summary")
        lines.append(f"- p50: {np.percentile(bbox_vals, 50):.1f}")
        lines.append(f"- p90: {np.percentile(bbox_vals, 90):.1f}")
        lines.append("")
    support_vals = [int(r.get("points_support", 0) or 0) for r in trace_records if int(r.get("raw_has_crosswalk", 0)) == 1]
    accum_vals = [int(r.get("points_support_accum", 0) or 0) for r in trace_records if int(r.get("raw_has_crosswalk", 0)) == 1]
    if support_vals:
        lines.append("## points_support Summary")
        lines.append(f"- p50: {np.percentile(support_vals, 50):.1f}")
        lines.append(f"- p90: {np.percentile(support_vals, 90):.1f}")
        lines.append("")
    if accum_vals:
        lines.append("## points_support_accum Summary")
        lines.append(f"- p50: {np.percentile(accum_vals, 50):.1f}")
        lines.append(f"- p90: {np.percentile(accum_vals, 90):.1f}")
        lines.append("")
    proj_methods = [str(r.get("proj_method") or "none") for r in trace_records]
    if proj_methods:
        lines.append("## proj_method Distribution")
        counts: Dict[str, int] = {}
        for val in proj_methods:
            counts[val] = counts.get(val, 0) + 1
        for key, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {key}: {count}")
        lines.append("")
    iou_vals = [float(r.get("reproj_iou") or 0.0) for r in trace_records if r.get("reproj_iou") is not None]
    if iou_vals:
        lines.append("## reproj_iou Summary")
        lines.append(f"- p50: {np.percentile(iou_vals, 50):.3f}")
        lines.append(f"- p90: {np.percentile(iou_vals, 90):.3f}")
        lines.append("")
    subset = [
        r
        for r in trace_records
        if int(r.get("geom_ok", 0)) == 1 and str(r.get("proj_method") or "") == "lidar"
    ]
    geom_areas = [float(r.get("geom_area_m2", 0.0)) for r in subset if float(r.get("geom_area_m2", 0.0)) > 0]
    if geom_areas:
        lines.append("## geom_area_m2 Summary")
        lines.append(f"- p50: {np.percentile(geom_areas, 50):.1f}")
        lines.append(f"- p90: {np.percentile(geom_areas, 90):.1f}")
        lines.append("- before_fix: N/A")
        lines.append("")
    reject_counts: Dict[str, int] = {}
    for row in trace_records:
        for reason in [r for r in str(row.get("reject_reasons") or "").split("|") if r]:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
    drop_counts: Dict[str, int] = {}
    for row in trace_records:
        code = str(row.get("drop_reason_code") or "")
        if code:
            drop_counts[code] = drop_counts.get(code, 0) + 1
    if reject_counts:
        lines.append("## Reject Reason Summary")
        for reason, count in sorted(reject_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            lines.append(f"- {reason}: {count}")
        lines.append("")
    if drop_counts:
        lines.append("## Drop Reason Summary")
        for reason, count in sorted(drop_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            lines.append(f"- {reason}: {count}")
        lines.append("")
    if cluster_info:
        frames_hit = [info.get("frames_hit_support", info.get("frames_hit", 0)) for info in cluster_info.values()]
        jitters = [info.get("jitter_p90", 0.0) for info in cluster_info.values()]
        gore_ratios = [info.get("gore_like_ratio", 0.0) for info in cluster_info.values()]
        stage2_added_total = int(sum([info.get("stage2_added_frames_count", 0) for info in cluster_info.values()]))
        lines.append("## cluster_frames_hit Summary")
        lines.append(f"- p50: {np.percentile(frames_hit, 50):.1f}")
        lines.append(f"- p90: {np.percentile(frames_hit, 90):.1f}")
        lines.append(f"- max: {max(frames_hit) if frames_hit else 0}")
        lines.append(f"- frames_hit_support_max_after: {max(frames_hit) if frames_hit else 0}")
        lines.append(f"- stage2_added_frames_total: {stage2_added_total}")
        lines.append("")
        if gore_ratios:
            lines.append("## gore_like_ratio Summary")
            lines.append(f"- p50: {np.percentile(gore_ratios, 50):.2f}")
            lines.append(f"- p90: {np.percentile(gore_ratios, 90):.2f}")
            lines.append("")
        lines.append("## jitter_p90 Summary")
        lines.append(f"- p50: {np.percentile(jitters, 50):.1f}")
        lines.append(f"- p90: {np.percentile(jitters, 90):.1f}")
        lines.append("")
        lines.append("## Top5 Near-Final Clusters")
        min_frames_final = int(final_cfg.get("min_frames_hit_final", 3))
        rect_min_w = float(final_cfg.get("rect_min_w", 1.5))
        rect_max_w = float(final_cfg.get("rect_max_w", 30.0))
        rect_min_l = float(final_cfg.get("rect_min_l", 3.0))
        rect_max_l = float(final_cfg.get("rect_max_l", 40.0))
        rect_min = float(final_cfg.get("rectangularity_min", 0.45))
        heading_max = float(final_cfg.get("heading_diff_to_perp_max_deg", 25.0))
        jitter_max = float(final_cfg.get("jitter_p90_max", 8.0))
        candidates = []
        for cid, info in cluster_info.items():
            if info.get("frames_hit_support", info.get("frames_hit", 0)) < 2:
                continue
            if cid in set(review_gdf.get("cluster_id", [])) or cid in set(final_gdf.get("cluster_id", [])):
                candidates.append((info.get("frames_hit_support", info.get("frames_hit", 0)), cid, info))
        candidates.sort(reverse=True)
        for frames_hit, cid, info in candidates[:5]:
            reasons = []
            if info.get("frames_hit_support", info.get("frames_hit", 0)) < min_frames_final:
                reasons.append("frames_hit_support<3")
            if info.get("rectangularity", 0.0) < rect_min:
                reasons.append("rectangularity")
            if info.get("rect_w_m", 0.0) < rect_min_w or info.get("rect_w_m", 0.0) > rect_max_w:
                reasons.append("rect_w")
            if info.get("rect_l_m", 0.0) < rect_min_l or info.get("rect_l_m", 0.0) > rect_max_l:
                reasons.append("rect_l")
            if info.get("heading_diff") is not None and info.get("heading_diff", 0.0) > heading_max:
                reasons.append("heading_diff")
            if info.get("jitter_p90", 0.0) > jitter_max:
                reasons.append("jitter")
            lines.append(f"- {cid} frames_hit={frames_hit} reasons={','.join(reasons) or 'n/a'}")
        lines.append("")
        metric_bug_clusters = []
        for cid, info in cluster_info.items():
            rect_w = float(info.get("rect_w_m", 0.0))
            rect_l = float(info.get("rect_l_m", 0.0))
            if rect_w < 0.5 or rect_l < 1.0:
                metric_bug_clusters.append(cid)
        if metric_bug_clusters:
            lines.append("## Metric Bug Suspected")
            for cid in metric_bug_clusters:
                frames = [
                    r.get("frame_id")
                    for r in trace_records
                    if str(r.get("cluster_id") or "") == cid
                ][:3]
                if not frames:
                    lines.append(f"- {cid}: no_frames_found")
                    continue
                lines.append(f"- {cid}: frames={','.join(frames)}")
                for frame_id in frames:
                    gated_path = outputs_dir / "qa_images" / drive_id / f"{frame_id}_overlay_gated.png"
                    raw_path = outputs_dir / "qa_images" / drive_id / f"{frame_id}_overlay_raw.png"
                    ent_path = outputs_dir / "qa_images" / drive_id / f"{frame_id}_overlay_entities.png"
                    lines.append(f"  - raw: {raw_path}")
                    lines.append(f"  - gated: {gated_path}")
                    lines.append(f"  - entities: {ent_path}")
            lines.append("")
        lines.append("## Top3 Cluster Before/After")
        rows = []
        for cid, info in cluster_info.items():
            before = int(info.get("frames_hit_support_before", 0))
            after = int(info.get("frames_hit_support", info.get("frames_hit", 0)))
            rows.append((after, cid, before, info.get("frames_hit_all_before", 0), info.get("frames_hit_all", 0)))
        rows.sort(reverse=True)
        for after, cid, before, all_before, all_after in rows[:3]:
            lines.append(f"- {cid}: support_before={before} support_after={after} all_before={all_before} all_after={all_after}")
    if proj_methods:
        lidar_ratio = counts.get("lidar", 0) / max(1, len(proj_methods))
        support_accum_p50 = float(np.percentile(accum_vals, 50)) if accum_vals else 0.0
        iou_p50 = float(np.percentile(iou_vals, 50)) if iou_vals else 0.0
        lines.append("")
        lines.append("## Conclusion")
        if support_accum_p50 > 0 and lidar_ratio >= 0.5 and iou_p50 > 0:
            lines.append("- 单帧稀疏已解决")
        else:
            lines.append("- 仍需检查帧对齐/索引/投影链路")
    lines.append(f"- outputs_dir: {outputs_dir}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _percentile(values: List[float], pct: float) -> float | None:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return float(np.percentile(arr, pct))


def _write_projection_alignment_report(
    out_path: Path,
    drive_id: str,
    frame_start: int,
    frame_end: int,
    roundtrip_rows: List[dict],
    offset_rows: List[dict],
    offset_hist: Dict[int, int],
    lidar_stats: Dict[str, dict],
    outputs_dir: Path,
) -> None:
    iou_lidar = [r["reproj_iou"] for r in roundtrip_rows if r.get("proj_method") == "lidar" and r.get("reproj_iou") is not None]
    iou_plane = [r["reproj_iou"] for r in roundtrip_rows if r.get("proj_method") == "plane" and r.get("reproj_iou") is not None]
    center_err = [r["reproj_center_err_px"] for r in roundtrip_rows if r.get("reproj_center_err_px") is not None]
    points_bbox = [int(r.get("points_in_bbox", 0) or 0) for r in roundtrip_rows]
    proj_ratio = [float(v.get("proj_in_image_ratio", 0.0) or 0.0) for v in lidar_stats.values()]
    proj_ratio_visible = [float(v.get("proj_in_image_ratio_visible", 0.0) or 0.0) for v in lidar_stats.values()]

    iou_lidar_p50 = _percentile(iou_lidar, 50)
    iou_lidar_p90 = _percentile(iou_lidar, 90)
    iou_plane_p50 = _percentile(iou_plane, 50)
    iou_plane_p90 = _percentile(iou_plane, 90)
    center_p50 = _percentile(center_err, 50)
    center_p90 = _percentile(center_err, 90)
    points_p50 = _percentile(points_bbox, 50)
    proj_ratio_p50 = _percentile(proj_ratio, 50)
    proj_ratio_p90 = _percentile(proj_ratio, 90)
    proj_ratio_visible_p50 = _percentile(proj_ratio_visible, 50)
    proj_ratio_visible_p90 = _percentile(proj_ratio_visible, 90)

    offset_mode = None
    offset_total = sum(offset_hist.values())
    if offset_hist:
        offset_mode = sorted(offset_hist.items(), key=lambda x: (-x[1], x[0]))[0][0]
    before = [r.get("iou_before") for r in offset_rows if r.get("iou_before") is not None]
    after = [r.get("iou_after") for r in offset_rows if r.get("iou_after") is not None]
    before_p50 = _percentile([v for v in before if v is not None], 50)
    after_p50 = _percentile([v for v in after if v is not None], 50)

    conclusion = "支撑点不足导致几何不稳"
    suggestion = "增加有效支撑点或放宽点数门槛后复测；确认 bbox/roi 覆盖是否正确。"
    if offset_mode not in (None, 0) and before_p50 is not None and after_p50 is not None and after_p50 - before_p50 > 0.05:
        conclusion = "帧对齐偏移"
        suggestion = "检查 frame_id 与 pose/velodyne 对齐索引；验证是否存在固定偏移。"
    elif iou_lidar_p50 is not None and iou_lidar_p50 < 0.1:
        if proj_ratio_p50 is not None and proj_ratio_p50 < 0.1:
            conclusion = "标定链路错误"
            suggestion = "检查相机标定矩阵与投影链路；验证 proj_in_image_ratio。"
        elif center_p50 is not None and center_p50 > 50:
            conclusion = "像素坐标回填/ROI坐标错误"
            suggestion = "检查 ROI/crop 的坐标回填到全图是否正确，核对 bbox/mask 是否超界。"

    lines = [
        "# Projection Alignment Report",
        f"- report_time: {dt.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- drive_id: {drive_id}",
        f"- frame_range: {frame_start}-{frame_end}",
        "",
        "## Roundtrip IoU Summary",
        f"- lidar_iou_p50: {iou_lidar_p50}",
        f"- lidar_iou_p90: {iou_lidar_p90}",
        f"- plane_iou_p50: {iou_plane_p50}",
        f"- plane_iou_p90: {iou_plane_p90}",
        f"- center_err_px_p50: {center_p50}",
        f"- center_err_px_p90: {center_p90}",
        f"- proj_in_image_ratio_p50: {proj_ratio_p50}",
        f"- proj_in_image_ratio_p90: {proj_ratio_p90}",
        f"- proj_in_image_ratio_visible_p50: {proj_ratio_visible_p50}",
        f"- proj_in_image_ratio_visible_p90: {proj_ratio_visible_p90}",
        "",
        "## Offset Scan",
        f"- best_offset_mode: {offset_mode}",
        f"- best_offset_samples: {offset_total}",
        f"- iou_before_p50: {before_p50}",
        f"- iou_after_p50: {after_p50}",
        "",
        "## Conclusion",
        f"- category: {conclusion}",
        f"- suggestion: {suggestion}",
        "",
        "## Outputs",
        f"- roundtrip_metrics: {outputs_dir / 'roundtrip_metrics.csv'}",
        f"- best_offset_summary: {outputs_dir / 'best_offset_summary.csv'}",
        f"- best_offset_hist: {outputs_dir / 'best_offset_hist.csv'}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _final_fail_reason(info: dict, final_cfg: dict) -> str:
    min_frames_final = int(final_cfg.get("min_frames_hit_final", 3))
    min_inside = float(final_cfg.get("min_inside_ratio_final", 0.5))
    rect_min_w = float(final_cfg.get("rect_min_w", 1.5))
    rect_max_w = float(final_cfg.get("rect_max_w", 8.0))
    rect_min_l = float(final_cfg.get("rect_min_l", 3.0))
    rect_max_l = float(final_cfg.get("rect_max_l", 25.0))
    rect_min = float(final_cfg.get("rectangularity_min", 0.45))
    heading_max = float(final_cfg.get("heading_diff_to_perp_max_deg", 25.0))
    jitter_max = float(final_cfg.get("jitter_p90_max", 8.0))
    frames_hit_support = int(info.get("frames_hit_support", 0))
    rect_w = float(info.get("rect_w_m", 0.0))
    rect_l = float(info.get("rect_l_m", 0.0))
    rectangularity = float(info.get("rectangularity", 0.0))
    heading_diff = float(info.get("heading_diff", 0.0)) if info.get("heading_diff") is not None else 0.0
    jitter_p90 = float(info.get("jitter_p90", 0.0))
    inside_ratio = info.get("inside_road_ratio")
    inside_ratio = float(inside_ratio) if inside_ratio is not None else 1.0
    if frames_hit_support < min_frames_final:
        return "frames_hit_support"
    if not (rect_min_w <= rect_w <= rect_max_w) or not (rect_min_l <= rect_l <= rect_max_l):
        return "size"
    if rectangularity < rect_min:
        return "rectangularity"
    if heading_diff > heading_max:
        return "heading_diff"
    if jitter_p90 > jitter_max:
        return "jitter"
    if inside_ratio < min_inside:
        return "inside_ratio"
    return "pass"


def _write_cluster_summary(
    outputs_dir: Path,
    drive_id: str,
    cluster_info: Dict[str, dict],
    final_gdf: gpd.GeoDataFrame,
    final_cfg: dict,
) -> Tuple[Path, Path]:
    rows = []
    final_clusters = set(final_gdf.get("cluster_id", [])) if not final_gdf.empty else set()
    for cid, info in cluster_info.items():
        frames_hit_all = int(info.get("frames_hit_all", info.get("frames_hit", 0)))
        frames_hit_support = int(info.get("frames_hit_support", 0))
        frames_hit_all_before = int(info.get("frames_hit_all_before", 0))
        frames_hit_support_before = int(info.get("frames_hit_support_before", 0))
        stage2_added_count = int(info.get("stage2_added_frames_count", 0))
        frames_hit_stage1 = frames_hit_all_before
        gore_like_ratio = float(info.get("gore_like_ratio", 0.0))
        support_frames = info.get("support_frames", [])
        refined_rect_w = float(info.get("rect_w_m", 0.0))
        refined_rect_l = float(info.get("rect_l_m", 0.0))
        metric_bug_suspected = 1 if refined_rect_w < 0.5 or refined_rect_l < 1.0 else 0
        refined_geom = info.get("refined_geom")
        refined_rect_area = float(refined_geom.area) if refined_geom is not None and not refined_geom.is_empty else 0.0
        rectangularity = float(info.get("rectangularity", 0.0))
        jitter_p90 = float(info.get("jitter_p90", 0.0))
        angle_jitter_p90 = float(info.get("angle_jitter_p90", 0.0))
        final_pass = 1 if cid in final_clusters else 0
        fail_reason = "pass" if final_pass else _final_fail_reason(info, final_cfg)
        rows.append(
            {
                "cluster_id": cid,
                "drive_id": drive_id,
                "frames_hit_all": frames_hit_all,
                "frames_hit_all_before": frames_hit_all_before,
                "frames_hit_support_before": frames_hit_support_before,
                "frames_hit_stage1": frames_hit_stage1,
                "frames_hit_stage2_added": stage2_added_count,
                "frames_hit_support": frames_hit_support,
                "support_frames": "|".join(support_frames[:10]),
                "stage2_added_frames_count": stage2_added_count,
                "gore_like_ratio": gore_like_ratio,
                "jitter_p90": jitter_p90,
                "angle_jitter_p90": angle_jitter_p90,
                "refined_rect_w_m": refined_rect_w,
                "refined_rect_l_m": refined_rect_l,
                "refined_rect_area_m2": refined_rect_area,
                "rectangularity": rectangularity,
                "metric_bug_suspected": metric_bug_suspected,
                "final_pass": final_pass,
                "final_fail_reason": fail_reason,
            }
        )
    csv_path = outputs_dir / "cluster_summary.csv"
    md_path = outputs_dir / "cluster_summary.md"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    lines = ["# Cluster Summary", f"- drive_id: {drive_id}", f"- clusters: {len(rows)}", ""]
    for row in rows:
        lines.append(
            f"- {row['cluster_id']}: stage1={row['frames_hit_stage1']} stage2_added={row['frames_hit_stage2_added']} support={row['frames_hit_support']} gore_like_ratio={row['gore_like_ratio']:.2f} metric_bug={row['metric_bug_suspected']} final_pass={row['final_pass']} reason={row['final_fail_reason']}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crosswalk_fix_range.yaml")
    ap.add_argument("--drive", default=None)
    ap.add_argument("--frame-start", type=int, default=None)
    ap.add_argument("--frame-end", type=int, default=None)
    ap.add_argument("--auto-frame-range", type=int, default=0)
    ap.add_argument("--kitti-root", default=None)
    ap.add_argument("--out-run", default="")
    args = ap.parse_args()

    log = _setup_logger()
    cfg = _load_yaml(Path(args.config))
    merged = _merge_config(
        cfg,
        {
            "drive_id": args.drive,
            "frame_start": args.frame_start,
            "frame_end": args.frame_end,
            "kitti_root": args.kitti_root,
            "auto_frame_range": bool(args.auto_frame_range),
        },
    )

    drive_id = str(merged.get("drive_id") or "")
    frame_start = int(merged.get("frame_start", 0)) if merged.get("frame_start") is not None else None
    frame_end = int(merged.get("frame_end", 0)) if merged.get("frame_end") is not None else None
    kitti_root = Path(str(merged.get("kitti_root") or ""))
    defaults = _load_camera_defaults(Path("configs/camera_defaults.yaml"))
    default_camera = str(defaults.get("default_camera") or "image_00")
    enforce_camera = bool(defaults.get("enforce_camera", True))
    allow_override = bool(defaults.get("allow_override", False))
    if str(os.environ.get("ALLOW_CAMERA_OVERRIDE", "0")).strip() == "1":
        allow_override = True
    camera = str(merged.get("camera") or default_camera)
    stage1_stride = int(merged.get("stage1_stride", 1))
    export_all_frames = bool(merged.get("export_all_frames", True))
    write_wgs84 = bool(merged.get("write_wgs84", True))
    raw_fallback_text = bool(merged.get("raw_fallback_text", True))
    on_demand_infer = bool(merged.get("on_demand_infer", False))
    lidar_world_mode = str(merged.get("lidar_world_mode") or os.environ.get("LIDAR_WORLD_MODE") or "").strip().lower()
    if str(os.environ.get("USE_FULLPOSE_LIDAR", "0")).strip() == "1":
        lidar_world_mode = "fullpose"
    if not lidar_world_mode:
        lidar_world_mode = "fullpose"
    image_run = Path(str(merged.get("image_run") or ""))
    image_provider = str(merged.get("image_provider") or "grounded_sam2_v1")
    image_evidence_gpkg = str(merged.get("image_evidence_gpkg") or "")
    road_root = Path(str(merged.get("road_root") or ""))
    base_config = Path(str(merged.get("config") or "configs/road_entities.yaml"))
    lidar_cfg = merged.get("lidar_proj", {}) if isinstance(merged.get("lidar_proj"), dict) else {}
    cluster_cfg = merged.get("cluster", {}) if isinstance(merged.get("cluster"), dict) else {}
    refine_cfg = merged.get("refine", {}) if isinstance(merged.get("refine"), dict) else {}
    final_cfg = merged.get("final_gate", {}) if isinstance(merged.get("final_gate"), dict) else {}
    review_cfg = merged.get("review_gate", {}) if isinstance(merged.get("review_gate"), dict) else {}

    auto_range = bool(merged.get("auto_frame_range", False) or args.auto_frame_range)
    if auto_range:
        image_dir = _find_image_dir(kitti_root, drive_id, camera)
        frames = []
        for path in _list_images(image_dir):
            fid = _extract_frame_id(path)
            if fid is None:
                continue
            frames.append(int(str(fid)))
        if not frames:
            log.error("auto frame range failed: no images")
            return 2
        frame_start = min(frames)
        frame_end = max(frames)
    if frame_start is None or frame_end is None or not drive_id or frame_end < frame_start:
        log.error("invalid drive/frame range")
        return 2
    if not kitti_root.exists():
        log.error("kitti_root missing: %s", kitti_root)
        return 2
    if not image_run.exists():
        log.error("image_run missing: %s", image_run)
        return 2
    if not road_root.exists():
        log.error("road_root missing: %s", road_root)
        return 2

    os.environ["POC_DATA_ROOT"] = str(kitti_root)
    if stage1_stride != 1:
        log.warning("stage1_stride=%s overridden to 1 for full coverage.", stage1_stride)
        stage1_stride = 1

    run_dir = Path(args.out_run) if args.out_run else Path("runs") / f"crosswalk_fix_{drive_id.split('_')[-2]}_{frame_start}_{frame_end}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    outputs_dir = run_dir / "outputs"
    debug_dir = run_dir / "debug"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    image_dir = _find_image_dir(kitti_root, drive_id, camera)
    try:
        calib = load_kitti360_calib(kitti_root, camera)
    except Exception:
        calib = None
    cam_to_pose_key = ""
    cam_to_pose = None
    try:
        cam_to_pose, cam_to_pose_key = load_kitti360_cam_to_pose_key(kitti_root, camera)
    except Exception:
        cam_to_pose_key = ""
    _assert_camera_consistency(
        camera,
        image_dir,
        calib,
        cam_to_pose_key,
        lidar_world_mode,
        default_camera,
        enforce_camera,
        allow_override,
    )

    index_records = _build_index_records(kitti_root, drive_id, frame_start, frame_end, camera)
    expected_frames = frame_end - frame_start + 1
    scanned_frames_total = len(index_records)
    if scanned_frames_total != expected_frames:
        log.error("scanned_frames_total=%d expected=%d", scanned_frames_total, expected_frames)
        return 2
    index_path = debug_dir / "monitor_index.jsonl"
    _write_index(index_records, index_path)
    log.info("index=%s total=%d", index_path, len(index_records))

    stage_cfg = debug_dir / "crosswalk_fix.yaml"
    stage_cfg.write_text(
        yaml.safe_dump(
            _merge_config(_load_yaml(base_config), {"crosswalk_final": {"min_frames_hit": 3}}),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    stage_dir = run_dir / "stage1"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    cmd = [
        sys.executable,
        "tools/build_road_entities.py",
        "--index",
        str(index_path),
        "--image-run",
        str(image_run),
        "--image-provider",
        str(image_provider),
        "--image-evidence-gpkg",
        str(image_evidence_gpkg),
        "--config",
        str(stage_cfg),
        "--road-root",
        str(road_root),
        "--out-dir",
        str(stage_dir),
        "--emit-qa-images",
        "1",
        "--lidar-world-mode",
        str(lidar_world_mode),
    ]
    log.info("stage1: %s", " ".join(cmd))
    if subprocess.run(cmd, check=False).returncode != 0:
        log.error("stage1 failed")
        return 3

    stage_outputs = stage_dir / "outputs"
    frame_evidence_path = stage_outputs / "frame_evidence_utm32.gpkg"
    try:
        _assert_frame_evidence_frame_id(
            frame_evidence_path,
            drive_id,
            frame_start,
            frame_end,
            stage_outputs / "frame_id_assert.txt",
        )
    except Exception as exc:
        log.error("frame_id assert failed: %s", exc)
        return 4
    qa_images_src = stage_outputs / "qa_images"
    qa_images_dst = outputs_dir / "qa_images"
    if qa_images_dst.exists():
        shutil.rmtree(qa_images_dst)
    if qa_images_src.exists():
        shutil.copytree(qa_images_src, qa_images_dst)

    qa_index_path = stage_outputs / "qa_index_wgs84.geojson"
    qa_out_path = outputs_dir / "qa_index_wgs84.geojson"
    if qa_out_path.exists():
        qa_out_path.unlink()
    if qa_index_path.exists():
        shutil.copy2(qa_index_path, qa_out_path)

    index_lookup = {(r["drive_id"], _normalize_frame_id(r["frame_id"])): r.get("image_path", "") for r in index_records}
    raw_stats, raw_frames = _ensure_raw_overlays(
        qa_out_path,
        outputs_dir,
        image_run,
        image_provider,
        index_lookup,
        raw_fallback_text,
        on_demand_infer,
        (debug_dir / "on_demand_infer").resolve(),
        drive_id,
        camera,
    )
    try:
        hit_frames = [
            frame_id
            for (d, frame_id), stats in raw_stats.items()
            if d == drive_id
            and int(stats.get("raw_has_crosswalk", 0)) == 1
            and stats.get("raw_status") in {"ok", "on_demand_infer_ok"}
        ]
        _assert_frame_evidence_hit_frames(
            stage_outputs / "frame_evidence_utm32.gpkg",
            drive_id,
            hit_frames,
            stage_outputs / "frame_id_hit_assert.txt",
            min_ratio=0.9,
        )
    except Exception as exc:
        log.error("frame_id hit-assert failed: %s", exc)
        return 4

    stage_gpkg = stage_outputs / "road_entities_utm32.gpkg"
    candidate_gdf = _read_candidates(stage_gpkg)
    if not candidate_gdf.empty and "source_frame_id" not in candidate_gdf.columns:
        candidate_gdf = candidate_gdf.copy()
        candidate_gdf["source_frame_id"] = candidate_gdf["frame_id"]
    pose_map: Dict[str, Tuple[float, float, float] | Tuple[float, float, float, float, float, float] | None] = {}
    for frame in range(frame_start, frame_end + 1):
        frame_id = _normalize_frame_id(str(frame))
        try:
            if lidar_world_mode == "fullpose":
                x, y, z, roll, pitch, yaw = load_kitti360_pose_full(kitti_root, drive_id, frame_id)
                pose_map[frame_id] = (x, y, z, roll, pitch, yaw)
            else:
                x, y, yaw = load_kitti360_pose(kitti_root, drive_id, frame_id)
                pose_map[frame_id] = (x, y, yaw)
        except Exception:
            pose_map[frame_id] = None
    calib_ok = calib is not None
    alignment_cfg = merged.get("alignment", {}) if isinstance(merged.get("alignment"), dict) else {}
    visible_cfg = alignment_cfg.get("visible_filter", {}) if isinstance(alignment_cfg.get("visible_filter"), dict) else {}
    lidar_cfg_norm = {
        "MASK_DILATE_PX": int(lidar_cfg.get("mask_dilate_px", 5)),
        "MIN_POINTS_BBOX": int(lidar_cfg.get("min_points_bbox", 20)),
        "MIN_POINTS_MASK": int(lidar_cfg.get("min_points_mask", 5)),
        "INTENSITY_TOP_PCT": int(lidar_cfg.get("intensity_top_pct", 10)),
        "MIN_IN_IMAGE_RATIO": float(lidar_cfg.get("min_in_image_ratio", 0.1)),
        "GROUND_Z_TOL": float(lidar_cfg.get("ground_z_tol", 0.2)),
        "DBSCAN_EPS_M": float(lidar_cfg.get("dbscan_eps_m", 0.6)),
        "DBSCAN_MIN_SAMPLES": int(lidar_cfg.get("dbscan_min_samples", 10)),
        "ANGLE_SNAP_DEG": float(lidar_cfg.get("angle_snap_deg", 20)),
        "ACCUM_PRE": int(lidar_cfg.get("accum_pre", 5)),
        "ACCUM_POST": int(lidar_cfg.get("accum_post", 5)),
        "VISIBLE_X_FORWARD_ONLY": bool(visible_cfg.get("x_forward_only", True)),
        "VISIBLE_MIN_RANGE_M": float(visible_cfg.get("min_range_m", 1.0)),
        "VISIBLE_MAX_RANGE_M": float(visible_cfg.get("max_range_m", 80.0)),
    }
    lidar_candidates, lidar_stats = _build_lidar_candidates_for_range(
        kitti_root,
        drive_id,
        frame_start,
        frame_end,
        index_lookup,
        raw_frames,
        pose_map,
        calib,
        cam_to_pose,
        lidar_cfg_norm,
        lidar_world_mode,
        camera,
        outputs_dir / "debug",
    )
    point_support_dir = outputs_dir / "debug"
    point_support_dir.mkdir(parents=True, exist_ok=True)
    for frame in range(frame_start, frame_end + 1):
        frame_id = _normalize_frame_id(str(frame))
        stats = lidar_stats.get(frame_id, {})
        raw_info = raw_stats.get(
            (drive_id, frame_id),
            {"raw_status": "unknown", "raw_has_crosswalk": 0.0},
        )
        payload = {
            "drive_id": drive_id,
            "frame_id": frame_id,
            "raw_status": str(raw_info.get("raw_status", "unknown")),
            "raw_has_crosswalk": int(raw_info.get("raw_has_crosswalk", 0)),
            "accum_pre": int(lidar_cfg_norm.get("ACCUM_PRE", 0)),
            "accum_post": int(lidar_cfg_norm.get("ACCUM_POST", 0)),
            "accum_frames_used": int(stats.get("accum_frames_used", 0)),
            "points_in_bbox": int(stats.get("points_in_bbox", 0)),
            "points_in_mask_dilated": int(stats.get("points_in_mask_dilated", 0)),
            "points_intensity_top": int(stats.get("points_intensity_top", 0)),
            "points_ground": int(stats.get("points_ground", 0)),
            "points_ground_global": int(stats.get("points_ground_global", 0)),
            "points_ground_local": int(stats.get("points_ground_local", 0)),
            "points_ground_plane": int(stats.get("points_ground_plane", 0)),
            "ground_z_global": stats.get("ground_z_global"),
            "ground_z_local": stats.get("ground_z_local"),
            "z_p01": stats.get("z_p01"),
            "z_p10": stats.get("z_p10"),
            "z_p50": stats.get("z_p50"),
            "z_p90": stats.get("z_p90"),
            "z_p99": stats.get("z_p99"),
            "dz_p01": stats.get("dz_p01"),
            "dz_p10": stats.get("dz_p10"),
            "dz_p50": stats.get("dz_p50"),
            "dz_p90": stats.get("dz_p90"),
            "dz_p99": stats.get("dz_p99"),
            "plane_ok": int(stats.get("plane_ok", 0)),
            "plane_dist_p10": stats.get("plane_dist_p10"),
            "plane_dist_p50": stats.get("plane_dist_p50"),
            "plane_dist_p90": stats.get("plane_dist_p90"),
            "points_support": int(stats.get("points_support", 0)),
            "points_support_accum": int(stats.get("points_support_accum", 0)),
            "proj_method": str(stats.get("proj_method", "")),
            "drop_reason_code": str(stats.get("drop_reason_code", "")),
            "support_source": str(stats.get("support_source", "")),
            "mask_dilate_px_used": int(stats.get("mask_dilate_px_used", 0)),
            "intensity_top_pct_used": int(stats.get("intensity_top_pct_used", 0)),
            "geom_ok_accum": int(stats.get("geom_ok_accum", 0)),
            "geom_area_m2_accum": float(stats.get("geom_area_m2_accum", 0.0)),
            "rect_w_m_accum": float(stats.get("rect_w_m_accum", 0.0)),
            "rect_l_m_accum": float(stats.get("rect_l_m_accum", 0.0)),
            "rectangularity_accum": float(stats.get("rectangularity_accum", 0.0)),
        }
        (point_support_dir / f"point_support_{frame_id}.json").write_text(
            json.dumps(payload, ensure_ascii=True),
            encoding="utf-8",
        )
    if lidar_candidates:
        lidar_gdf = gpd.GeoDataFrame.from_features(lidar_candidates, crs="EPSG:32632")
        if candidate_gdf.empty:
            candidate_gdf = lidar_gdf
        else:
            candidate_gdf = gpd.GeoDataFrame(
                pd.concat([candidate_gdf, lidar_gdf], ignore_index=True),
                geometry="geometry",
                crs="EPSG:32632",
            )
    plane_score_min = float(lidar_cfg.get("plane_score_min", 0.3))
    candidate_gdf, fallback_frames = _augment_candidates_with_fallback(
        candidate_gdf,
        raw_stats,
        pose_map,
        calib_ok,
        raw_frames,
        lidar_stats,
        drive_id,
        frame_start,
        frame_end,
        plane_score_min,
    )
    if not candidate_gdf.empty:
        candidate_gdf = candidate_gdf.copy()
        candidate_gdf["rect_w_m"] = candidate_gdf.get("rect_w_m", 0.0)
        candidate_gdf["rect_l_m"] = candidate_gdf.get("rect_l_m", 0.0)
        candidate_gdf["rectangularity"] = candidate_gdf.get("rectangularity", 0.0)
        candidate_gdf["reject_reasons"] = candidate_gdf.get("reject_reasons", "").fillna("").astype(str)
        gore_flags = []
        for _, row in candidate_gdf.iterrows():
            gore_like = _is_gore_like(float(row.get("rect_w_m") or 0.0), float(row.get("rect_l_m") or 0.0), float(row.get("rectangularity") or 0.0))
            gore_flags.append(gore_like)
        candidate_gdf["gore_like"] = gore_flags
        candidate_gdf.loc[candidate_gdf["gore_like"], "reject_reasons"] = candidate_gdf.loc[candidate_gdf["gore_like"], "reject_reasons"].apply(
            lambda v: _append_reject_reason(v, "gore_like")
        )
    candidate_gdf, cluster_info = _build_clusters(candidate_gdf, float(cluster_cfg.get("cluster_eps_m", 10.0)))
    candidate_gdf, cluster_info = _refine_clusters(candidate_gdf, cluster_info, pose_map, lidar_stats, raw_stats, drive_id, refine_cfg)
    for info in cluster_info.values():
        info["frames_hit_all_before"] = int(info.get("frames_hit_all", info.get("frames_hit", 0)))
        info["frames_hit_support_before"] = int(info.get("frames_hit_support", 0))
        info.setdefault("stage2_added_frames_count", 0)
        info.setdefault("stage2_support_frames", [])
    stage2_cfg = merged.get("stage2", {}) if isinstance(merged.get("stage2"), dict) else {}
    stage2_window = int(stage2_cfg.get("stage2_window", 50))
    prop_min_area = float(stage2_cfg.get("prop_min_area_px", 400.0))
    prop_max_ratio = float(stage2_cfg.get("prop_max_area_ratio", 0.6))
    prop_min_ratio = float(stage2_cfg.get("prop_min_area_ratio", stage2_cfg.get("AREA_RATIO_MIN", 0.0)))
    rel_area_min = float(stage2_cfg.get("area_ratio_min", stage2_cfg.get("AREA_RATIO_MIN", 0.0)))
    rel_area_max = float(stage2_cfg.get("area_ratio_max", stage2_cfg.get("AREA_RATIO_MAX", 3.0)))
    rect_min_support = float(stage2_cfg.get("rectangularity_min_support", 0.35))
    allow_gore_support = bool(stage2_cfg.get("allow_gore_like_support", False))
    roi_margin_px = int(stage2_cfg.get("roi_margin_px", stage2_cfg.get("ROI_MARGIN_PX", 80)))
    iou_prev_min = float(stage2_cfg.get("iou_prev_min", stage2_cfg.get("IOU_PREV_MIN", 0.05)))
    center_drift_ratio = float(stage2_cfg.get("center_drift_ratio", stage2_cfg.get("CENTER_DRIFT_RATIO", 0.35)))
    drift_consec_stop = int(stage2_cfg.get("drift_consec_stop", stage2_cfg.get("DRIFT_CONSEC_STOP", 3)))
    stage2_stats: Dict[Tuple[str, str], dict] = {}
    stage2_overlay: Dict[str, List[dict]] = {}

    image_dir = _find_image_dir(kitti_root, drive_id, camera)
    frame_ids = [_normalize_frame_id(str(frame)) for frame in range(frame_start, frame_end + 1)]
    frame_index = {frame_id: idx for idx, frame_id in enumerate(frame_ids)}
    stage2_video_dir = debug_dir / "stage2_video_frames"
    prepare_video_frames(image_dir, frame_ids, stage2_video_dir)
    predictor = build_video_predictor(image_provider)

    seeds = []
    seed_dir = outputs_dir / "stage2_seeds"
    for cid in cluster_info.keys():
        seeds.extend(
            _choose_stage2_seeds(
                drive_id,
                cid,
                candidate_gdf,
                raw_stats,
                raw_frames,
                index_lookup,
                seed_dir,
                int(stage2_cfg.get("seeds_per_cluster", 1)),
                float(stage2_cfg.get("min_seed_dist_m", 5.0)),
            )
        )
    _write_seeds_jsonl(seeds, outputs_dir / "seeds.jsonl")

    stage2_candidate_rows: List[dict] = []
    if predictor is not None:
        for seed in seeds:
            cid = seed["cluster_id"]
            seed_frame = seed["seed_frame_id"]
            seed_idx = frame_ids.index(seed_frame) if seed_frame in frame_ids else None
            if seed_idx is None:
                continue
            seed_mask_path = Path(seed["seed_mask_path"]) if seed.get("seed_mask_path") else None
            seed_bbox = seed.get("seed_bbox_px")
            if seed_bbox is None and seed_mask_path is not None and seed_mask_path.exists():
                seed_mask = np.array(Image.open(seed_mask_path).convert("L")) > 0
                seed_bbox = _mask_to_bbox(seed_mask)
            seed_bbox = seed_bbox if isinstance(seed_bbox, list) else None
            masks = propagate_seed(
                predictor,
                stage2_video_dir,
                seed_idx,
                seed_mask_path,
                seed_bbox,
                stage2_window,
            )
            out_mask_dir = outputs_dir / "stage2_masks" / cid
            saved = save_masks(masks, frame_ids, out_mask_dir)
            if not saved:
                continue
            image_path = index_lookup.get((drive_id, seed_frame), "")
            width, height = _load_image_size(image_path)
            roi_bbox = None
            if seed_bbox is not None and width > 0 and height > 0 and roi_margin_px > 0:
                roi_bbox = _expand_bbox(seed_bbox, float(roi_margin_px), width, height)
            prev_bbox = None
            prev_area = None
            consec_drift = 0
            for frame_id, mask_path in sorted(saved.items(), key=lambda item: frame_index.get(item[0], 1_000_000)):
                mask = np.array(Image.open(mask_path).convert("L")) > 0
                area = mask_area_px(mask)
                image_path = index_lookup.get((drive_id, frame_id), "")
                width, height = _load_image_size(image_path)
                image_area = float(width * height) if width > 0 and height > 0 else 0.0
                area_ratio = (area / image_area) if image_area > 0 else 0.0
                area_max = image_area * prop_max_ratio if image_area > 0 else 0.0
                mask_bbox = _mask_to_bbox(mask)
                drift_reasons = []
                if area < prop_min_area:
                    drift_reasons.append("area_min")
                if image_area > 0 and area_ratio < prop_min_ratio:
                    drift_reasons.append("area_ratio_min")
                if area_max > 0.0 and area > area_max:
                    drift_reasons.append("area_ratio_max")
                if mask_bbox is None:
                    drift_reasons.append("bbox_missing")
                if roi_bbox is not None and mask_bbox is not None:
                    center = _bbox_center(mask_bbox)
                    if not _bbox_contains_point(roi_bbox, center):
                        drift_reasons.append("roi_center")
                if prev_bbox is not None and mask_bbox is not None:
                    if _bbox_iou(prev_bbox, mask_bbox) < iou_prev_min:
                        drift_reasons.append("iou_prev")
                    center = _bbox_center(mask_bbox)
                    prev_center = _bbox_center(prev_bbox)
                    max_dim = float(max(width, height)) if width > 0 and height > 0 else 0.0
                    if roi_bbox is not None:
                        max_dim = float(max(roi_bbox[2] - roi_bbox[0], roi_bbox[3] - roi_bbox[1]))
                    if max_dim > 0:
                        dist = math.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
                        if dist / max_dim > center_drift_ratio:
                            drift_reasons.append("center_drift")
                if prev_area is not None and prev_area > 0:
                    ratio = area / prev_area if prev_area > 0 else 0.0
                    if ratio < rel_area_min or ratio > rel_area_max:
                        drift_reasons.append("area_ratio_prev")
                prop_ok = int(len(drift_reasons) == 0)
                prop_drift = int(len(drift_reasons) > 0)
                if prop_drift:
                    consec_drift += 1
                else:
                    consec_drift = 0
                if drift_consec_stop > 0 and consec_drift >= drift_consec_stop:
                    break
                key = (drive_id, frame_id)
                stat = stage2_stats.setdefault(
                    key,
                    {
                        "cluster_id": cid,
                        "stage2_added": 0,
                        "prop_ok": 0,
                        "prop_area_px": 0.0,
                        "prop_drift_flag": 0,
                        "prop_reason": "",
                    },
                )
                stat["cluster_id"] = cid
                stat["prop_area_px"] = area
                stat["prop_drift_flag"] = prop_drift
                stat["prop_ok"] = prop_ok
                stat["prop_reason"] = ",".join(drift_reasons)
                if prop_ok:
                    stat["stage2_added"] = 1
                    prev_bbox = mask_bbox or prev_bbox
                    prev_area = area
                    info = cluster_info.get(cid, {})
                    info["stage2_added_frames_count"] = int(info.get("stage2_added_frames_count", 0)) + 1
                    poly = _mask_to_polygon(mask)
                    rect_val = 0.0
                    aspect = 0.0
                    if poly is not None:
                        rect = poly.minimum_rotated_rectangle
                        metrics = _rect_metrics(poly)
                        rect_val = float(metrics.get("rectangularity", 0.0))
                        aspect = float(metrics.get("aspect", 0.0))
                    is_gore = False
                    if not allow_gore_support:
                        existing = candidate_gdf[candidate_gdf.get("frame_id", "") == frame_id]
                        if not existing.empty:
                            is_gore = bool(existing.get("gore_like", False).fillna(False).any())
                    if rect_val >= rect_min_support and not is_gore:
                        info.setdefault("stage2_support_frames", []).append(frame_id)
                    bbox = None
                    if poly is not None:
                        minx, miny, maxx, maxy = poly.bounds
                        bbox = [float(minx), float(miny), float(maxx), float(maxy)]
                    stage2_overlay.setdefault(frame_id, []).append(
                        {"mask_path": str(mask_path), "bbox_px": bbox, "cluster_id": cid}
                    )
                    info["stage2_support_frames"] = list(dict.fromkeys(info.get("stage2_support_frames", [])))
                    support_frames = info.get("support_frames", [])
                    merged_support = list(dict.fromkeys(support_frames + info.get("stage2_support_frames", [])))
                    info["support_frames"] = merged_support
                    info["frames_hit_support"] = len(merged_support)
                    info["frames_hit_all"] = int(info.get("frames_hit_all_before", 0)) + int(
                        info.get("stage2_added_frames_count", 0)
                    )
                    if bbox is None and seed_bbox:
                        bbox = seed_bbox
                    if bbox:
                        stage2_candidate_rows.append(
                            {
                                "geometry": info.get("refined_geom"),
                                "properties": {
                                "candidate_id": f"{drive_id}_crosswalk_stage2_{cid}_{frame_id}",
                                "drive_id": drive_id,
                                "frame_id": frame_id,
                                "source_frame_id": seed_frame,
                                "entity_type": "crosswalk",
                                    "reject_reasons": "stage2_video",
                                    "proj_method": "stage2_video",
                                    "geom_ok": 0,
                                    "stage2_added": 1,
                                    "bbox_px": bbox,
                                    "cluster_id": cid,
                                    "qa_flag": "stage2_added",
                                },
                            }
                        )
    if stage2_candidate_rows:
        stage2_gdf = gpd.GeoDataFrame.from_features(stage2_candidate_rows, crs="EPSG:32632")
        candidate_gdf = gpd.GeoDataFrame(
            pd.concat([candidate_gdf, stage2_gdf], ignore_index=True),
            geometry="geometry",
            crs="EPSG:32632",
        )
    frame_candidates_path = outputs_dir / "frame_candidates_utm32.gpkg"
    if frame_candidates_path.exists():
        frame_candidates_path.unlink()
    if candidate_gdf.empty:
        gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32632").to_file(
            frame_candidates_path,
            layer="frame_candidates",
            driver="GPKG",
        )
    else:
        candidate_gdf.to_file(frame_candidates_path, layer="frame_candidates", driver="GPKG")
    roundtrip_rows, roundtrip_by_frame = _compute_roundtrip_metrics_for_range(
        drive_id,
        frame_start,
        frame_end,
        candidate_gdf,
        raw_frames,
        pose_map,
        calib,
        lidar_stats,
    )
    roundtrip_path = outputs_dir / "roundtrip_metrics.csv"
    pd.DataFrame(roundtrip_rows).to_csv(roundtrip_path, index=False)

    alignment_cfg = merged.get("alignment", {}) if isinstance(merged.get("alignment"), dict) else {}
    offset_min = int(alignment_cfg.get("offset_scan_min", -5))
    offset_max = int(alignment_cfg.get("offset_scan_max", 5))
    offsets = list(range(offset_min, offset_max + 1))
    offset_rows, offset_hist = _scan_offsets_for_range(
        kitti_root,
        drive_id,
        frame_start,
        frame_end,
        offsets,
        raw_frames,
        calib,
        cam_to_pose,
        lidar_cfg_norm,
        index_lookup,
        lidar_world_mode,
        camera,
    )
    offset_path = outputs_dir / "best_offset_summary.csv"
    pd.DataFrame(offset_rows).to_csv(offset_path, index=False)
    hist_path = outputs_dir / "best_offset_hist.csv"
    pd.DataFrame(
        [{"offset": k, "count": v} for k, v in sorted(offset_hist.items(), key=lambda x: x[0])]
    ).to_csv(hist_path, index=False)

    debug_every = int(alignment_cfg.get("debug_every_n", 100))
    debug_frames = {frame_start}
    if roundtrip_rows:
        valid = [r for r in roundtrip_rows if r.get("reproj_iou") is not None]
        if valid:
            min_row = min(valid, key=lambda r: r.get("reproj_iou"))
            max_row = max(valid, key=lambda r: r.get("reproj_iou"))
            debug_frames.update(
                [
                    _parse_frame_id(min_row["frame_id"]) or frame_start,
                    _parse_frame_id(max_row["frame_id"]) or frame_start,
                ]
            )
    for frame in range(frame_start, frame_end + 1):
        if debug_every > 0 and (frame - frame_start) % debug_every == 0:
            debug_frames.add(frame)
    for frame in sorted(debug_frames):
        frame_id = _normalize_frame_id(str(frame))
        image_path = index_lookup.get((drive_id, frame_id), "")
        if not image_path:
            continue
        try:
            points_world = load_kitti360_lidar_points_world(
                kitti_root,
                drive_id,
                frame_id,
                mode=lidar_world_mode,
                cam_id=camera,
            )
        except Exception:
            points_world = np.empty((0, 3), dtype=float)
        debug_path = outputs_dir / f"debug_lidar_proj_{frame_id}.png"
        _render_lidar_proj_debug(image_path, debug_path, points_world, pose_map.get(frame_id), calib)
    review_gdf, final_gdf = _build_review_final_layers(drive_id, candidate_gdf, cluster_info, final_cfg, review_cfg)
    out_gpkg = outputs_dir / "crosswalk_entities_utm32.gpkg"
    _write_crosswalk_gpkg(candidate_gdf, review_gdf, final_gdf, out_gpkg)
    if write_wgs84:
        out_wgs84 = outputs_dir / "crosswalk_entities_wgs84.gpkg"
        if out_wgs84.exists():
            out_wgs84.unlink()
        cand_wgs84 = candidate_gdf.to_crs("EPSG:4326") if not candidate_gdf.empty else gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:4326")
        review_wgs84 = review_gdf.to_crs("EPSG:4326") if not review_gdf.empty else gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:4326")
        final_wgs84 = final_gdf.to_crs("EPSG:4326") if not final_gdf.empty else gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:4326")
        if not _ensure_wgs84_range(cand_wgs84) or not _ensure_wgs84_range(review_wgs84) or not _ensure_wgs84_range(final_wgs84):
            log.error("wgs84 range check failed")
            return 4
        cand_wgs84.to_file(out_wgs84, layer="crosswalk_candidate_poly", driver="GPKG")
        review_wgs84.to_file(out_wgs84, layer="crosswalk_review_poly", driver="GPKG")
        final_wgs84.to_file(out_wgs84, layer="crosswalk_poly", driver="GPKG")

    final_support = _update_qa_index_with_final(qa_out_path, final_gdf)
    _ensure_gated_entities_images(
        qa_out_path,
        outputs_dir,
        candidate_gdf,
        raw_frames,
        lidar_stats,
        final_support,
        index_lookup,
        final_gdf,
        kitti_root,
        lidar_world_mode,
        camera,
        stage2_overlay,
    )

    pose_source_label = "oxts_full" if lidar_world_mode == "fullpose" else "oxts"
    trace_records = _build_trace_records(
        drive_id,
        frame_start,
        frame_end,
        index_lookup,
        raw_stats,
        candidate_gdf,
        final_support,
        cluster_info,
        lidar_stats,
        pose_map,
        calib_ok,
        stage2_stats,
        roundtrip_by_frame,
        lidar_world_mode,
        pose_source_label,
    )
    trace_path = outputs_dir / "crosswalk_trace.csv"
    _build_trace(trace_path, trace_records)

    report_path = outputs_dir / "crosswalk_stage2_report.md"
    quick_report_path = outputs_dir / "crosswalk_quick_report.md"
    runtime_snapshot = {
        "camera": camera,
        "image_dir": str(image_dir or ""),
        "pose_source": pose_source_label,
        "lidar_world_mode": lidar_world_mode,
        "p_rect_key": str(calib.get("p_rect_key", "") if calib else ""),
        "r_rect_key": str(calib.get("r_rect_key", "") if calib else ""),
        "cam_to_pose_key": cam_to_pose_key,
        "cam_to_velo_path": str(kitti_root / "calibration" / "calib_cam_to_velo.txt"),
        "roi_used": "false",
        "roi_strategy": "full_frame",
        "scan_stride": str(stage1_stride),
    }
    _build_report(
        report_path,
        drive_id,
        frame_start,
        frame_end,
        trace_records,
        candidate_gdf,
        review_gdf,
        final_gdf,
        cluster_info,
        outputs_dir,
        final_cfg,
        float(lidar_cfg_norm.get("MIN_IN_IMAGE_RATIO", 0.1)),
        runtime_snapshot,
    )
    _build_report(
        quick_report_path,
        drive_id,
        frame_start,
        frame_end,
        trace_records,
        candidate_gdf,
        review_gdf,
        final_gdf,
        cluster_info,
        outputs_dir,
        final_cfg,
        float(lidar_cfg_norm.get("MIN_IN_IMAGE_RATIO", 0.1)),
        runtime_snapshot,
    )
    alignment_report = outputs_dir / "projection_alignment_report.md"
    _write_projection_alignment_report(
        alignment_report,
        drive_id,
        frame_start,
        frame_end,
        roundtrip_rows,
        offset_rows,
        offset_hist,
        lidar_stats,
        outputs_dir,
    )
    legacy_report = outputs_dir / "crosswalk_refine_report.md"
    if legacy_report != report_path:
        shutil.copy2(report_path, legacy_report)
    _write_cluster_summary(outputs_dir, drive_id, cluster_info, final_gdf, final_cfg)

    miss_frames = [
        r["frame_id"]
        for r in trace_records
        if int(r.get("raw_has_crosswalk", 0)) == 1
        and r.get("raw_status") in {"ok", "on_demand_infer_ok"}
        and int(r.get("candidate_written", 0)) == 0
    ]
    if miss_frames:
        log.error("candidate_missing_count=%d", len(miss_frames))
        log.error("candidate_missing_frames=%s", ",".join(miss_frames[:20]))
        return 6

    if export_all_frames and qa_out_path.exists():
        qa = gpd.read_file(qa_out_path)
        if len(qa) != frame_end - frame_start + 1:
            log.warning("qa_index_count=%d expected=%d", len(qa), frame_end - frame_start + 1)

    log.info("done: %s", outputs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

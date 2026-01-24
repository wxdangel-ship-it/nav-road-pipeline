from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
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
from shapely.geometry import MultiPoint, Point, Polygon, box

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.datasets.kitti360_io import load_kitti360_calib, load_kitti360_lidar_points, load_kitti360_pose
from tools.build_image_sample_index import _extract_frame_id, _find_image_dir, _list_images
from tools.build_road_entities import _geom_to_image_points
from tools.run_crosswalk_monitor_drive import _run_on_demand_infer
from tools.run_image_basemodel import _ensure_cache_env, _resolve_sam2_checkpoint


LOG = logging.getLogger("run_crosswalk_monitor_range")


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("run_crosswalk_monitor_range")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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
def _normalize_frame_id(frame_id: str) -> str:
    digits = "".join(ch for ch in str(frame_id) if ch.isdigit())
    if not digits:
        return str(frame_id)
    return digits.zfill(10)


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
    pose_xy_yaw: Tuple[float, float, float],
    calib: Dict[str, np.ndarray],
) -> np.ndarray:
    x0, y0, yaw = pose_xy_yaw
    c = float(np.cos(-yaw))
    s = float(np.sin(-yaw))
    dx = points[:, 0] - x0
    dy = points[:, 1] - y0
    x_ego = c * dx - s * dy
    y_ego = s * dx + c * dy
    z_ego = points[:, 2]
    ones = np.ones_like(x_ego)
    pts_h = np.stack([x_ego, y_ego, z_ego, ones], axis=0)
    cam = calib["t_velo_to_cam"] @ pts_h
    proj = calib["p_rect"] @ np.vstack([cam[:3, :], np.ones((1, cam.shape[1]))])
    zs = proj[2, :]
    valid = zs > 1e-3
    us = np.zeros_like(zs)
    vs = np.zeros_like(zs)
    us[valid] = proj[0, valid] / zs[valid]
    vs[valid] = proj[1, valid] / zs[valid]
    return np.stack([us, vs, valid], axis=1)


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


def _rect_metrics(rect: Polygon, union_geom: Polygon) -> dict:
    rect_area = rect.area if rect is not None else 0.0
    area = union_geom.area if union_geom is not None else 0.0
    rectangularity = area / rect_area if rect_area > 0 else 0.0
    rect_coords = list(rect.exterior.coords) if rect is not None else []
    edge_lengths = []
    for i in range(len(rect_coords) - 1):
        p0 = rect_coords[i]
        p1 = rect_coords[i + 1]
        edge_lengths.append(float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])))
    edge_lengths = sorted(edge_lengths, reverse=True)
    rect_l = edge_lengths[0] if edge_lengths else 0.0
    rect_w = edge_lengths[-1] if edge_lengths else 0.0
    aspect = rect_l / max(1e-6, rect_w) if rect_w > 0 else 0.0
    return {
        "rect_w_m": rect_w,
        "rect_l_m": rect_l,
        "aspect": aspect,
        "rectangularity": rectangularity,
        "geom_area_m2": area,
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
) -> Tuple[np.ndarray, np.ndarray]:
    if pose is None:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    try:
        pts = load_kitti360_lidar_points(data_root, drive_id, frame_id)
    except Exception:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    if pts.size == 0:
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
    lidar_cfg: dict,
) -> Tuple[dict | None, dict]:
    stats = {
        "proj_method": "none",
        "pose_ok": 1 if pose is not None else 0,
        "calib_ok": 1 if calib is not None else 0,
        "proj_in_image_ratio": 0.0,
        "points_total": 0,
        "points_in_bbox": 0,
        "points_in_mask": 0,
        "mask_dilate_px": int(lidar_cfg.get("MASK_DILATE_PX", 5)),
        "intensity_top_pct": 0,
        "ground_filter_used": 0,
        "dbscan_points": 0,
        "geom_ok": 0,
        "geom_area_m2": 0.0,
        "rect_w_m": 0.0,
        "rect_l_m": 0.0,
        "rectangularity": 0.0,
        "drop_reason_code": "GEOM_INVALID",
        "accum_frames_used": 1,
        "points_accum_total": 0,
        "support_points": np.empty((0, 2), dtype=float),
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

    points_world, intensities = _lidar_points_world_with_intensity(data_root, drive_id, frame_id, pose)
    if points_world.size == 0:
        stats["drop_reason_code"] = "LIDAR_NO_POINTS_BBOX"
        return None, stats

    z_vals = points_world[:, 2]
    z_ground = float(np.percentile(z_vals, 10)) if z_vals.size > 0 else 0.0
    ground_mask = np.abs(z_vals - z_ground) < float(lidar_cfg.get("GROUND_Z_TOL", 0.2))
    if np.any(ground_mask):
        points_world = points_world[ground_mask]
        intensities = intensities[ground_mask]
        stats["ground_filter_used"] = 1
    if points_world.size == 0:
        stats["drop_reason_code"] = "LIDAR_NO_POINTS_BBOX"
        return None, stats

    proj = _project_world_to_image(points_world, pose, calib)
    u = proj[:, 0]
    v = proj[:, 1]
    valid = proj[:, 2].astype(bool)
    width, height = _load_image_size(image_path)
    in_image = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    stats["points_total"] = int(points_world.shape[0])
    stats["proj_in_image_ratio"] = float(np.mean(in_image)) if points_world.shape[0] > 0 else 0.0

    if stats["proj_in_image_ratio"] < float(lidar_cfg.get("MIN_IN_IMAGE_RATIO", 0.1)):
        stats["drop_reason_code"] = "LIDAR_CALIB_MISMATCH"
        return None, stats

    if not (isinstance(bbox_px, (list, tuple)) and len(bbox_px) == 4):
        bounds = raw_gdf.total_bounds if hasattr(raw_gdf, "total_bounds") else None
        if bounds is not None and len(bounds) == 4:
            bbox_px = [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])]
    if not (isinstance(bbox_px, (list, tuple)) and len(bbox_px) == 4):
        stats["drop_reason_code"] = "LIDAR_NO_POINTS_BBOX"
        return None, stats

    minx, miny, maxx, maxy = bbox_px
    in_bbox = in_image & (u >= minx) & (u <= maxx) & (v >= miny) & (v <= maxy)
    stats["points_in_bbox"] = int(np.sum(in_bbox))
    min_points_bbox = int(lidar_cfg.get("MIN_POINTS_BBOX", 20))
    if stats["points_in_bbox"] < min_points_bbox:
        stats["drop_reason_code"] = "LIDAR_NO_POINTS_BBOX"
        return None, stats

    mask_geom = raw_gdf.geometry.union_all() if not raw_gdf.empty else None
    if mask_geom is None or mask_geom.is_empty:
        stats["drop_reason_code"] = "LIDAR_NO_POINTS_MASK"
        return None, stats
    dilate_px = int(lidar_cfg.get("MASK_DILATE_PX", 5))
    if dilate_px > 0:
        mask_geom = mask_geom.buffer(float(dilate_px))
    mask_hits = []
    for idx, ok in enumerate(in_bbox):
        if not ok:
            continue
        if mask_geom.contains(Point(float(u[idx]), float(v[idx]))):
            mask_hits.append(idx)
    stats["points_in_mask"] = int(len(mask_hits))
    intensity_pct = int(lidar_cfg.get("INTENSITY_TOP_PCT", 10))
    stats["intensity_top_pct"] = intensity_pct
    bbox_indices = np.where(in_bbox)[0]
    if bbox_indices.size == 0:
        stats["drop_reason_code"] = "LIDAR_NO_POINTS_BBOX"
        return None, stats
    vals = intensities[bbox_indices]
    thr = np.percentile(vals, 100 - intensity_pct) if vals.size > 0 else None
    support_idx = bbox_indices[vals >= thr].tolist() if thr is not None else bbox_indices.tolist()
    min_points_mask = int(lidar_cfg.get("MIN_POINTS_MASK", 5))
    if len(support_idx) < min_points_mask:
        stats["drop_reason_code"] = "LIDAR_NO_POINTS_MASK"
        return None, stats

    support_pts = points_world[support_idx][:, :2]
    eps = float(lidar_cfg.get("DBSCAN_EPS_M", 0.6))
    min_samples = int(lidar_cfg.get("DBSCAN_MIN_SAMPLES", 10))
    cluster_idx = _dbscan_largest_cluster(support_pts, eps, min_samples)
    if cluster_idx.size > 0:
        support_pts = support_pts[cluster_idx]
        stats["dbscan_points"] = int(cluster_idx.size)
    else:
        stats["dbscan_points"] = int(support_pts.shape[0])
    if support_pts.shape[0] < 3:
        stats["drop_reason_code"] = "GEOM_INVALID"
        return None, stats
    hull = MultiPoint([Point(float(x), float(y)) for x, y in support_pts]).convex_hull
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
    metrics = _rect_metrics(rect, hull)

    stats["proj_method"] = "lidar"
    stats["geom_ok"] = 1
    stats["geom_area_m2"] = float(metrics["geom_area_m2"])
    stats["rect_w_m"] = float(metrics["rect_w_m"])
    stats["rect_l_m"] = float(metrics["rect_l_m"])
    stats["rectangularity"] = float(metrics["rectangularity"])
    stats["drop_reason_code"] = "LIDAR_OK"
    stats["support_points"] = support_pts
    candidate = {
        "geometry": rect,
        "properties": {
            "candidate_id": f"{drive_id}_crosswalk_lidar_{frame_id}",
            "drive_id": drive_id,
            "frame_id": frame_id,
            "entity_type": "crosswalk",
            "reject_reasons": "",
            "proj_method": "lidar",
            "points_in_bbox": stats["points_in_bbox"],
            "points_in_mask": stats["points_in_mask"],
            "mask_dilate_px": stats["mask_dilate_px"],
            "intensity_top_pct": stats["intensity_top_pct"],
            "ground_filter_used": stats["ground_filter_used"],
            "dbscan_points": stats["dbscan_points"],
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
    lidar_cfg: dict,
) -> Tuple[List[dict], Dict[str, dict]]:
    candidates = []
    stats_by_frame: Dict[str, dict] = {}
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
            lidar_cfg,
        )
        stats_by_frame[frame_id] = stats
        if cand:
            candidates.append(cand)
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
    camera: str,
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
                x, y, yaw = load_kitti360_pose(kitti_root, drive_id, frame_id)
                pose_map[frame_id] = (x, y, yaw)
            except Exception:
                pose_map[frame_id] = None
        candidates = candidate_by_frame.get(frame_id, gpd.GeoDataFrame())
        kept, _ = _render_gated_overlay(
            gated_path,
            image_path,
            frame_id,
            candidates,
            raw_frames.get((drive_id, frame_id)),
            lidar_stats.get(frame_id, {}),
            pose_map.get(frame_id),
            calib,
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
        x, y, yaw = load_kitti360_pose(kitti_root, drive_id, frame_id)
    except Exception:
        _render_text_overlay(image_path, out_path, ["NO_POSE"])
        return
    pose = (x, y, yaw)
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
        drop_reason = str(diag.get("drop_reason_code") or "")
        draw.text((10, 60), f"{proj_method} bbox={points_in_bbox} mask={points_in_mask}", fill=(200, 200, 200, 220))
        if drop_reason:
            draw.text((10, 86), drop_reason, fill=(200, 120, 120, 220))
        if proj_method in {"plane", "bbox_only"}:
            draw.text((10, 112), "WEAK_PROJ", fill=(255, 128, 0, 220))
    if stage2_added:
        draw.text((10, 138), "STAGE2_ADDED", fill=(0, 200, 255, 220))

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
            rect = geom.minimum_rotated_rectangle
            metrics = _rect_metrics(rect, geom)
            rect_ws.append(metrics["rect_w_m"])
            rect_ls.append(metrics["rect_l_m"])
            rectangularities.append(metrics["rectangularity"])
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
            metrics = _rect_metrics(rect, union_geom)
            cluster_info[cid]["refined_geom"] = rect
            cluster_info[cid]["rect_w_m"] = float(metrics["rect_w_m"])
            cluster_info[cid]["rect_l_m"] = float(metrics["rect_l_m"])
            cluster_info[cid]["rectangularity"] = float(metrics["rectangularity"])
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
            metrics = _rect_metrics(rect, union_geom)
            info["refined_geom"] = rect
            info["rect_w_m"] = float(metrics["rect_w_m"])
            info["rect_l_m"] = float(metrics["rect_l_m"])
            info["rectangularity"] = float(metrics["rectangularity"])
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
        metrics = _rect_metrics(rect, hull)
        info["refined_geom"] = rect
        info["rect_w_m"] = float(metrics["rect_w_m"])
        info["rect_l_m"] = float(metrics["rect_l_m"])
        info["rectangularity"] = float(metrics["rectangularity"])
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
                lidar_cfg,
            )
            if cand is None:
                reason = str(stats.get("drop_reason_code") or "geom_invalid")
                stat["reasons"].append(reason)
                continue
            cand["properties"]["candidate_id"] = f"{drive_id}_crosswalk_stage2_{cid}_{frame_id}"
            cand["properties"]["cluster_id"] = cid
            cand["properties"]["stage2_added"] = 1
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
    pose: Tuple[float, float, float] | None,
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
        x, y, yaw = pose
        reject_extra = ""
        proj_method = "plane" if use_plane else "bbox_only"
        plane_ok = 1 if use_plane else 0
    half = max(0.5, size_m / 2.0)
    geom = box(x - half, y - half, x + half, y + half)
    if proj_method == "plane":
        geom = affinity.rotate(geom, np.degrees(yaw), origin=(x, y))
    return {
        "geometry": geom,
        "properties": {
            "candidate_id": f"{drive_id}_crosswalk_fallback_{frame_id}",
            "drive_id": drive_id,
            "frame_id": frame_id,
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
        cand_ids = sorted(set(candidate_ids.get(key, [])))
        rejects = sorted(set(candidate_rejects.get(key, [])))
        final_ids = sorted(set(final_support.get(key, [])))
        stage2 = stage2_stats.get(key, {"attempted": 0, "added": 0, "reasons": []})
        candidates = candidate_by_frame.get(frame_id, gpd.GeoDataFrame())
        candidate_count = int(len(candidates)) if not candidates.empty else 0
        kept = 0
        geom_ok = 0
        geom_area = 0.0
        proj_method = str(lidar_info.get("proj_method") or "none")
        if not candidates.empty:
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
                "raw_status": raw_info.get("raw_status", "unknown"),
                "raw_has_crosswalk": int(raw_info.get("raw_has_crosswalk", 0)),
                "raw_top_score": raw_info.get("raw_top_score", 0.0),
                "pose_ok": pose_ok,
                "calib_ok": 1 if calib_ok else 0,
                "proj_method": proj_method,
                "proj_in_image_ratio": float(lidar_info.get("proj_in_image_ratio", 0.0)),
                "points_in_bbox": int(lidar_info.get("points_in_bbox", 0)),
                "points_in_mask": int(lidar_info.get("points_in_mask", 0)),
                "mask_dilate_px": int(lidar_info.get("mask_dilate_px", 0)),
                "intensity_top_pct": int(lidar_info.get("intensity_top_pct", 0)),
                "ground_filter_used": int(lidar_info.get("ground_filter_used", 0)),
                "dbscan_points": int(lidar_info.get("dbscan_points", 0)),
                "geom_ok": geom_ok if geom_ok else int(lidar_info.get("geom_ok", 0)),
                "geom_area_m2": geom_area if geom_area else float(lidar_info.get("geom_area_m2", 0.0)),
                "rect_w_m": rect_w_m,
                "rect_l_m": rect_l_m,
                "rectangularity": rectangularity,
                "cluster_id": cluster_id,
                "cluster_frames_hit": cluster_frames_hit,
                "jitter_p90": jitter_p90,
                "angle_jitter_p90": angle_jitter_p90,
                "support_flag": support_flag,
                "stage2_roi_attempted": int(stage2.get("attempted", 0)),
                "stage2_roi_added": int(stage2.get("added", 0)),
                "stage2_roi_reason": "|".join(sorted(set(stage2.get("reasons", [])))),
                "frames_hit_support_after": cluster_frames_hit,
                "candidate_written": 1 if cand_ids else 0,
                "candidate_count": candidate_count,
                "reject_reasons": "|".join(rejects),
                "final_support": 1 if final_ids else 0,
                "final_entity_id": "|".join(final_ids),
                "drop_reason_code": drop_reason,
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
) -> None:
    lines = []
    lines.append("# Crosswalk Refine Report\n")
    lines.append(f"- report_time: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"- drive_id: {drive_id}")
    lines.append(f"- frame_range: {frame_start}-{frame_end}")
    lines.append(f"- total_frames: {frame_end - frame_start + 1}")
    lines.append("")
    n_pos = len([r for r in trace_records if int(r.get("raw_has_crosswalk", 0)) == 1 and r.get("raw_status") in {"ok", "on_demand_infer_ok"}])
    n_miss = len([r for r in trace_records if int(r.get("raw_has_crosswalk", 0)) == 1 and r.get("raw_status") in {"ok", "on_demand_infer_ok"} and int(r.get("candidate_written", 0)) == 0])
    lines.append(f"- N_pos: {n_pos}")
    lines.append(f"- N_miss: {n_miss}")
    lines.append(f"- candidate_count_total: {len(candidate_gdf)}")
    lines.append(f"- review_count: {len(review_gdf)}")
    lines.append(f"- final_count: {len(final_gdf)}")
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
    if reject_counts:
        lines.append("## Reject Reason Summary")
        for reason, count in sorted(reject_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
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
        candidates = []
        for cid, info in cluster_info.items():
            if info.get("frames_hit_support", info.get("frames_hit", 0)) < 2:
                continue
            if cid in set(review_gdf.get("cluster_id", [])) or cid in set(final_gdf.get("cluster_id", [])):
                candidates.append((info.get("frames_hit_support", info.get("frames_hit", 0)), cid, info))
        candidates.sort(reverse=True)
        for frames_hit, cid, info in candidates[:5]:
            reasons = []
            if info.get("frames_hit_support", info.get("frames_hit", 0)) < 3:
                reasons.append("frames_hit_support<3")
            if info.get("rectangularity", 0.0) < 0.45:
                reasons.append("rectangularity")
            if info.get("rect_w_m", 0.0) < 1.5 or info.get("rect_w_m", 0.0) > 30.0:
                reasons.append("rect_w")
            if info.get("rect_l_m", 0.0) < 3.0 or info.get("rect_l_m", 0.0) > 40.0:
                reasons.append("rect_l")
            if info.get("heading_diff") is not None and info.get("heading_diff", 0.0) > 25.0:
                reasons.append("heading_diff")
            if info.get("jitter_p90", 0.0) > 8.0:
                reasons.append("jitter")
            lines.append(f"- {cid} frames_hit={frames_hit} reasons={','.join(reasons) or 'n/a'}")
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
    lines.append(f"- outputs_dir: {outputs_dir}")
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
        gore_like_ratio = float(info.get("gore_like_ratio", 0.0))
        support_frames = info.get("support_frames", [])
        refined_rect_w = float(info.get("rect_w_m", 0.0))
        refined_rect_l = float(info.get("rect_l_m", 0.0))
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
            f"- {row['cluster_id']}: frames_hit_all={row['frames_hit_all']} frames_hit_support={row['frames_hit_support']} stage2_added={row['stage2_added_frames_count']} gore_like_ratio={row['gore_like_ratio']:.2f} final_pass={row['final_pass']} reason={row['final_fail_reason']}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crosswalk_fix_range.yaml")
    ap.add_argument("--drive", default=None)
    ap.add_argument("--frame-start", type=int, default=None)
    ap.add_argument("--frame-end", type=int, default=None)
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
        },
    )

    drive_id = str(merged.get("drive_id") or "")
    frame_start = int(merged.get("frame_start", 0))
    frame_end = int(merged.get("frame_end", 0))
    kitti_root = Path(str(merged.get("kitti_root") or ""))
    camera = str(merged.get("camera") or "image_00")
    stage1_stride = int(merged.get("stage1_stride", 1))
    export_all_frames = bool(merged.get("export_all_frames", True))
    write_wgs84 = bool(merged.get("write_wgs84", True))
    raw_fallback_text = bool(merged.get("raw_fallback_text", True))
    on_demand_infer = bool(merged.get("on_demand_infer", False))
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

    if not drive_id or frame_end < frame_start:
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

    index_records = _build_index_records(kitti_root, drive_id, frame_start, frame_end, camera)
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
    ]
    log.info("stage1: %s", " ".join(cmd))
    if subprocess.run(cmd, check=False).returncode != 0:
        log.error("stage1 failed")
        return 3

    stage_outputs = stage_dir / "outputs"
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

    stage_gpkg = stage_outputs / "road_entities_utm32.gpkg"
    candidate_gdf = _read_candidates(stage_gpkg)
    pose_map: Dict[str, Tuple[float, float, float] | None] = {}
    for frame in range(frame_start, frame_end + 1):
        frame_id = _normalize_frame_id(str(frame))
        try:
            x, y, yaw = load_kitti360_pose(kitti_root, drive_id, frame_id)
            pose_map[frame_id] = (x, y, yaw)
        except Exception:
            pose_map[frame_id] = None
    try:
        calib = load_kitti360_calib(kitti_root, camera)
        calib_ok = True
    except Exception:
        calib = None
        calib_ok = False
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
        lidar_cfg_norm,
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
    stage2_cfg = merged.get("stage2", {}) if isinstance(merged.get("stage2"), dict) else {}
    candidate_gdf, stage2_stats = _stage2_roi_refine(
        candidate_gdf,
        cluster_info,
        drive_id,
        frame_start,
        frame_end,
        index_lookup,
        pose_map,
        calib,
        lidar_cfg_norm,
        image_provider,
        kitti_root,
        stage2_cfg,
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
    cluster_info = _refresh_cluster_members(candidate_gdf, cluster_info)
    candidate_gdf, cluster_info = _refine_clusters(candidate_gdf, cluster_info, pose_map, lidar_stats, raw_stats, drive_id, refine_cfg)
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
        camera,
    )

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
    )
    trace_path = outputs_dir / "crosswalk_trace.csv"
    _build_trace(trace_path, trace_records)

    report_path = outputs_dir / "crosswalk_refine_report.md"
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
    )
    legacy_report = outputs_dir / "crosswalk_fix_report.md"
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

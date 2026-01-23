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


LOG = logging.getLogger("run_crosswalk_monitor_range")


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("run_crosswalk_monitor_range")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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
        "geom_ok": 0,
        "geom_area_m2": 0.0,
        "drop_reason_code": "GEOM_INVALID",
        "accum_frames_used": 1,
        "points_accum_total": 0,
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
    min_points_mask = int(lidar_cfg.get("MIN_POINTS_MASK", 5))
    support_idx = mask_hits
    if stats["points_in_mask"] < min_points_mask:
        intensity_pct = int(lidar_cfg.get("INTENSITY_TOP_PCT", 20))
        stats["intensity_top_pct"] = intensity_pct
        bbox_indices = np.where(in_bbox)[0]
        if bbox_indices.size > 0:
            vals = intensities[bbox_indices]
            thr = np.percentile(vals, 100 - intensity_pct) if vals.size > 0 else None
            if thr is not None:
                support_idx = bbox_indices[vals >= thr].tolist()
        if len(support_idx) < min_points_mask:
            stats["drop_reason_code"] = "LIDAR_NO_POINTS_MASK"
            return None, stats

    support_pts = points_world[support_idx][:, :2]
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

    stats["proj_method"] = "lidar"
    stats["geom_ok"] = 1
    stats["geom_area_m2"] = float(rect.area)
    stats["drop_reason_code"] = "LIDAR_OK"
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
            "drop_reason_code": stats["drop_reason_code"],
            "geom_ok": stats["geom_ok"],
            "geom_area_m2": stats["geom_area_m2"],
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
    for idx, row in qa_gdf.iterrows():
        drive_id = str(row.get("drive_id") or "")
        frame_id = _normalize_frame_id(str(row.get("frame_id") or ""))
        if not drive_id or not frame_id:
            continue
        image_path = index_lookup.get((drive_id, frame_id), "")
        out_path = qa_dir / drive_id / f"{frame_id}_overlay_raw.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_gdf, raw_score, raw_status = _load_crosswalk_raw(feature_store_root, drive_id, frame_id, raw_cache)
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
        if raw_status == "missing_feature_store":
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


def _write_crosswalk_gpkg(candidate_gdf: gpd.GeoDataFrame, final_gdf: gpd.GeoDataFrame, out_gpkg: Path) -> None:
    if out_gpkg.exists():
        out_gpkg.unlink()
    cand = candidate_gdf if not candidate_gdf.empty else gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:32632")
    final = final_gdf if not final_gdf.empty else gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:32632")
    cand.to_file(out_gpkg, layer="crosswalk_candidate_poly", driver="GPKG")
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
    diag = lidar_info or {}
    for _, row in candidates.iterrows():
        geom = row.geometry
        reasons = str(row.get("reject_reasons") or "")
        is_rejected = bool(reasons)
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
        use_plane = bool(calib_ok and pose_ok and score >= 0.3)
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
    feature_store_map_root: Path,
    fallback_frames: set[str],
    pose_map: Dict[str, Tuple[float, float, float] | None],
    calib_ok: bool,
    lidar_stats: Dict[str, dict],
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
        map_path = feature_store_map_root / drive_id / frame_id / "map_evidence_utm32.gpkg"
        project_ok = 1 if map_path.exists() else 0
        cand_ids = sorted(set(candidate_ids.get(key, [])))
        rejects = sorted(set(candidate_rejects.get(key, [])))
        final_ids = sorted(set(final_support.get(key, [])))
        candidates = candidate_by_frame.get(frame_id, gpd.GeoDataFrame())
        candidate_count = int(len(candidates)) if not candidates.empty else 0
        kept = 0
        geom_ok = 0
        geom_area = 0.0
        proj_method = str(lidar_info.get("proj_method") or "none")
        plane_ok = int(lidar_info.get("plane_ok") or 0)
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
                if "proj_method" in row and pd.notna(row.get("proj_method")):
                    val = str(row.get("proj_method"))
                    if val:
                        proj_method = val
                if "plane_ok" in row and pd.notna(row.get("plane_ok")):
                    try:
                        plane_ok = max(plane_ok, int(row.get("plane_ok")))
                    except Exception:
                        pass
        pose_ok = 1 if pose_map.get(frame_id) is not None else 0
        if frame_id in fallback_frames and proj_method == "none":
            proj_method = "plane" if calib_ok and pose_ok else "bbox_only"
            plane_ok = 1 if proj_method == "plane" else 0
            geom_ok = 0 if proj_method == "bbox_only" else geom_ok
        proj_points = int(lidar_info.get("points_in_mask", -1))
        if proj_method == "lidar" and proj_points < 0:
            proj_points = 0 if project_ok == 0 else 1
        drop_reason = ""
        if int(raw_info.get("raw_has_crosswalk", 0)) == 0:
            drop_reason = "RAW_EMPTY"
        elif lidar_info.get("drop_reason_code"):
            drop_reason = str(lidar_info.get("drop_reason_code"))
        elif not cand_ids:
            if pose_ok == 0:
                drop_reason = "POSE_MISSING"
            elif not calib_ok:
                drop_reason = "CALIB_MISSING"
            elif proj_method == "plane" and plane_ok == 0:
                drop_reason = "PLANE_FAIL"
            elif proj_method == "bbox_only":
                drop_reason = "GEOM_INVALID"
            elif project_ok == 0:
                drop_reason = "LIDAR_NO_POINTS_BBOX"
            else:
                drop_reason = "WRITE_CANDIDATE_FAIL"
        if proj_method == "plane" and drop_reason not in {"PLANE_FAIL", "RAW_EMPTY"}:
            drop_reason = "PLANE_FALLBACK_USED"
        records.append(
            {
                "drive_id": drive_id,
                "frame_id": frame_id,
                "image_path": index_lookup.get(key, ""),
                "raw_status": raw_info.get("raw_status", "unknown"),
                "raw_has_crosswalk": int(raw_info.get("raw_has_crosswalk", 0)),
                "raw_top_score": raw_info.get("raw_top_score", 0.0),
                "project_ok": project_ok,
                "pose_ok": pose_ok,
                "calib_ok": 1 if calib_ok else 0,
                "proj_method": proj_method,
                "proj_in_image_ratio": float(lidar_info.get("proj_in_image_ratio", 0.0)),
                "points_total": int(lidar_info.get("points_total", 0)),
                "points_in_bbox": int(lidar_info.get("points_in_bbox", 0)),
                "points_in_mask": int(lidar_info.get("points_in_mask", 0)),
                "mask_dilate_px": int(lidar_info.get("mask_dilate_px", 0)),
                "intensity_top_pct": int(lidar_info.get("intensity_top_pct", 0)),
                "plane_ok": plane_ok,
                "geom_ok": geom_ok if geom_ok else int(lidar_info.get("geom_ok", 0)),
                "geom_area_m2": geom_area if geom_area else float(lidar_info.get("geom_area_m2", 0.0)),
                "candidate_written": 1 if cand_ids else 0,
                "candidate_count": candidate_count,
                "candidate_id": "|".join(cand_ids),
                "reject_reasons": "|".join(rejects),
                "gated_kept": 1 if kept > 0 else 0,
                "final_support": 1 if final_ids else 0,
                "final_entity_ids": "|".join(final_ids),
                "drop_reason_code": drop_reason,
                "accum_frames_used": int(lidar_info.get("accum_frames_used", 1)),
                "points_accum_total": int(lidar_info.get("points_accum_total", 0)),
            }
        )
    return records


def _build_report(
    out_path: Path,
    drive_id: str,
    frame_start: int,
    frame_end: int,
    trace_records: List[dict],
    final_gdf: gpd.GeoDataFrame,
    top_reject: List[dict],
    outputs_dir: Path,
) -> None:
    lines = []
    lines.append("# Crosswalk Monitor Report\n")
    lines.append(f"- report_time: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"- drive_id: {drive_id}")
    lines.append(f"- frame_range: {frame_start}-{frame_end}")
    lines.append(f"- total_frames: {frame_end - frame_start + 1}")
    lines.append("")
    raw_hits = [r["frame_id"] for r in trace_records if int(r.get("raw_has_crosswalk", 0)) == 1]
    lines.append(f"- raw_has_crosswalk_count: {len(raw_hits)}")
    lines.append(f"- raw_has_crosswalk_frames: {', '.join(raw_hits)}")
    n_pos = len(
        [
            r
            for r in trace_records
            if int(r.get("raw_has_crosswalk", 0)) == 1 and r.get("raw_status") in {"ok", "on_demand_infer_ok"}
        ]
    )
    n_miss = len(
        [
            r
            for r in trace_records
            if int(r.get("raw_has_crosswalk", 0)) == 1
            and r.get("raw_status") in {"ok", "on_demand_infer_ok"}
            and int(r.get("candidate_written", 0)) == 0
        ]
    )
    lines.append(f"- N_pos: {n_pos}")
    lines.append(f"- N_miss: {n_miss}")
    lines.append("")
    lidar_ok = [
        r
        for r in trace_records
        if r.get("proj_method") == "lidar"
        and int(r.get("geom_ok", 0)) == 1
        and int(r.get("points_in_bbox", 0)) >= 20
    ]
    lines.append(f"- lidar_ok_count: {len(lidar_ok)}")
    lines.append("")
    cand_missing = [r for r in trace_records if int(r.get("candidate_written", 0)) == 0]
    lines.append(f"- candidate_written_zero: {len(cand_missing)}")
    drop_counts: Dict[str, int] = {}
    for row in trace_records:
        reason = str(row.get("drop_reason_code") or "")
        if reason:
            drop_counts[reason] = drop_counts.get(reason, 0) + 1
    if drop_counts:
        lines.append("## Drop Reason Summary")
        for reason, count in sorted(drop_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {reason}: {count}")
        lines.append("")
    proj_counts: Dict[str, int] = {}
    for row in trace_records:
        method = str(row.get("proj_method") or "none")
        proj_counts[method] = proj_counts.get(method, 0) + 1
    if proj_counts:
        lines.append("## Projection Method Summary")
        for method, count in sorted(proj_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {method}: {count}")
        lines.append("")
    subset = [r for r in trace_records if int(r.get("raw_has_crosswalk", 0)) == 1]
    pts_bbox = [int(r.get("points_in_bbox", 0)) for r in subset if int(r.get("points_in_bbox", -1)) >= 0]
    pts_mask = [int(r.get("points_in_mask", 0)) for r in subset if int(r.get("points_in_mask", -1)) >= 0]
    ratios = [float(r.get("proj_in_image_ratio", 0.0)) for r in subset]
    if pts_bbox:
        lines.append("## points_in_bbox Summary")
        lines.append(f"- p50: {np.percentile(pts_bbox, 50):.1f}")
        lines.append(f"- p90: {np.percentile(pts_bbox, 90):.1f}")
        lines.append("")
    if pts_mask:
        lines.append("## points_in_mask Summary")
        lines.append(f"- p50: {np.percentile(pts_mask, 50):.1f}")
        lines.append(f"- p90: {np.percentile(pts_mask, 90):.1f}")
        lines.append("")
    if ratios:
        lines.append("## proj_in_image_ratio Summary")
        lines.append(f"- p50: {np.percentile(ratios, 50):.3f}")
        lines.append(f"- p90: {np.percentile(ratios, 90):.3f}")
        lines.append("")
    reject_counts: Dict[str, int] = {}
    for row in trace_records:
        for reason in [r for r in str(row.get("reject_reasons") or "").split("|") if r]:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
    if reject_counts:
        lines.append("## Reject Reason Summary")
        for reason, count in sorted(reject_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {reason}: {count}")
        lines.append("")
    lines.append("## Raw Has But Final Missing Samples")
    miss_final = [
        r for r in trace_records if int(r.get("raw_has_crosswalk", 0)) == 1 and int(r.get("final_support", 0)) == 0
    ]
    for row in miss_final[:10]:
        frame_id = row.get("frame_id")
        qa_dir = outputs_dir / "qa_images" / drive_id
        lines.append(
            f"- {frame_id} raw={qa_dir / f'{frame_id}_overlay_raw.png'} gated={qa_dir / f'{frame_id}_overlay_gated.png'} entities={qa_dir / f'{frame_id}_overlay_entities.png'}"
        )
    if not miss_final:
        lines.append("- none")
    lines.append("")
    lines.append("## Final Summary")
    lines.append(f"- final_count: {len(final_gdf)}")
    if not final_gdf.empty:
        frames = [int(v) for v in final_gdf["frames_hit"].tolist() if v]
        if frames:
            lines.append(f"- frames_hit_p50: {np.percentile(frames, 50):.1f}")
            lines.append(f"- frames_hit_p90: {np.percentile(frames, 90):.1f}")
        lines.append(f"- final_entity_ids: {', '.join(final_gdf['entity_id'].astype(str).tolist())}")
    lines.append("")
    lines.append("## Top Reject Samples")
    if not top_reject:
        lines.append("- none")
    else:
        for row in top_reject[:10]:
            lines.append(
                f"- {row.get('drive_id')}:{row.get('frame_id')} score={row.get('score_total'):.3f} reject={row.get('reject_reasons')}"
            )
    lines.append("")
    lines.append(f"- outputs_dir: {outputs_dir}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crosswalk_monitor.yaml")
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
    min_frames_hit_final = int(merged.get("min_frames_hit_final", 3))
    write_wgs84 = bool(merged.get("write_wgs84", True))
    raw_fallback_text = bool(merged.get("raw_fallback_text", True))
    image_run = Path(str(merged.get("image_run") or ""))
    image_provider = str(merged.get("image_provider") or "grounded_sam2_v1")
    image_evidence_gpkg = str(merged.get("image_evidence_gpkg") or "")
    road_root = Path(str(merged.get("road_root") or ""))
    base_config = Path(str(merged.get("config") or "configs/road_entities.yaml"))

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
        log.warning("stage1_stride=%s ignored; monitor runs full coverage.", stage1_stride)

    run_dir = Path(args.out_run) if args.out_run else Path("runs") / f"crosswalk_monitor_{drive_id}_{frame_start}_{frame_end}_{dt.datetime.now():%Y%m%d_%H%M%S}"
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

    stage_cfg = debug_dir / "crosswalk_monitor.yaml"
    stage_cfg.write_text(
        yaml.safe_dump(
            _merge_config(_load_yaml(base_config), {"crosswalk_final": {"min_frames_hit": min_frames_hit_final}}),
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
    )

    stage_gpkg = stage_outputs / "road_entities_utm32.gpkg"
    candidate_gdf = _read_candidates(stage_gpkg)
    final_gdf = _read_final(stage_gpkg)
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
    lidar_cfg = merged.get("lidar_proj", {}) if isinstance(merged.get("lidar_proj"), dict) else {}
    lidar_candidates, lidar_stats = _build_lidar_candidates_for_range(
        kitti_root,
        drive_id,
        frame_start,
        frame_end,
        index_lookup,
        raw_frames,
        pose_map,
        calib,
        lidar_cfg,
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
    )
    out_gpkg = outputs_dir / "crosswalk_entities_utm32.gpkg"
    _write_crosswalk_gpkg(candidate_gdf, final_gdf, out_gpkg)
    if write_wgs84:
        out_wgs84 = outputs_dir / "crosswalk_entities_wgs84.gpkg"
        if out_wgs84.exists():
            out_wgs84.unlink()
        cand_wgs84 = candidate_gdf.to_crs("EPSG:4326") if not candidate_gdf.empty else gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:4326")
        final_wgs84 = final_gdf.to_crs("EPSG:4326") if not final_gdf.empty else gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:4326")
        if not _ensure_wgs84_range(cand_wgs84) or not _ensure_wgs84_range(final_wgs84):
            log.error("wgs84 range check failed")
            return 4
        cand_wgs84.to_file(out_wgs84, layer="crosswalk_candidate_poly", driver="GPKG")
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

    feature_store_map_root = image_run / f"feature_store_map_{image_provider}"
    trace_records = _build_trace_records(
        drive_id,
        frame_start,
        frame_end,
        index_lookup,
        raw_stats,
        candidate_gdf,
        final_support,
        feature_store_map_root,
        fallback_frames,
        pose_map,
        calib_ok,
        lidar_stats,
    )
    trace_path = outputs_dir / "crosswalk_trace.csv"
    _build_trace(trace_path, trace_records)

    top_reject = []
    if not candidate_gdf.empty and "score_total" in candidate_gdf.columns:
        top_reject = (
            candidate_gdf.sort_values("score_total", ascending=False)
            .head(10)
            .to_dict(orient="records")
        )
    report_path = outputs_dir / "crosswalk_monitor_report.md"
    _build_report(
        report_path,
        drive_id,
        frame_start,
        frame_end,
        trace_records,
        final_gdf,
        top_reject,
        outputs_dir,
    )

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

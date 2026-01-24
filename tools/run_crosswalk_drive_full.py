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

from pipeline.datasets.kitti360_io import (
    load_kitti360_calib,
    load_kitti360_cam_to_pose_key,
    load_kitti360_lidar_points,
    load_kitti360_lidar_points_world,
    load_kitti360_pose,
    load_kitti360_pose_full,
)
from tools.build_image_sample_index import _extract_frame_id, _find_image_dir, _list_images


LOG = logging.getLogger("run_crosswalk_drive_full")


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("run_crosswalk_drive_full")


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
    camera: str,
    stride: int,
) -> List[dict]:
    image_dir = _find_image_dir(kitti_root, drive_id, camera)
    if not image_dir:
        return []
    images = _list_images(image_dir)
    if stride and stride > 1:
        images = images[::stride]
    records: List[dict] = []
    for path in images:
        frame_id = _normalize_frame_id(_extract_frame_id(path))
        records.append(
            {
                "drive_id": drive_id,
                "camera": camera,
                "frame_id": frame_id,
                "image_path": str(path),
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
    pose_xy_yaw: Tuple[float, ...],
    calib: Dict[str, np.ndarray],
) -> np.ndarray:
    if len(pose_xy_yaw) == 6:
        x0, y0, z0, roll, pitch, yaw = pose_xy_yaw
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
        x0, y0, yaw = pose_xy_yaw
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        dx = points[:, 0] - x0
        dy = points[:, 1] - y0
        x_ego = c * dx - s * dy
        y_ego = s * dx + c * dy
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


def _geom_to_image_points(
    geom: object,
    pose_xy_yaw: Tuple[float, ...],
    calib: Dict[str, np.ndarray],
) -> List[Tuple[float, float]]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        coords = np.array(list(geom.exterior.coords), dtype=float)
    elif geom.geom_type == "LineString":
        coords = np.array(list(geom.coords), dtype=float)
    else:
        return []
    if coords.shape[0] == 0:
        return []
    points = np.column_stack([coords[:, 0], coords[:, 1], np.zeros(coords.shape[0], dtype=float)])
    proj = _project_world_to_image(points, pose_xy_yaw, calib)
    out: List[Tuple[float, float]] = []
    for u, v, valid in proj:
        if not valid:
            continue
        out.append((float(u), float(v)))
    return out


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
    pose: Tuple[float, ...] | None,
    lidar_world_mode: str,
    cam_id: str,
) -> Tuple[np.ndarray, np.ndarray]:
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
        except Exception:
            return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
        return world, pts[:, 3].astype(float)
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

    points_world, intensities = _lidar_points_world_with_intensity(
        data_root,
        drive_id,
        frame_id,
        pose,
        lidar_cfg.get("lidar_world_mode", "fullpose"),
        str(lidar_cfg.get("cam_id", "image_00")),
    )
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
    frames_to_render: set[str],
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
        if not frames_to_render or frame_id in frames_to_render:
            _render_raw_overlay(image_path, raw_gdf, out_path, raw_status, raw_fallback_text)
            qa_gdf.at[idx, "overlay_raw_path"] = str(out_path)
        else:
            qa_gdf.at[idx, "overlay_raw_path"] = ""
    for col in qa_gdf.columns:
        if col == "geometry":
            continue
        qa_gdf[col] = qa_gdf[col].apply(lambda v: v.tolist() if isinstance(v, np.ndarray) else v)
    qa_gdf.to_file(qa_index_path, driver="GeoJSON")
    missing_path = outputs_dir / "missing_feature_store_list.csv"
    pd.DataFrame(
        missing_rows,
        columns=["drive_id", "frame_id", "raw_status", "image_path"],
    ).to_csv(missing_path, index=False)
    return raw_stats, raw_frames


def _build_lidar_candidates_for_records(
    data_root: Path,
    drive_id: str,
    records: List[dict],
    index_lookup: Dict[Tuple[str, str], str],
    raw_frames: Dict[Tuple[str, str], dict],
    pose_map: Dict[str, Tuple[float, float, float] | None],
    calib: Dict[str, np.ndarray] | None,
    lidar_cfg: dict,
) -> Tuple[List[dict], Dict[str, dict]]:
    candidates = []
    stats_by_frame: Dict[str, dict] = {}
    for record in records:
        frame_id = _normalize_frame_id(record.get("frame_id") or "")
        if not frame_id:
            continue
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


def _write_crosswalk_gpkg(candidate_gdf: gpd.GeoDataFrame, review_gdf: gpd.GeoDataFrame, final_gdf: gpd.GeoDataFrame, out_gpkg: Path) -> None:
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
    frames_to_render: set[str],
) -> None:
    if not qa_index_path.exists():
        return
    qa = gpd.read_file(qa_index_path)
    pose_map: Dict[str, Tuple[float, float, float] | Tuple[float, float, float, float, float, float]] = {}
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
        if frames_to_render and frame_id not in frames_to_render:
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
            for n in neighbors:
                used[n] = True
                queue.append(int(n))
        clusters.append(members)
    return clusters


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
        "area_m2": area,
        "rect_area_m2": rect_area,
        "rectangularity": rectangularity,
        "rect_l_m": rect_l,
        "rect_w_m": rect_w,
        "aspect": aspect,
    }


def _build_clusters(
    candidate_gdf: gpd.GeoDataFrame,
    cluster_eps_m: float,
) -> Tuple[gpd.GeoDataFrame, Dict[str, dict]]:
    if candidate_gdf.empty:
        return candidate_gdf, {}
    gdf = candidate_gdf.copy()
    gdf["frame_id_norm"] = gdf["frame_id"].apply(_normalize_frame_id)
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]
    geoms = gdf.geometry.tolist()
    clusters = _cluster_by_centroid(geoms, cluster_eps_m)
    cluster_info: Dict[str, dict] = {}
    cluster_ids = ["" for _ in range(len(gdf))]
    for idx, members in enumerate(clusters):
        cid = f"cluster_{idx:04d}"
        frames = set()
        centroids = []
        rect_ws = []
        rect_ls = []
        rectangularities = []
        headings = []
        for member in members:
            row = gdf.iloc[member]
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
        if centroids:
            xs = [c[0] for c in centroids]
            ys = [c[1] for c in centroids]
            med_x = float(np.median(xs))
            med_y = float(np.median(ys))
            dists = [np.hypot(x - med_x, y - med_y) for x, y in centroids]
            jitter_p90 = float(np.percentile(dists, 90)) if dists else 0.0
        else:
            jitter_p90 = 0.0
        median_w = float(np.median(rect_ws)) if rect_ws else 0.0
        median_l = float(np.median(rect_ls)) if rect_ls else 0.0
        median_rectangularity = float(np.median(rectangularities)) if rectangularities else 0.0
        median_heading = float(np.median(headings)) if headings else None
        cluster_info[cid] = {
            "frames_hit": len(frames),
            "jitter_p90": jitter_p90,
            "rect_w_m": median_w,
            "rect_l_m": median_l,
            "rectangularity": median_rectangularity,
            "heading_diff": median_heading,
            "member_indices": members,
        }
        for member in members:
            cluster_ids[member] = cid
    gdf["cluster_id"] = cluster_ids
    return gdf, cluster_info


def _build_review_layer(
    candidate_gdf: gpd.GeoDataFrame,
    cluster_info: Dict[str, dict],
    review_cfg: dict,
    drive_id: str,
) -> gpd.GeoDataFrame:
    if candidate_gdf.empty or not cluster_info:
        return gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:32632")
    min_frames = int(review_cfg.get("min_frames_hit_review", 2))
    rect_min = float(review_cfg.get("rectangularity_min_review", 0.35))
    heading_max = float(review_cfg.get("heading_diff_to_perp_max_deg_review", 35.0))
    keep = []
    for cid, info in cluster_info.items():
        if info["frames_hit"] < min_frames:
            continue
        rect_ok = float(info.get("rectangularity") or 0.0) >= rect_min
        heading_ok = True
        if info["heading_diff"] is not None:
            heading_ok = info["heading_diff"] <= heading_max
        if not rect_ok or not heading_ok:
            continue
        members = info["member_indices"]
        subset = candidate_gdf.iloc[members]
        geom = subset.geometry.iloc[0]
        keep.append(
            {
                "geometry": geom,
                "properties": {
                    "entity_id": f"{drive_id}_crosswalk_review_{cid}",
                    "drive_id": drive_id,
                    "frames_hit": info["frames_hit"],
                    "jitter_p90": info["jitter_p90"],
                    "rect_w_m": info["rect_w_m"],
                    "rect_l_m": info["rect_l_m"],
                    "rectangularity": info.get("rectangularity", 0.0),
                    "cluster_id": cid,
                },
            }
        )
    if not keep:
        return gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:32632")
    return gpd.GeoDataFrame.from_features(keep, crs="EPSG:32632")


def _select_qa_frames(
    drive_id: str,
    records: List[dict],
    final_support: Dict[Tuple[str, str], List[str]],
    candidate_gdf: gpd.GeoDataFrame,
    review_gdf: gpd.GeoDataFrame,
    topn_review: int,
    export_all_frames: bool,
) -> Tuple[set[str], Dict[str, str]]:
    if export_all_frames:
        return {str(_normalize_frame_id(r.get("frame_id") or "")) for r in records}, {}
    frames = set()
    for (drv, frame_id), _ids in final_support.items():
        if drv == drive_id:
            frames.add(_normalize_frame_id(frame_id))
    rep_by_cluster: Dict[str, str] = {}
    if review_gdf.empty or candidate_gdf.empty:
        return frames, rep_by_cluster
    cand = candidate_gdf.copy()
    cand["frame_id_norm"] = cand["frame_id"].apply(_normalize_frame_id)
    cand["cluster_id"] = cand.get("cluster_id", "").astype(str)
    if "score_total" in cand.columns:
        score = pd.to_numeric(cand["score_total"], errors="coerce").fillna(0.0)
        cand = cand.assign(_score=score)
    else:
        cand = cand.assign(_score=0.0)
    review_clusters = sorted({str(v) for v in review_gdf.get("cluster_id", []) if v})
    ranked = []
    for cid in review_clusters:
        subset = cand[cand["cluster_id"] == cid]
        if subset.empty:
            continue
        best_row = subset.sort_values("_score", ascending=False).iloc[0]
        frame_id = str(best_row.get("frame_id_norm") or "")
        ranked.append((float(best_row.get("_score") or 0.0), cid, frame_id))
    ranked.sort(reverse=True)
    for _score, cid, frame_id in ranked[:topn_review]:
        rep_by_cluster[cid] = frame_id
        frames.add(frame_id)
    return frames, rep_by_cluster


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
    records: List[dict],
) -> Tuple[gpd.GeoDataFrame, set[str]]:
    fallback = []
    fallback_frames: set[str] = set()
    existing = set()
    if not candidate_gdf.empty:
        candidate_gdf = candidate_gdf.copy()
        candidate_gdf["frame_id_norm"] = candidate_gdf["frame_id"].apply(_normalize_frame_id)
        existing = set(candidate_gdf["frame_id_norm"].tolist())
    for record in records:
        frame_id = _normalize_frame_id(record.get("frame_id") or "")
        if not frame_id:
            continue
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
    records: List[dict],
    raw_stats: Dict[Tuple[str, str], Dict[str, float]],
    candidate_gdf: gpd.GeoDataFrame,
    final_support: Dict[Tuple[str, str], List[str]],
    cluster_info: Dict[str, dict],
    lidar_stats: Dict[str, dict],
    lidar_world_mode: str,
    pose_source_label: str,
) -> List[dict]:
    candidate_by_frame: Dict[str, gpd.GeoDataFrame] = {}
    candidate_rejects: Dict[str, List[str]] = {}
    if not candidate_gdf.empty:
        candidate_gdf = candidate_gdf.copy()
        candidate_gdf["frame_id_norm"] = candidate_gdf["frame_id"].apply(_normalize_frame_id)
        for frame_id, group in candidate_gdf.groupby("frame_id_norm"):
            candidate_by_frame[str(frame_id)] = group
            reasons = []
            for _, row in group.iterrows():
                tokens = [r for r in str(row.get("reject_reasons") or "").split(",") if r]
                reasons.extend(tokens)
            if reasons:
                candidate_rejects[str(frame_id)] = sorted(set(reasons))
    records_out: List[dict] = []
    for record in records:
        frame_id = _normalize_frame_id(record["frame_id"])
        key = (drive_id, frame_id)
        raw_info = raw_stats.get(
            key,
            {"raw_has_crosswalk": 0.0, "raw_top_score": 0.0, "raw_status": "unknown"},
        )
        lidar_info = lidar_stats.get(frame_id, {})
        candidates = candidate_by_frame.get(frame_id, gpd.GeoDataFrame())
        candidate_count = int(len(candidates)) if not candidates.empty else 0
        cluster_id = ""
        cluster_frames_hit = ""
        if not candidates.empty:
            clusters = [str(v) for v in candidates.get("cluster_id", []).tolist() if v]
            if clusters:
                cluster_id = sorted(set(clusters))[0]
                cluster_frames_hit = str(cluster_info.get(cluster_id, {}).get("frames_hit", ""))
        final_ids = sorted(set(final_support.get(key, [])))
        records_out.append(
            {
                "drive_id": drive_id,
                "frame_id": frame_id,
                "image_path": record.get("image_path", ""),
                "raw_status": raw_info.get("raw_status", "unknown"),
                "raw_has_crosswalk": int(raw_info.get("raw_has_crosswalk", 0)),
                "raw_top_score": raw_info.get("raw_top_score", 0.0),
                "pose_source": pose_source_label,
                "lidar_world_mode": lidar_world_mode,
                "pose_ok": int(lidar_info.get("pose_ok", 0)),
                "calib_ok": int(lidar_info.get("calib_ok", 0)),
                "proj_method": lidar_info.get("proj_method", "none"),
                "proj_in_image_ratio": float(lidar_info.get("proj_in_image_ratio", 0.0)),
                "points_in_bbox": int(lidar_info.get("points_in_bbox", 0)),
                "points_in_mask": int(lidar_info.get("points_in_mask", 0)),
                "mask_dilate_px": int(lidar_info.get("mask_dilate_px", 0)),
                "intensity_top_pct": int(lidar_info.get("intensity_top_pct", 0)),
                "geom_ok": int(lidar_info.get("geom_ok", 0)),
                "geom_area_m2": float(lidar_info.get("geom_area_m2", 0.0)),
                "candidate_written": 1 if candidate_count > 0 else 0,
                "candidate_count": candidate_count,
                "reject_reasons": "|".join(candidate_rejects.get(frame_id, [])),
                "cluster_id": cluster_id,
                "cluster_frames_hit": cluster_frames_hit,
                "final_support": 1 if final_ids else 0,
                "final_entity_id": "|".join(final_ids),
            }
        )
    return records_out


def _build_report(
    out_path: Path,
    drive_id: str,
    trace_records: List[dict],
    candidate_gdf: gpd.GeoDataFrame,
    review_gdf: gpd.GeoDataFrame,
    final_gdf: gpd.GeoDataFrame,
    cluster_info: Dict[str, dict],
    outputs_dir: Path,
    topn_review: int,
    rep_frames_by_cluster: Dict[str, str],
    min_frames_hit_final: int,
    runtime_snapshot: Dict[str, str],
) -> None:
    lines = []
    lines.append("# Crosswalk Full Report\n")
    lines.append(f"- report_time: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("")
    lines.append("## Runtime Config Snapshot")
    if runtime_snapshot:
        for key in sorted(runtime_snapshot.keys()):
            lines.append(f"- {key}: {runtime_snapshot[key]}")
    else:
        lines.append("- none")
    lines.append(f"- drive_id: {drive_id}")
    lines.append(f"- total_frames: {len(trace_records)}")
    lines.append("")
    raw_hits = [r for r in trace_records if int(r.get("raw_has_crosswalk", 0)) == 1]
    geom_hits = [r for r in trace_records if int(r.get("geom_ok", 0)) == 1]
    gate_hits = [r for r in trace_records if str(r.get("reject_reasons") or "") == ""]
    lines.append(f"- raw_has_crosswalk_count: {len(raw_hits)}")
    lines.append(f"- geom_ok_count: {len(geom_hits)}")
    lines.append(f"- frame_gate_pass_count: {len(gate_hits)}")
    lines.append(f"- candidate_count_total: {len(candidate_gdf)}")
    lines.append(f"- review_count: {len(review_gdf)}")
    lines.append(f"- final_count: {len(final_gdf)}")
    frames_hit_ge_final = len([info for info in cluster_info.values() if info.get("frames_hit", 0) >= min_frames_hit_final])
    lines.append(f"- clusters_frames_hit_ge_{min_frames_hit_final}: {frames_hit_ge_final}")
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
    frames_hit = [info["frames_hit"] for info in cluster_info.values()]
    jitters = [info["jitter_p90"] for info in cluster_info.values()]
    if frames_hit:
        lines.append("## cluster_frames_hit Summary")
        lines.append(f"- p50: {np.percentile(frames_hit, 50):.1f}")
        lines.append(f"- p90: {np.percentile(frames_hit, 90):.1f}")
        lines.append("")
    if jitters:
        lines.append("## jitter_p90 Summary")
        lines.append(f"- p50: {np.percentile(jitters, 50):.1f}")
        lines.append(f"- p90: {np.percentile(jitters, 90):.1f}")
        lines.append("")
    lines.append("## Suspicious Review Clusters")
    review_ids = set(review_gdf.get("cluster_id", [])) if not review_gdf.empty else set()
    final_ids = set()
    if not final_gdf.empty:
        for _, row in final_gdf.iterrows():
            cid = row.get("cluster_id")
            if cid:
                final_ids.add(str(cid))
    suspicious = [cid for cid in review_ids if cid not in final_ids]
    for cid in sorted(suspicious)[:topn_review]:
        info = cluster_info.get(cid, {})
        frame_id = rep_frames_by_cluster.get(cid, "")
        qa_dir = outputs_dir / "qa_images" / drive_id
        qa_paths = ""
        if frame_id:
            qa_paths = f" raw={qa_dir / f'{frame_id}_overlay_raw.png'} gated={qa_dir / f'{frame_id}_overlay_gated.png'} entities={qa_dir / f'{frame_id}_overlay_entities.png'}"
        lines.append(
            f"- {cid} frames_hit={info.get('frames_hit')} jitter_p90={info.get('jitter_p90')} rect_w={info.get('rect_w_m')} rect_l={info.get('rect_l_m')} rectangularity={info.get('rectangularity')}{qa_paths}"
        )
    if not suspicious:
        lines.append("- none")
    lines.append("")
    lines.append(f"- outputs_dir: {outputs_dir}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crosswalk_drive_full.yaml")
    ap.add_argument("--drive", default=None)
    ap.add_argument("--kitti-root", default=None)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--export-all-frames", type=int, default=None)
    ap.add_argument("--out-run", default="")
    args = ap.parse_args()

    log = _setup_logger()
    cfg = _load_yaml(Path(args.config))
    merged = _merge_config(
        cfg,
        {
            "drive_id": args.drive,
            "kitti_root": args.kitti_root,
            "camera": args.camera,
            "export_all_frames": args.export_all_frames,
        },
    )

    drive_id = str(merged.get("drive_id") or "")
    kitti_root = Path(str(merged.get("kitti_root") or ""))
    defaults = _load_camera_defaults(Path("configs/camera_defaults.yaml"))
    default_camera = str(defaults.get("default_camera") or "image_00")
    enforce_camera = bool(defaults.get("enforce_camera", True))
    allow_override = bool(defaults.get("allow_override", False))
    if str(os.environ.get("ALLOW_CAMERA_OVERRIDE", "0")).strip() == "1":
        allow_override = True
    camera = str(merged.get("camera") or default_camera)
    stage1_stride = int(merged.get("stage1_stride", 1))
    export_all_frames = bool(merged.get("export_all_frames", False))
    min_frames_hit_final = int(merged.get("min_frames_hit_final", 3))
    write_wgs84 = bool(merged.get("write_wgs84", True))
    raw_fallback_text = bool(merged.get("raw_fallback_text", True))
    image_run = Path(str(merged.get("image_run") or ""))
    image_provider = str(merged.get("image_provider") or "grounded_sam2_v1")
    image_evidence_gpkg = str(merged.get("image_evidence_gpkg") or "")
    road_root = Path(str(merged.get("road_root") or ""))
    base_config = Path(str(merged.get("config") or "configs/road_entities.yaml"))
    cluster_cfg = merged.get("cluster", {}) if isinstance(merged.get("cluster"), dict) else {}
    final_cfg = merged.get("final_gate", {}) if isinstance(merged.get("final_gate"), dict) else {}
    review_cfg = merged.get("review_gate", {}) if isinstance(merged.get("review_gate"), dict) else {}
    lidar_cfg = merged.get("lidar_proj", {}) if isinstance(merged.get("lidar_proj"), dict) else {}
    topn_review = int(merged.get("topn_review", 50))

    if not drive_id:
        log.error("drive_id required")
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

    drive_tag = drive_id
    parts = drive_id.split("_")
    if len(parts) >= 2 and parts[-2].isdigit():
        drive_tag = parts[-2]
    run_dir = Path(args.out_run) if args.out_run else Path("runs") / f"crosswalk_drive{drive_tag}_full_{dt.datetime.now():%Y%m%d_%H%M%S}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    outputs_dir = run_dir / "outputs"
    debug_dir = run_dir / "debug"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    lidar_world_mode = str(merged.get("lidar_world_mode") or os.environ.get("LIDAR_WORLD_MODE") or "").strip().lower()
    if str(os.environ.get("USE_FULLPOSE_LIDAR", "0")).strip() == "1":
        lidar_world_mode = "fullpose"
    if not lidar_world_mode:
        lidar_world_mode = "fullpose"
    image_dir = _find_image_dir(kitti_root, drive_id, camera)
    try:
        calib = load_kitti360_calib(kitti_root, camera)
    except Exception:
        calib = None
    cam_to_pose_key = ""
    try:
        _cam_to_pose, cam_to_pose_key = load_kitti360_cam_to_pose_key(kitti_root, camera)
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

    index_records = _build_index_records(kitti_root, drive_id, camera, stage1_stride)
    if not index_records:
        log.error("no frames found for drive=%s", drive_id)
        return 3
    index_path = debug_dir / "monitor_index.jsonl"
    _write_index(index_records, index_path)
    log.info("index=%s total=%d", index_path, len(index_records))

    stage_cfg = debug_dir / "crosswalk_monitor.yaml"
    final_overrides = {
        "min_frames_hit": int(final_cfg.get("min_frames_hit_final", min_frames_hit_final)),
        "min_inside_ratio": float(final_cfg.get("min_inside_ratio_final", 0.5)),
        "min_rect_w_m": float(final_cfg.get("rect_min_w", 1.5)),
        "max_rect_w_m": float(final_cfg.get("rect_max_w", 8.0)),
        "min_rect_l_m": float(final_cfg.get("rect_min_l", 3.0)),
        "max_rect_l_m": float(final_cfg.get("rect_max_l", 25.0)),
        "min_rectangularity": float(final_cfg.get("rectangularity_min", 0.45)),
        "max_heading_diff_deg": float(final_cfg.get("heading_diff_to_perp_max_deg", 25.0)),
        "jitter_p90_max": float(cluster_cfg.get("jitter_p90_max", 8.0)),
        "angle_jitter_p90_max": float(cluster_cfg.get("angle_jitter_p90_max", 25.0)),
    }
    cluster_overrides = {
        "crosswalk_eps_m": float(cluster_cfg.get("cluster_eps_m", 6.0)),
    }
    stage_cfg.write_text(
        yaml.safe_dump(
            _merge_config(
                _load_yaml(base_config),
                {
                    "crosswalk_final": final_overrides,
                    "clustering": cluster_overrides,
                },
            ),
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
    qa_index_path = stage_outputs / "qa_index_wgs84.geojson"
    qa_out_path = outputs_dir / "qa_index_wgs84.geojson"
    if qa_out_path.exists():
        qa_out_path.unlink()
    if qa_index_path.exists():
        shutil.copy2(qa_index_path, qa_out_path)

    index_lookup = {(r["drive_id"], _normalize_frame_id(r["frame_id"])): r.get("image_path", "") for r in index_records}
    pose_map: Dict[str, Tuple[float, float, float] | Tuple[float, float, float, float, float, float] | None] = {}
    for record in index_records:
        frame_id = _normalize_frame_id(record.get("frame_id") or "")
        if not frame_id:
            continue
        try:
            if lidar_world_mode == "fullpose":
                x, y, z, roll, pitch, yaw = load_kitti360_pose_full(kitti_root, drive_id, frame_id)
                pose_map[frame_id] = (x, y, z, roll, pitch, yaw)
            else:
                x, y, yaw = load_kitti360_pose(kitti_root, drive_id, frame_id)
                pose_map[frame_id] = (x, y, yaw)
        except Exception:
            pose_map[frame_id] = None
    raw_stats, raw_frames = _ensure_raw_overlays(
        qa_out_path,
        outputs_dir,
        image_run,
        image_provider,
        index_lookup,
        raw_fallback_text,
        {"__skip__"},
    )

    stage_gpkg = stage_outputs / "road_entities_utm32.gpkg"
    candidate_gdf = _read_candidates(stage_gpkg)
    final_gdf = _read_final(stage_gpkg)
    calib_ok = calib is not None
    lidar_cfg_norm = {
        "MASK_DILATE_PX": int(lidar_cfg.get("mask_dilate_px", 5)),
        "MIN_POINTS_BBOX": int(lidar_cfg.get("min_points_bbox", 20)),
        "MIN_POINTS_MASK": int(lidar_cfg.get("min_points_mask", 5)),
        "INTENSITY_TOP_PCT": int(lidar_cfg.get("intensity_top_pct", 20)),
        "MIN_IN_IMAGE_RATIO": float(lidar_cfg.get("min_in_image_ratio", 0.1)),
        "ACCUM_FRAMES": int(lidar_cfg.get("accum_frames", 3)),
        "lidar_world_mode": lidar_world_mode,
        "cam_id": camera,
    }
    lidar_candidates, lidar_stats = _build_lidar_candidates_for_records(
        kitti_root,
        drive_id,
        index_records,
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
    candidate_gdf, _fallback_frames = _augment_candidates_with_fallback(
        candidate_gdf,
        raw_stats,
        pose_map,
        calib_ok,
        raw_frames,
        lidar_stats,
        drive_id,
        index_records,
    )
    candidate_gdf, cluster_info = _build_clusters(
        candidate_gdf,
        float(cluster_cfg.get("cluster_eps_m", 6.0)),
    )
    review_gdf = _build_review_layer(candidate_gdf, cluster_info, review_cfg, drive_id)
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
    frames_to_render, rep_frames_by_cluster = _select_qa_frames(
        drive_id,
        index_records,
        final_support,
        candidate_gdf,
        review_gdf,
        topn_review,
        export_all_frames,
    )
    raw_stats, raw_frames = _ensure_raw_overlays(
        qa_out_path,
        outputs_dir,
        image_run,
        image_provider,
        index_lookup,
        raw_fallback_text,
        frames_to_render,
    )
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
        frames_to_render,
    )

    pose_source_label = "oxts_full" if lidar_world_mode == "fullpose" else "oxts"
    trace_records = _build_trace_records(
        drive_id,
        index_records,
        raw_stats,
        candidate_gdf,
        final_support,
        cluster_info,
        lidar_stats,
        lidar_world_mode,
        pose_source_label,
    )
    trace_path = outputs_dir / "crosswalk_trace.csv"
    _build_trace(trace_path, trace_records)

    report_path = outputs_dir / "crosswalk_full_report.md"
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
    }
    _build_report(
        report_path,
        drive_id,
        trace_records,
        candidate_gdf,
        review_gdf,
        final_gdf,
        cluster_info,
        outputs_dir,
        topn_review,
        rep_frames_by_cluster,
        min_frames_hit_final,
        runtime_snapshot,
    )

    log.info("done: %s", outputs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

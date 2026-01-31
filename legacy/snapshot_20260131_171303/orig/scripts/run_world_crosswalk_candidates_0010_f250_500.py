from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import subprocess
from collections import Counter
import random
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
from pipeline.calib.kitti360_world import kitti_world_to_utm32, utm32_to_kitti_world, utm32_to_wk
from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_csv, write_json, write_text


CFG_DEFAULT = Path("configs/world_crosswalk_candidates_0010_f250_500.yaml")

MARGIN_ALONG_M = 0.6
MARGIN_ACROSS_M = 0.8
CANONICAL_V3_MARGIN_M = 0.30
CANONICAL_V3_MIN_AREA_M2 = 10.0

STRIPE_ASPECT_MIN = 3.0
STRIPE_AREA_MIN_PX = 200
STRIPE_SHORT_MIN_PX = 3.0
STRIPE_SHORT_MAX_PX = 40.0
STRIPE_LONG_MIN_PX = 25.0
STRIPE_LONG_MAX_PX = 400.0
STRIPE_COUNT_MIN = 6

CANONICAL_AREA_MIN_M2 = 10.0
CANONICAL_AREA_MAX_M2 = 350.0

CANONICAL_MERGE_CENTROID_DIST_M = 2.0
CANONICAL_MERGE_ORI_DIFF_DEG = 15.0

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
    candidates = [p for p in runs_dir.glob("image_stage12_crosswalk_0010_250_500_*") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_latest_gate_run() -> Optional[Path]:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.glob("backproject_cycle_gate_0010_f290_*") if p.is_dir()]
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


def _extract_stripes(mask: np.ndarray) -> List[Dict[str, float]]:
    stripes = []
    try:
        import cv2
    except Exception:
        return stripes
    mask_u8 = (mask.astype(np.uint8) * 255)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < STRIPE_AREA_MIN_PX:
            continue
        comp = (labels == idx).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=lambda c: c.shape[0])
        if contour is None or len(contour) < 5:
            continue
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect
        long_side = max(float(w), float(h))
        short_side = min(float(w), float(h))
        if short_side <= 0:
            continue
        aspect = long_side / max(short_side, 1e-6)
        if aspect < STRIPE_ASPECT_MIN:
            continue
        if short_side < STRIPE_SHORT_MIN_PX or short_side > STRIPE_SHORT_MAX_PX:
            continue
        if long_side < STRIPE_LONG_MIN_PX or long_side > STRIPE_LONG_MAX_PX:
            continue
        if w < h:
            angle = angle + 90.0
        theta = np.deg2rad(angle)
        dx = 0.5 * long_side * float(np.cos(theta))
        dy = 0.5 * long_side * float(np.sin(theta))
        stripes.append(
            {
                "cx": float(cx),
                "cy": float(cy),
                "u1": float(cx - dx),
                "v1": float(cy - dy),
                "u2": float(cx + dx),
                "v2": float(cy + dy),
                "long_px": float(long_side),
                "short_px": float(short_side),
                "aspect": float(aspect),
                "area_px": float(area),
                "angle_deg": float(angle % 180.0),
            }
        )
    return stripes


def _pca_dir(points_xy: np.ndarray) -> Optional[np.ndarray]:
    if points_xy.size == 0 or points_xy.shape[0] < 2:
        return None
    pts = points_xy[:, :2].astype(np.float64)
    mean = np.mean(pts, axis=0, keepdims=True)
    centered = pts - mean
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    if vecs is None or vecs.shape[1] < 2:
        return None
    dir_vec = vecs[:, np.argmax(vals)]
    norm = np.linalg.norm(dir_vec)
    if norm <= 0:
        return None
    dir_vec = dir_vec / norm
    if dir_vec[0] < 0:
        dir_vec = -dir_vec
    return dir_vec.astype(np.float64)


def _canonical_rect_from_points(points_wu: np.ndarray) -> Optional[Dict[str, object]]:
    if points_wu.size == 0 or points_wu.shape[0] < 3:
        return None
    dir_vec = _pca_dir(points_wu[:, :2])
    if dir_vec is None:
        return None
    perp = np.array([-dir_vec[1], dir_vec[0]], dtype=np.float64)
    s = points_wu[:, :2] @ dir_vec
    t = points_wu[:, :2] @ perp
    s_min, s_max = float(np.min(s)), float(np.max(s))
    t_min, t_max = float(np.min(t)), float(np.max(t))
    s_min -= MARGIN_ALONG_M
    s_max += MARGIN_ALONG_M
    t_min -= MARGIN_ACROSS_M
    t_max += MARGIN_ACROSS_M
    length = s_max - s_min
    width = t_max - t_min
    s0 = 0.5 * (s_min + s_max)
    t0 = 0.5 * (t_min + t_max)
    center = dir_vec * s0 + perp * t0
    corners_st = [
        (s0 - 0.5 * length, t0 - 0.5 * width),
        (s0 + 0.5 * length, t0 - 0.5 * width),
        (s0 + 0.5 * length, t0 + 0.5 * width),
        (s0 - 0.5 * length, t0 + 0.5 * width),
    ]
    corners_xy = []
    for s_i, t_i in corners_st:
        pt = dir_vec * float(s_i) + perp * float(t_i)
        corners_xy.append((float(pt[0]), float(pt[1])))
    try:
        from shapely.geometry import Polygon

        poly = Polygon(corners_xy)
    except Exception:
        return None
    if poly.is_empty:
        return None
    area = float(poly.area)
    angle_deg = float(np.degrees(np.arctan2(dir_vec[1], dir_vec[0])) % 180.0)
    return {
        "polygon": poly,
        "center": (float(center[0]), float(center[1])),
        "angle_deg": angle_deg,
        "length_m": float(length),
        "width_m": float(width),
        "area_m2": area,
    }


def _rect_from_center(center: Tuple[float, float], angle_deg: float, length: float, width: float):
    try:
        from shapely.geometry import Polygon
    except Exception:
        return None
    theta = np.deg2rad(angle_deg)
    dir_vec = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    perp = np.array([-dir_vec[1], dir_vec[0]], dtype=np.float64)
    half_l = 0.5 * float(length)
    half_w = 0.5 * float(width)
    c = np.array([float(center[0]), float(center[1])], dtype=np.float64)
    corners = [
        c + dir_vec * half_l + perp * half_w,
        c + dir_vec * half_l - perp * half_w,
        c - dir_vec * half_l - perp * half_w,
        c - dir_vec * half_l + perp * half_w,
    ]
    return Polygon([(float(p[0]), float(p[1])) for p in corners])


def _project_polygon_to_pixel(
    frame_id: str, poly_wu, ctx: BackprojectContext, z0: float
) -> Tuple[List[Tuple[float, float]], float]:
    if poly_wu is None or poly_wu.is_empty:
        return [], 0.0
    coords = list(poly_wu.exterior.coords) if hasattr(poly_wu, "exterior") else []
    if len(coords) < 3:
        return [], 0.0
    pts = []
    for x, y in coords:
        z = None
        if ctx.dtm is not None:
            try:
                val = next(ctx.dtm.sample([(float(x), float(y))]))
                if val is not None and len(val) > 0:
                    z = float(val[0])
                    if ctx.dtm_nodata is not None and np.isfinite(ctx.dtm_nodata):
                        if abs(z - float(ctx.dtm_nodata)) < 1e-6:
                            z = None
                    if z is not None and not np.isfinite(z):
                        z = None
            except Exception:
                z = None
        if z is None:
            z = float(z0)
        pts.append([float(x), float(y), float(z)])
    pts_wu = np.array(pts, dtype=np.float64)
    pts_wk = utm32_to_kitti_world(pts_wu, ctx.data_root, ctx.drive_id, frame_id)
    u, v, valid = world_to_pixel_cam0(frame_id, pts_wk, ctx=ctx)
    proj = [(float(uu), float(vv)) for uu, vv, ok in zip(u, v, valid) if ok]
    return proj, float(np.count_nonzero(valid) / max(1, len(valid)))


def _project_polygon_wk_to_pixel(
    frame_id: str, poly_wk, ctx: BackprojectContext, z_wk: float
) -> Tuple[List[Tuple[float, float]], float]:
    if poly_wk is None or poly_wk.is_empty:
        return [], 0.0
    coords = list(poly_wk.exterior.coords) if hasattr(poly_wk, "exterior") else []
    if len(coords) < 3:
        return [], 0.0
    pts_wk = np.array([[float(x), float(y), float(z_wk)] for x, y in coords], dtype=np.float64)
    u, v, valid = world_to_pixel_cam0(frame_id, pts_wk, ctx=ctx)
    proj = [(float(uu), float(vv)) for uu, vv, ok in zip(u, v, valid) if ok]
    return proj, float(np.count_nonzero(valid) / max(1, len(valid)))


def _poly_wk_to_wu(poly_wk, z_wk: float, ctx: BackprojectContext, frame_id: str):
    if poly_wk is None or poly_wk.is_empty:
        return None
    try:
        from shapely.geometry import Polygon, MultiPolygon
    except Exception:
        return None
    polys = []
    if hasattr(poly_wk, "geoms"):
        geom_list = list(poly_wk.geoms)
    else:
        geom_list = [poly_wk]
    for geom in geom_list:
        coords = list(geom.exterior.coords) if hasattr(geom, "exterior") else []
        if len(coords) < 3:
            continue
        pts_wk = np.array([[float(x), float(y), float(z_wk)] for x, y in coords], dtype=np.float64)
        pts_wu = kitti_world_to_utm32(pts_wk, ctx.data_root, ctx.drive_id, frame_id)
        polys.append(Polygon([(float(p[0]), float(p[1])) for p in pts_wu]))
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    return MultiPolygon(polys)


def _poly_wu_to_wk(poly_wu, z_wu: float, ctx: BackprojectContext, frame_id: str):
    if poly_wu is None or poly_wu.is_empty:
        return None
    try:
        from shapely.geometry import Polygon, MultiPolygon
    except Exception:
        return None
    polys = []
    geom_list = list(poly_wu.geoms) if hasattr(poly_wu, "geoms") else [poly_wu]
    for geom in geom_list:
        coords = list(geom.exterior.coords) if hasattr(geom, "exterior") else []
        if len(coords) < 3:
            continue
        pts_wu = np.array([[float(x), float(y), float(z_wu)] for x, y in coords], dtype=np.float64)
        pts_wk = utm32_to_wk(pts_wu, ctx.data_root, ctx.drive_id, frame_id)
        polys.append(Polygon([(float(p[0]), float(p[1])) for p in pts_wk]))
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    return MultiPolygon(polys)


def _rect_params(rect) -> Tuple[Tuple[float, float], float, float, float]:
    coords = list(rect.exterior.coords) if hasattr(rect, "exterior") else []
    if len(coords) < 4:
        return (0.0, 0.0), 0.0, 0.0, 0.0
    pts = np.array(coords[:-1], dtype=np.float64)
    edges = pts[1:] - pts[:-1]
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.size == 0:
        return (0.0, 0.0), 0.0, 0.0, 0.0
    idx = int(np.argmax(lengths))
    edge = edges[idx]
    angle = float(np.degrees(np.arctan2(edge[1], edge[0])) % 180.0)
    length = float(np.max(lengths))
    width = float(np.min(lengths))
    center = rect.centroid
    return (float(center.x), float(center.y)), angle, length, width


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


def _rect_metrics(rect, raw_union) -> Tuple[float, float, float]:
    if rect is None or rect.is_empty or raw_union is None or raw_union.is_empty:
        return 0.0, 0.0, 0.0
    try:
        inter = rect.intersection(raw_union).area
        uni = rect.union(raw_union).area
    except Exception:
        return 0.0, 0.0, 0.0
    iou = float(inter / uni) if uni > 0 else 0.0
    coverage = float(inter / raw_union.area) if raw_union.area > 0 else 0.0
    dist = float(rect.centroid.distance(raw_union.centroid))
    return dist, iou, coverage


def _load_raw_union_items_from_frames(frames_dir: Path) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    try:
        import geopandas as gpd
    except Exception:
        return items
    for frame_dir in sorted(frames_dir.iterdir()):
        if not frame_dir.is_dir():
            continue
        gpkg_utm = frame_dir / "world_crosswalk_utm32.gpkg"
        gpkg_wk = frame_dir / "world_crosswalk_wk.gpkg"
        if not gpkg_utm.exists() or not gpkg_wk.exists():
            continue
        try:
            gdf_utm = gpd.read_file(gpkg_utm, layer="crosswalk_frame_raw")
            gdf_wk = gpd.read_file(gpkg_wk, layer="crosswalk_frame_raw_wk")
        except Exception:
            continue
        if gdf_utm.empty or gdf_wk.empty:
            continue
        geom_utm = _make_valid(gdf_utm.geometry.iloc[0])
        geom_wk = _make_valid(gdf_wk.geometry.iloc[0])
        if geom_utm is None or geom_utm.is_empty or geom_wk is None or geom_wk.is_empty:
            continue
        raw_z = 0.0
        debug_path = frame_dir / "landing_debug.json"
        if debug_path.exists():
            try:
                debug = json.loads(debug_path.read_text())
                raw_z = float(debug.get("canonical_z_wk", raw_z))
            except Exception:
                raw_z = raw_z
        items.append(
            {
                "frame_id": frame_dir.name,
                "geometry_utm": geom_utm,
                "geometry_wk": geom_wk,
                "raw_z_wk": raw_z,
            }
        )
    return items


def _merge_from_raw_union_items(
    raw_union_items: List[Dict[str, object]],
    ctx: BackprojectContext,
    merge_cfg: Dict[str, object],
    frame_start: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], Dict[str, int]]:
    merged_items: List[Dict[str, object]] = []
    merged_raw_items: List[Dict[str, object]] = []
    canonical_metrics: List[Dict[str, object]] = []
    clusters: List[Dict[str, object]] = []
    reason_counts: Dict[str, int] = {}

    raw_union_by_frame_utm = {
        item["frame_id"]: _make_valid(item.get("geometry_utm")) for item in raw_union_items
    }
    raw_union_by_frame_wk = {
        item["frame_id"]: _make_valid(item.get("geometry_wk")) for item in raw_union_items
    }
    raw_z_by_frame = {item["frame_id"]: float(item.get("raw_z_wk", 0.0)) for item in raw_union_items}

    for item in raw_union_items:
        raw_geom = raw_union_by_frame_utm.get(item["frame_id"])
        if raw_geom is None or raw_geom.is_empty:
            continue
        try:
            rect = raw_geom.minimum_rotated_rectangle
            center, angle, _, _ = _rect_params(rect)
        except Exception:
            continue
        center = np.array(center, dtype=np.float64)
        angle = float(angle) % 180.0
        best_idx = -1
        best_dist = 1e9
        for idx, cluster in enumerate(clusters):
            c_center = np.array(cluster["center"], dtype=np.float64)
            c_angle = float(cluster["angle_deg"]) % 180.0
            dist = float(np.linalg.norm(center - c_center))
            ang_diff = abs(angle - c_angle)
            ang_diff = min(ang_diff, 180.0 - ang_diff)
            if dist <= CANONICAL_MERGE_CENTROID_DIST_M and ang_diff <= CANONICAL_MERGE_ORI_DIFF_DEG:
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
        if best_idx >= 0:
            clusters[best_idx]["frames"].append(item["frame_id"])
        else:
            clusters.append({"frames": [item["frame_id"]], "center": center.tolist(), "angle_deg": angle})

    min_support = int(merge_cfg.get("min_support_frames", 3))
    clusters_ge_min = 0
    for idx, cluster in enumerate(clusters, start=1):
        frame_ids = cluster["frames"]
        if len(frame_ids) < min_support:
            reason_counts["min_support"] = reason_counts.get("min_support", 0) + 1
            continue
        clusters_ge_min += 1
        frame_id_ref = sorted(frame_ids)[0] if frame_ids else f"{frame_start:010d}"
        raw_geoms_utm = [
            g
            for fid in frame_ids
            for g in [raw_union_by_frame_utm.get(fid)]
            if g is not None and not g.is_empty and float(getattr(g, "area", 0.0)) > 0.0
        ]
        raw_geoms_wk = [
            g
            for fid in frame_ids
            for g in [raw_union_by_frame_wk.get(fid)]
            if g is not None and not g.is_empty and float(getattr(g, "area", 0.0)) > 0.0
        ]
        raw_z_list = [
            raw_z_by_frame.get(fid, 0.0)
            for fid in frame_ids
            if raw_union_by_frame_wk.get(fid) is not None
        ]
        if not raw_geoms_utm or not raw_geoms_wk:
            reason_counts["missing_geoms"] = reason_counts.get("missing_geoms", 0) + 1
            continue
        try:
            merged_raw_union_utm = unary_union(raw_geoms_utm)
        except Exception:
            merged_raw_union_utm = None
        try:
            merged_raw_union_wk = unary_union(raw_geoms_wk)
        except Exception:
            merged_raw_union_wk = None
        if merged_raw_union_utm is None or merged_raw_union_utm.is_empty or merged_raw_union_wk is None or merged_raw_union_wk.is_empty:
            reason_counts["empty_union"] = reason_counts.get("empty_union", 0) + 1
            continue
        bbox_ok_raw, _ = _check_utm_bbox(merged_raw_union_utm)
        if not bbox_ok_raw:
            reason_counts["bbox_invalid"] = reason_counts.get("bbox_invalid", 0) + 1
            continue
        try:
            rect_wk = merged_raw_union_wk.minimum_rotated_rectangle
            geom_wk = rect_wk.buffer(CANONICAL_V3_MARGIN_M).buffer(-CANONICAL_V3_MARGIN_M)
            geom_wk = _make_valid(geom_wk)
        except Exception:
            geom_wk = None
        if geom_wk is None or geom_wk.is_empty or float(geom_wk.area) < CANONICAL_V3_MIN_AREA_M2:
            reason_counts["geom_wk_invalid"] = reason_counts.get("geom_wk_invalid", 0) + 1
            continue
        merged_z_wk = float(np.median(raw_z_list)) if raw_z_list else 0.0
        geom = _poly_wk_to_wu(geom_wk, merged_z_wk, ctx, frame_id_ref)
        if geom is None or geom.is_empty:
            reason_counts["geom_utm_empty"] = reason_counts.get("geom_utm_empty", 0) + 1
            continue
        center_pt = geom.centroid
        rect_utm = geom.minimum_rotated_rectangle
        center_c, angle_deg, length_med, width_med = _rect_params(rect_utm)
        merged_items.append(
            {
                "candidate_id": f"cand_{idx:03d}",
                "support_frames": len(frame_ids),
                "geometry": geom,
                "frame_id_ref": frame_id_ref,
                "center_x": float(center_pt.x),
                "center_y": float(center_pt.y),
                "angle_deg": angle_deg,
                "length_m": length_med,
                "width_m": width_med,
                "coord_space": "utm32",
            }
        )
        merged_raw_items.append(
            {
                "candidate_id": f"cand_{idx:03d}",
                "support_frames": len(frame_ids),
                "geometry": merged_raw_union_utm,
                "frame_id_ref": frame_id_ref,
                "coord_space": "utm32",
            }
        )
        dist_center, iou_val, coverage_raw = _rect_metrics(geom, merged_raw_union_utm)
        canonical_metrics.append(
            {
                "candidate_id": f"cand_{idx:03d}",
                "support_frames": len(frame_ids),
                "dist_center_m": round(dist_center, 3),
                "iou": round(iou_val, 3),
                "coverage_raw": round(coverage_raw, 3),
            }
        )

    debug = {
        "raw_union_items": len(raw_union_items),
        "clusters": len(clusters),
        "clusters_ge_min_support": clusters_ge_min,
        "reason_counts": reason_counts,
    }
    return merged_items, merged_raw_items, canonical_metrics, debug


def _polygon_iou(mask: np.ndarray, poly_pts: List[Tuple[float, float]]) -> float:
    if mask.size == 0 or not poly_pts:
        return 0.0
    h, w = mask.shape[:2]
    poly_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(poly_img)
    draw.polygon(poly_pts, outline=1, fill=1)
    poly_arr = np.array(poly_img) > 0
    inter = np.logical_and(mask, poly_arr).sum()
    union = np.logical_or(mask, poly_arr).sum()
    if union <= 0:
        return 0.0
    return float(inter / union)

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
    min_pts: int,
) -> Tuple[List[np.ndarray], List[Tuple[float, float]], str]:
    sampled_contours, sampled_pts = _sample_contour_points(
        contours, step_px, img_w, img_h, bottom_crop, side_crop
    )
    if len(sampled_pts) >= min_pts:
        return sampled_contours, sampled_pts, "strict"
    sampled_contours, sampled_pts = _sample_contour_points(
        contours, step_px, img_w, img_h, bottom_crop_fb, side_crop_fb
    )
    return sampled_contours, sampled_pts, "fallback"


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
    max_ray_t: float,
) -> Tuple[Optional[np.ndarray], str]:
    z_est = float(z0)
    for _ in range(max(1, dtm_iter)):
        pt = pixel_to_world_on_ground(frame_id, u, v, {"mode": "fixed_plane", "z0": z_est}, ctx=ctx)
        if pt is None:
            return None, "ray_upwards"
        if use_dtm and ctx.dtm is not None:
            dtm_z = _sample_dtm(ctx, frame_id, pt[0], pt[1], pt[2])
            if dtm_z is None:
                break
            z_est = float(dtm_z)
        else:
            break
    origin = ctx.pose_provider.get_t_w_c0(str(frame_id))[:3, 3]
    dist = float(np.linalg.norm(pt - origin))
    if dist > max_ray_t:
        return None, "ray_t_exceed"
    return pt, "ok"


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


def _simplify_geom(geom, tol: float, buffer_m: float, min_area: float):
    if geom is None or geom.is_empty:
        return None
    try:
        g = geom.simplify(tol, preserve_topology=True)
        if buffer_m > 0:
            g = g.buffer(buffer_m).buffer(-buffer_m)
        if g.is_empty:
            return None
        if g.area < min_area:
            return None
        return g
    except Exception:
        return None


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


def _pixel_contour_path(contours: List[np.ndarray]) -> Optional[np.ndarray]:
    if not contours:
        return None
    longest = max(contours, key=lambda c: c.shape[0])
    return longest


def _draw_pixel_contour(img: Image.Image, contour: np.ndarray, out_path: Path) -> None:
    overlay = img.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    pts = [(float(u), float(v)) for u, v in contour]
    if len(pts) >= 2:
        draw.line(pts + [pts[0]], fill=(255, 0, 0), width=2)
    overlay.save(out_path)


def _draw_roundtrip_overlay(
    img: Image.Image,
    pixel_contour: np.ndarray,
    reproj_pts: List[Tuple[float, float]],
    out_path: Path,
) -> None:
    overlay = img.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    if pixel_contour is not None and len(pixel_contour) >= 2:
        pts = [(float(u), float(v)) for u, v in pixel_contour]
        draw.line(pts + [pts[0]], fill=(255, 0, 0), width=2)
    if len(reproj_pts) >= 2:
        draw.line(reproj_pts + [reproj_pts[0]], fill=(0, 255, 0), width=2)
    overlay.save(out_path)


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


def _transform_geom(geom, fn):
    if geom is None:
        return None
    try:
        from shapely.ops import transform

        return transform(fn, geom)
    except Exception:
        return geom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CFG_DEFAULT))
    ap.add_argument("--frames", default="")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    if not cfg:
        raise SystemExit(f"missing config: {cfg_path}")

    drive_id = str(cfg.get("drive_id") or "")
    frame_start = int(cfg.get("frame_start", 250))
    frame_end = int(cfg.get("frame_end", 500))
    image_cam = str(cfg.get("image_cam") or "image_00")
    cycle_gate_enable = bool(cfg.get("cycle_gate_enable", True))

    run_id = now_ts()
    run_dir = Path("runs") / f"world_crosswalk_candidates_0010_250_500_{run_id}"
    ensure_overwrite(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")

    input_run = str(cfg.get("input_stage12_run") or "auto_latest")
    if input_run == "auto_latest":
        stage12_run = _find_latest_stage12_run()
    else:
        stage12_run = Path(input_run)
    if stage12_run is None or not stage12_run.exists():
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "stage12_run_not_found"})
        raise SystemExit("stage12 run not found")

    if cycle_gate_enable:
        gate_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_backproject_cycle_gate_0010_frame290.py"),
            "--config",
            str(args.config),
            "--stage12-run",
            str(stage12_run),
        ]
        subprocess.run(gate_cmd, check=False)
        gate_run = _find_latest_gate_run()
        gate_ok = False
        if gate_run:
            gate_decision = gate_run / "decision.json"
            if gate_decision.exists():
                data = json.loads(gate_decision.read_text(encoding="utf-8"))
                gate_ok = data.get("status") == "PASS"
        if not gate_ok:
            write_json(
                run_dir / "decision.json",
                {
                    "status": "FAIL",
                    "reason": "cycle_gate_failed",
                    "gate_run": str(gate_run) if gate_run else "",
                },
            )
            raise SystemExit("cycle gate failed")

    mask_dir = stage12_run / str(cfg.get("input_mask_dir") or "stage2/masks")
    if not mask_dir.exists():
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "mask_dir_missing"})
        raise SystemExit("mask_dir missing")

    qa_path = stage12_run / str(cfg.get("qa_frames_path") or "qa/qa_frames.json")
    qa_frames: List[int] = [250, 290, 341]
    if qa_path.exists():
        qa_frames = json.loads(qa_path.read_text(encoding="utf-8")).get("frames", qa_frames)

    data_root = _find_data_root(str(cfg.get("kitti_root") or ""))
    image_dir = _find_image_dir(data_root, drive_id, image_cam)

    back_cfg = cfg.get("backproject") or {}
    use_dtm = bool(back_cfg.get("use_dtm", True))
    dtm_path_cfg = str(back_cfg.get("dtm_path") or "auto_latest_clean_dtm")
    if dtm_path_cfg == "auto_latest_clean_dtm":
        dtm_path = _find_latest_dtm()
    else:
        dtm_path = Path(dtm_path_cfg) if dtm_path_cfg else None
    dtm_median = _dtm_median(dtm_path) if use_dtm else None

    ctx = configure_default_context(
        data_root,
        drive_id,
        cam_id=image_cam,
        dtm_path=dtm_path,
        frame_id_for_size=f"{frame_start:010d}",
    )

    frames_dir = run_dir / "frames"
    merged_dir = run_dir / "merged"
    tables_dir = run_dir / "tables"
    images_dir = run_dir / "images"
    overlays_dir = run_dir / "overlays"
    for d in (frames_dir, merged_dir, tables_dir, images_dir, overlays_dir):
        d.mkdir(parents=True, exist_ok=True)

    simplify_cfg = cfg.get("simplify") or {}
    merge_cfg = cfg.get("merge") or {}
    roundtrip_cfg = cfg.get("roundtrip") or {}

    per_frame_rows = []
    roundtrip_rows = []
    qa_overlays = []
    qa_overlays_canonical = []
    simplified_items = []
    canonical_items = []
    raw_union_items = []
    reason_counter = Counter()
    qa_roundtrip_p90 = {}
    qa_canonical_iou = {}
    mask_area_by_frame: Dict[str, float] = {}
    pixel_present_frames: List[str] = []
    resize_notes: List[str] = []
    frames_ok_canonical = 0

    frame_list = []
    if args.frames:
        for part in str(args.frames).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                frame_list.append(int(part))
            except ValueError:
                raise SystemExit(f"invalid frame id: {part}")
    frame_set = set(frame_list) if frame_list else None

    for frame in range(frame_start, frame_end + 1):
        if frame_set is not None and frame not in frame_set:
            continue
        frame_id = f"{frame:010d}"
        try:
            frame_dir = frames_dir / frame_id
            frame_dir.mkdir(parents=True, exist_ok=True)
    
            mask_path = mask_dir / f"frame_{frame_id}.png"
            if not mask_path.exists():
                per_frame_rows.append(
                    {
                        "frame_id": frame_id,
                        "mask_area_px": 0.0,
                        "status": "not_crosswalk",
                        "reason": "mask_missing",
                        "has_pixel": 0,
                        "has_world_raw": 0,
                        "has_world_simplified": 0,
                        "mask_area_px_resized": 0.0,
                        "filter_stage_used": "",
                        "valid_world_pts": 0,
                        "dtm_hit_ratio": 0.0,
                        "valid_ratio": 0.0,
                    }
                )
                reason_counter["mask_missing"] += 1
                continue
    
            mask = _load_mask(mask_path)
            img_path = _find_frame_path(image_dir, frame_id)
            if img_path is None or not img_path.exists():
                per_frame_rows.append(
                    {
                        "frame_id": frame_id,
                        "mask_area_px": 0.0,
                        "status": "not_crosswalk",
                        "reason": "image_missing",
                        "has_pixel": 0,
                        "has_world_raw": 0,
                        "has_world_simplified": 0,
                        "mask_area_px_resized": 0.0,
                        "filter_stage_used": "",
                        "valid_world_pts": 0,
                        "dtm_hit_ratio": 0.0,
                        "valid_ratio": 0.0,
                    }
                )
                reason_counter["image_missing"] += 1
                continue
            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size
            before_shape = (mask.shape[1], mask.shape[0])
            mask = _resize_mask(mask, (img_w, img_h))
            if before_shape != (img_w, img_h):
                resize_notes.append(f"frame {frame_id}: mask resized from {before_shape} to {(img_w, img_h)}")
            offset_x, offset_y = 0.0, 0.0
            unknown_crop = True
            area_px = float(np.sum(mask))
            mask_area_by_frame[frame_id] = area_px
            min_area_px = float(back_cfg.get("min_area_px", 200))
            if area_px < min_area_px:
                per_frame_rows.append(
                    {
                        "frame_id": frame_id,
                        "mask_area_px": round(area_px, 2),
                        "status": "not_crosswalk",
                        "reason": "not_crosswalk",
                        "has_pixel": 0,
                        "has_world_raw": 0,
                        "has_world_simplified": 0,
                        "mask_area_px_resized": round(area_px, 2),
                        "filter_stage_used": "",
                        "valid_world_pts": 0,
                        "dtm_hit_ratio": 0.0,
                        "valid_ratio": 0.0,
                    }
                )
                reason_counter["not_crosswalk"] += 1
                continue
            pixel_present_frames.append(frame_id)
    
            contours = _extract_contours(mask)
            stripes = _extract_stripes(mask)
            stripe_count = len(stripes)
            if not contours:
                write_json(
                    frame_dir / "landing_debug.json",
                    {
                        "frame_id": frame_id,
                        "pixel_points": 0,
                        "valid_world_pts": 0,
                        "valid_ratio": 0.0,
                        "dtm_miss": 0,
                        "dtm_hit_ratio": 0.0,
                        "offset_x": offset_x,
                        "offset_y": offset_y,
                        "unknown_crop": unknown_crop,
                        "reason": "no_contour",
                    },
                )
                per_frame_rows.append(
                    {
                        "frame_id": frame_id,
                        "mask_area_px": round(area_px, 2),
                        "status": "backproject_failed",
                        "reason": "no_contour",
                        "has_pixel": 1,
                        "has_world_raw": 0,
                        "has_world_simplified": 0,
                        "mask_area_px_resized": round(area_px, 2),
                        "filter_stage_used": "",
                        "valid_world_pts": 0,
                        "dtm_hit_ratio": 0.0,
                        "valid_ratio": 0.0,
                    }
                )
                reason_counter["backproject_failed"] += 1
                continue
    
            img_h, img_w = mask.shape[:2]
            sampled_contours, sampled_pts, filter_stage = _filter_with_fallback(
                contours,
                int(back_cfg.get("contour_sample_step_px", 2)),
                img_w,
                img_h,
                float(back_cfg.get("pixel_use_bottom_crop", 0.45)),
                tuple(back_cfg.get("pixel_use_side_crop", [0.05, 0.95])),
                float(back_cfg.get("pixel_use_bottom_crop_fallback", 0.30)),
                tuple(back_cfg.get("pixel_use_side_crop_fallback", [0.02, 0.98])),
                int(back_cfg.get("min_valid_world_pts", 60)),
            )
    
            if not sampled_pts:
                write_json(
                    frame_dir / "landing_debug.json",
                    {
                        "frame_id": frame_id,
                        "pixel_points": 0,
                        "valid_world_pts": 0,
                        "valid_ratio": 0.0,
                        "dtm_miss": 0,
                        "dtm_hit_ratio": 0.0,
                        "offset_x": offset_x,
                        "offset_y": offset_y,
                        "unknown_crop": unknown_crop,
                        "reason": "no_sampled_points",
                    },
                )
                per_frame_rows.append(
                    {
                        "frame_id": frame_id,
                        "mask_area_px": round(area_px, 2),
                        "status": "backproject_failed",
                        "reason": "no_sampled_points",
                        "has_pixel": 1,
                        "has_world_raw": 0,
                        "has_world_simplified": 0,
                        "mask_area_px_resized": round(area_px, 2),
                        "filter_stage_used": filter_stage,
                        "valid_world_pts": 0,
                        "dtm_hit_ratio": 0.0,
                        "valid_ratio": 0.0,
                    }
                )
                reason_counter["backproject_failed"] += 1
                continue
    
            world_contours_xy = []
            world_contours_xyz = []
            valid_pts = 0
            dtm_miss = 0
            dtm_hit = 0
            camera_height = float(back_cfg.get("camera_height_m", 1.6))
            origin_z = float(ctx.pose_provider.get_t_w_c0(str(frame_id))[:3, 3][2])
            cam_z0 = origin_z - camera_height
            if dtm_median is not None and float(dtm_median) < origin_z:
                z0_use = float(dtm_median)
            else:
                z0_use = cam_z0
            for contour in sampled_contours:
                world_pts_xy = []
                world_pts_xyz = []
                for u, v in contour:
                    u_off = float(u) + offset_x
                    v_off = float(v) + offset_y
                    pt, reason = _backproject_point(
                        frame_id,
                        u_off,
                        v_off,
                        ctx,
                        use_dtm,
                        int(back_cfg.get("dtm_iterations", 2)),
                        z0_use,
                        float(back_cfg.get("max_ray_t_m", 200.0)),
                    )
                    if pt is None:
                        if reason == "dtm_nodata":
                            dtm_miss += 1
                        continue
                    if use_dtm and ctx.dtm is not None:
                        if _sample_dtm(ctx, frame_id, pt[0], pt[1], pt[2]) is not None:
                            dtm_hit += 1
                    valid_pts += 1
                    world_pts_xy.append(pt[:2])
                    world_pts_xyz.append(pt[:3])
                if len(world_pts_xy) >= 3:
                    world_contours_xy.append(np.array(world_pts_xy, dtype=float))
                    world_contours_xyz.append(np.array(world_pts_xyz, dtype=float))
    
            valid_ratio = valid_pts / max(1, len(sampled_pts))
            min_valid = int(back_cfg.get("min_valid_world_pts", 60))
            dtm_hit_ratio = (dtm_hit / max(1, valid_pts)) if use_dtm else 0.0
            if use_dtm and ctx.dtm is not None and valid_pts > 0 and dtm_hit_ratio == 0.0:
                all_xyz = np.vstack(world_contours_xyz) if world_contours_xyz else np.empty((0, 3), dtype=float)
                if all_xyz.size > 0:
                    sample_xyz = all_xyz[:: max(1, all_xyz.shape[0] // 200)]
                    pts_wu = kitti_world_to_utm32(sample_xyz, ctx.data_root, ctx.drive_id, frame_id)
                    hits = 0
                    for x, y, _ in pts_wu:
                        try:
                            val = next(ctx.dtm.sample([(float(x), float(y))]))
                        except Exception:
                            val = None
                        z = float(val[0]) if val is not None and len(val) > 0 else None
                        if z is None:
                            continue
                        if ctx.dtm_nodata is not None and np.isfinite(ctx.dtm_nodata):
                            if abs(float(z) - float(ctx.dtm_nodata)) < 1e-6:
                                continue
                        if not np.isfinite(float(z)):
                            continue
                        hits += 1
                    dtm_hit_ratio = hits / max(1, len(pts_wu))
            if valid_pts < min_valid or not world_contours_xy:
                reason = "backproject_failed"
                if dtm_miss >= max(1, valid_pts):
                    reason = "dtm_nodata"
                per_frame_rows.append(
                    {
                        "frame_id": frame_id,
                        "mask_area_px": round(area_px, 2),
                        "status": "backproject_failed",
                        "reason": reason,
                        "has_pixel": 1,
                        "has_world_raw": 0,
                        "has_world_simplified": 0,
                        "mask_area_px_resized": round(area_px, 2),
                        "filter_stage_used": filter_stage,
                        "valid_world_pts": valid_pts,
                        "dtm_hit_ratio": round(dtm_hit_ratio, 4),
                        "valid_ratio": round(valid_ratio, 4),
                    }
                )
                reason_counter["backproject_failed"] += 1
                debug = {
                    "frame_id": frame_id,
                    "pixel_points": len(sampled_pts),
                    "valid_world_pts": valid_pts,
                    "valid_ratio": round(valid_ratio, 4),
                    "dtm_miss": dtm_miss,
                    "dtm_hit_ratio": round(dtm_hit_ratio, 4),
                    "offset_x": offset_x,
                    "offset_y": offset_y,
                    "unknown_crop": unknown_crop,
                    "reason": reason,
                }
                write_json(frame_dir / "landing_debug.json", debug)
                continue
    
            world_raw_wk = _polygon_from_world_points(world_contours_xy)
            world_raw_wk = _make_valid(world_raw_wk)
            if world_raw_wk is None or world_raw_wk.is_empty:
                per_frame_rows.append(
                    {
                        "frame_id": frame_id,
                        "mask_area_px": round(area_px, 2),
                        "status": "backproject_failed",
                        "reason": "backproject_failed",
                        "has_pixel": 1,
                        "has_world_raw": 0,
                        "has_world_simplified": 0,
                        "mask_area_px_resized": round(area_px, 2),
                        "filter_stage_used": filter_stage,
                        "valid_world_pts": valid_pts,
                        "dtm_hit_ratio": round(dtm_hit_ratio, 4),
                        "valid_ratio": round(valid_ratio, 4),
                    }
                )
                reason_counter["backproject_failed"] += 1
                continue
    
            raw_z_wk = z0_use
            if world_contours_xyz:
                all_xyz = np.vstack(world_contours_xyz)
                if all_xyz.size > 0:
                    raw_z_wk = float(np.median(all_xyz[:, 2]))
            world_raw_utm = _poly_wk_to_wu(world_raw_wk, raw_z_wk, ctx, frame_id)
            if world_raw_utm is None or world_raw_utm.is_empty:
                per_frame_rows.append(
                    {
                        "frame_id": frame_id,
                        "mask_area_px": round(area_px, 2),
                        "status": "backproject_failed",
                        "reason": "utm_convert_failed",
                        "has_pixel": 1,
                        "has_world_raw": 0,
                        "has_world_simplified": 0,
                        "mask_area_px_resized": round(area_px, 2),
                        "filter_stage_used": filter_stage,
                        "valid_world_pts": valid_pts,
                        "dtm_hit_ratio": round(dtm_hit_ratio, 4),
                        "valid_ratio": round(valid_ratio, 4),
                    }
                )
                reason_counter["backproject_failed"] += 1
                continue
    
            world_simpl_wk = None
            world_simpl_utm = None
    
            stripe_world_pts = []
            stripes_wu_geoms = []
            if stripes:
                try:
                    from shapely.geometry import LineString
                except Exception:
                    LineString = None
                for s in stripes:
                    angle = float(s.get("angle_deg", 0.0))
                    short_px = float(s.get("short_px", 0.0))
                    theta = np.deg2rad(angle)
                    nx = -np.sin(theta)
                    ny = np.cos(theta)
                    cx = float(s["cx"])
                    cy = float(s["cy"])
                    off = 0.5 * short_px
                    u3 = cx + nx * off
                    v3 = cy + ny * off
                    u4 = cx - nx * off
                    v4 = cy - ny * off
                    p1, _ = _backproject_point(
                        frame_id,
                        float(s["u1"]) + offset_x,
                        float(s["v1"]) + offset_y,
                        ctx,
                        use_dtm,
                        int(back_cfg.get("dtm_iterations", 2)),
                        z0_use,
                        float(back_cfg.get("max_ray_t_m", 200.0)),
                    )
                    p2, _ = _backproject_point(
                        frame_id,
                        float(s["u2"]) + offset_x,
                        float(s["v2"]) + offset_y,
                        ctx,
                        use_dtm,
                        int(back_cfg.get("dtm_iterations", 2)),
                        z0_use,
                        float(back_cfg.get("max_ray_t_m", 200.0)),
                    )
                    pc, _ = _backproject_point(
                        frame_id,
                        float(s["cx"]) + offset_x,
                        float(s["cy"]) + offset_y,
                        ctx,
                        use_dtm,
                        int(back_cfg.get("dtm_iterations", 2)),
                        z0_use,
                        float(back_cfg.get("max_ray_t_m", 200.0)),
                    )
                    p3, _ = _backproject_point(
                        frame_id,
                        float(u3) + offset_x,
                        float(v3) + offset_y,
                        ctx,
                        use_dtm,
                        int(back_cfg.get("dtm_iterations", 2)),
                        z0_use,
                        float(back_cfg.get("max_ray_t_m", 200.0)),
                    )
                    p4, _ = _backproject_point(
                        frame_id,
                        float(u4) + offset_x,
                        float(v4) + offset_y,
                        ctx,
                        use_dtm,
                        int(back_cfg.get("dtm_iterations", 2)),
                        z0_use,
                        float(back_cfg.get("max_ray_t_m", 200.0)),
                    )
                    for pt in (p1, p2, pc, p3, p4):
                        if pt is not None:
                            stripe_world_pts.append(pt[:3])
                    if LineString is not None and p1 is not None and p2 is not None:
                        pts_wu = kitti_world_to_utm32(np.array([p1, p2], dtype=np.float64), ctx.data_root, ctx.drive_id, frame_id)
                        stripes_wu_geoms.append(LineString([(float(pts_wu[0, 0]), float(pts_wu[0, 1])), (float(pts_wu[1, 0]), float(pts_wu[1, 1]))]))
    
            canonical_status = "ok"
            canonical_area = 0.0
            canonical_z_wk = raw_z_wk
            world_canonical_utm = None
            world_canonical_wk = None
            canonical_center = None
            canonical_angle = None
            canonical_length = None
            canonical_width = None
            try:
                raw_union_wk = unary_union(world_raw_wk)
            except Exception:
                raw_union_wk = world_raw_wk
            try:
                raw_union_utm = unary_union(world_raw_utm)
            except Exception:
                raw_union_utm = world_raw_utm
            raw_union_wk = _make_valid(raw_union_wk)
            raw_union_utm = _make_valid(raw_union_utm)
            if raw_union_wk is None or raw_union_wk.is_empty:
                canonical_status = "not_crosswalk"
            else:
                try:
                    rect_wk = raw_union_wk.minimum_rotated_rectangle
                    canonical_v3_wk = rect_wk.buffer(CANONICAL_V3_MARGIN_M).buffer(-CANONICAL_V3_MARGIN_M)
                    canonical_area = float(canonical_v3_wk.area) if canonical_v3_wk is not None else 0.0
                    if canonical_v3_wk is None or canonical_v3_wk.is_empty or canonical_area < CANONICAL_V3_MIN_AREA_M2:
                        canonical_status = "canonical_invalid"
                    else:
                        world_canonical_wk = canonical_v3_wk
                        canonical_center, canonical_angle, canonical_length, canonical_width = _rect_params(rect_wk)
                        world_canonical_utm = _poly_wk_to_wu(canonical_v3_wk, canonical_z_wk, ctx, frame_id)
                        if world_canonical_utm is None or world_canonical_utm.is_empty:
                            canonical_status = "canonical_invalid"
                except Exception:
                    canonical_status = "canonical_invalid"
    
            world_simpl_wk = world_canonical_wk
            world_simpl_utm = world_canonical_utm
            bbox_ok_raw, bbox_raw = _check_utm_bbox(raw_union_utm)
            if not bbox_ok_raw:
                write_json(
                    run_dir / "decision.json",
                    {
                        "status": "FAIL",
                        "reason": "utm_bbox_invalid",
                        "frame_id": frame_id,
                        "bbox_utm32": bbox_raw,
                    },
                )
                raise SystemExit("utm_bbox_invalid")
            bbox_ok_can, bbox_can = _check_utm_bbox(world_canonical_utm) if world_canonical_utm is not None else (True, {})
            if not bbox_ok_can:
                write_json(
                    run_dir / "decision.json",
                    {
                        "status": "FAIL",
                        "reason": "utm_bbox_invalid_canonical",
                        "frame_id": frame_id,
                        "bbox_utm32": bbox_can,
                    },
                )
                raise SystemExit("utm_bbox_invalid_canonical")
    
            if use_dtm and ctx.dtm is not None and dtm_hit_ratio == 0.0:
                try:
                    minx, miny, maxx, maxy = raw_union_utm.bounds
                    cx = 0.5 * (minx + maxx)
                    cy = 0.5 * (miny + maxy)
                    val = next(ctx.dtm.sample([(float(cx), float(cy))]))
                    z = float(val[0]) if val is not None and len(val) > 0 else None
                    if z is not None:
                        if ctx.dtm_nodata is not None and np.isfinite(ctx.dtm_nodata):
                            if abs(float(z) - float(ctx.dtm_nodata)) >= 1e-6 and np.isfinite(float(z)):
                                dtm_hit_ratio = 1.0
                        elif np.isfinite(float(z)):
                            dtm_hit_ratio = 1.0
                except Exception:
                    pass
    
            try:
                import geopandas as gpd
    
                gdf_raw_utm = gpd.GeoDataFrame(
                    [{"frame_id": frame_id, "kind": "raw", "geometry": raw_union_utm, "coord_space": "utm32"}],
                    geometry="geometry",
                    crs="EPSG:32632",
                )
                gdf_raw_wk = gpd.GeoDataFrame(
                    [{"frame_id": frame_id, "kind": "raw", "geometry": world_raw_wk, "coord_space": "wk"}],
                    geometry="geometry",
                )
                gdf_canonical_utm = gpd.GeoDataFrame(
                    [{"frame_id": frame_id, "kind": "canonical", "geometry": world_canonical_utm, "coord_space": "utm32"}],
                    geometry="geometry",
                    crs="EPSG:32632",
                )
                gdf_canonical_wk = gpd.GeoDataFrame(
                    [{"frame_id": frame_id, "kind": "canonical", "geometry": world_canonical_wk, "coord_space": "wk"}],
                    geometry="geometry",
                )
                gdf_simpl_utm = gpd.GeoDataFrame(
                    [{"frame_id": frame_id, "kind": "canonical_v3", "geometry": world_simpl_utm, "coord_space": "utm32"}],
                    geometry="geometry",
                    crs="EPSG:32632",
                )
                gdf_simpl_wk = gpd.GeoDataFrame(
                    [{"frame_id": frame_id, "kind": "simplified", "geometry": world_simpl_wk, "coord_space": "wk"}],
                    geometry="geometry",
                )
                gpkg_utm_path = frame_dir / "world_crosswalk_utm32.gpkg"
                gdf_raw_utm.to_file(gpkg_utm_path, layer="crosswalk_frame_raw", driver="GPKG")
                if world_simpl_utm is not None and not world_simpl_utm.is_empty:
                    gdf_simpl_utm.to_file(gpkg_utm_path, layer="crosswalk_frame_canonical_v3", driver="GPKG")
                if stripes_wu_geoms:
                    gdf_stripes = gpd.GeoDataFrame(
                        [{"frame_id": frame_id, "geometry": g, "coord_space": "utm32"} for g in stripes_wu_geoms],
                        geometry="geometry",
                        crs="EPSG:32632",
                    )
                    gdf_stripes.to_file(gpkg_utm_path, layer="stripes_world_utm32", driver="GPKG")
    
                gpkg_wk_path = frame_dir / "world_crosswalk_wk.gpkg"
                gdf_raw_wk.to_file(gpkg_wk_path, layer="crosswalk_frame_raw_wk", driver="GPKG")
                if world_canonical_wk is not None and not world_canonical_wk.is_empty:
                    gdf_canonical_wk.to_file(gpkg_wk_path, layer="crosswalk_frame_canonical_wk", driver="GPKG")
            except Exception as exc:
                write_json(frame_dir / "landing_debug.json", {"frame_id": frame_id, "reason": f"gpkg_write_failed:{exc}"})
                per_frame_rows.append(
                    {
                        "frame_id": frame_id,
                        "mask_area_px": round(area_px, 2),
                        "status": "backproject_failed",
                        "reason": "gpkg_write_failed",
                        "has_pixel": 1,
                        "has_world_raw": 0,
                        "has_world_simplified": 0,
                        "mask_area_px_resized": round(area_px, 2),
                        "filter_stage_used": filter_stage,
                        "valid_world_pts": valid_pts,
                        "dtm_hit_ratio": round(dtm_hit_ratio, 4),
                        "valid_ratio": round(valid_ratio, 4),
                    }
                )
                reason_counter["backproject_failed"] += 1
                continue
    
            debug = {
                "frame_id": frame_id,
                "pixel_points": len(sampled_pts),
                "valid_world_pts": valid_pts,
                "valid_ratio": round(valid_ratio, 4),
                "dtm_miss": dtm_miss,
                "dtm_hit_ratio": round(dtm_hit_ratio, 4),
                "offset_x": offset_x,
                "offset_y": offset_y,
                "unknown_crop": unknown_crop,
                "coord_space_written": "utm32",
                "bbox_utm32": bbox_raw,
                "bbox_sanity_ok": bool(bbox_ok_raw),
                "stripe_count": stripe_count,
                "canonical_status": canonical_status,
                "canonical_area_m2": round(canonical_area, 3),
                "canonical_z_wk": round(float(canonical_z_wk), 3),
                "reason": "ok",
            }
            write_json(frame_dir / "landing_debug.json", debug)
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": round(area_px, 2),
                    "status": "ok",
                    "reason": "ok",
                    "has_pixel": 1,
                    "has_world_raw": 1,
                    "has_world_simplified": 1 if world_simpl_utm is not None and not world_simpl_utm.is_empty else 0,
                    "mask_area_px_resized": round(area_px, 2),
                    "filter_stage_used": filter_stage,
                    "valid_world_pts": valid_pts,
                    "dtm_hit_ratio": round(dtm_hit_ratio, 4),
                    "valid_ratio": round(valid_ratio, 4),
                    "stripe_count": stripe_count,
                    "canonical_status": canonical_status,
                    "canonical_area_m2": round(canonical_area, 3),
                }
            )
            reason_counter["ok"] += 1
    
            if world_canonical_utm is not None and not world_canonical_utm.is_empty and canonical_status == "ok":
                frames_ok_canonical += 1
                canonical_items.append(
                    {
                        "frame_id": frame_id,
                        "geometry": world_canonical_utm,
                        "center": canonical_center,
                        "angle_deg": canonical_angle,
                        "length_m": canonical_length,
                        "width_m": canonical_width,
                    }
                )
                raw_union_items.append(
                    {
                        "frame_id": frame_id,
                        "geometry_utm": raw_union_utm,
                        "geometry_wk": raw_union_wk,
                        "raw_z_wk": raw_z_wk,
                    }
                )
    
            if img_path and img_path.exists():
                pixel_contour = _pixel_contour_path(contours)
                if pixel_contour is not None:
                    _draw_pixel_contour(img, pixel_contour, frame_dir / "pixel_contour.png")
    
            step_px = int(back_cfg.get("contour_sample_step_px", 2))
            pixel_contours = []
            for c in contours:
                if c is None or len(c) < 3:
                    continue
                pixel_contours.append(c[:: max(1, step_px)].copy())
            if not pixel_contours:
                roundtrip_rows.append(
                    {
                        "frame_id": frame_id,
                        "p50": "",
                        "p90": "",
                        "valid_ratio": 0.0,
                        "reason": "missing_mask",
                    }
                )
                continue
    
            total_pts = sum(c.shape[0] for c in world_contours_xyz)
            if total_pts <= 0:
                roundtrip_rows.append(
                    {
                        "frame_id": frame_id,
                        "p50": "",
                        "p90": "",
                        "valid_ratio": 0.0,
                        "reason": "missing_world_raw",
                    }
                )
                continue
    
            target_n = int(roundtrip_cfg.get("roundtrip_sample_pts", 200))
            stride = max(1, total_pts // max(1, target_n))
            pts_wk = []
            for contour in world_contours_xyz:
                if contour.size == 0:
                    continue
                pts_wk.append(contour[::stride])
            pts_wk = np.concatenate(pts_wk, axis=0) if pts_wk else np.empty((0, 3), dtype=float)
            if pts_wk.size == 0:
                roundtrip_rows.append(
                    {
                        "frame_id": frame_id,
                        "p50": "",
                        "p90": "",
                        "valid_ratio": 0.0,
                        "reason": "missing_world_raw",
                    }
                )
                continue
            pts_wu = kitti_world_to_utm32(pts_wk, ctx.data_root, ctx.drive_id, frame_id)
            xyz = utm32_to_kitti_world(pts_wu, ctx.data_root, ctx.drive_id, frame_id)
            u, v, valid = world_to_pixel_cam0(frame_id, xyz, ctx=ctx)
            valid_pts = [(float(uu), float(vv)) for uu, vv, ok in zip(u, v, valid) if ok]
            valid_ratio = len(valid_pts) / max(1, len(xyz))
            if valid_ratio < float(roundtrip_cfg.get("roundtrip_valid_ratio_min", 0.3)):
                roundtrip_rows.append(
                    {
                        "frame_id": frame_id,
                        "p50": "",
                        "p90": "",
                        "valid_ratio": round(valid_ratio, 4),
                        "reason": "low_valid",
                    }
                )
                continue
    
            from shapely.geometry import LineString, Point, MultiLineString
    
            lines = [LineString([(float(u0), float(v0)) for u0, v0 in c]) for c in pixel_contours if len(c) >= 2]
            pixel_line = MultiLineString(lines) if len(lines) > 1 else (lines[0] if lines else None)
            if pixel_line is None:
                roundtrip_rows.append(
                    {
                        "frame_id": frame_id,
                        "p50": "",
                        "p90": "",
                        "valid_ratio": round(valid_ratio, 4),
                        "reason": "missing_mask",
                    }
                )
                continue
            dists = []
            for uu, vv in valid_pts:
                dists.append(float(pixel_line.distance(Point(uu, vv))))
    
            p50 = float(np.percentile(dists, 50)) if dists else None
            p90 = float(np.percentile(dists, 90)) if dists else None
            roundtrip_rows.append(
                {
                    "frame_id": frame_id,
                    "p50": p50 if p50 is not None else "",
                    "p90": p90 if p90 is not None else "",
                    "valid_ratio": round(valid_ratio, 4),
                    "reason": "ok",
                }
            )
    
            if int(frame_id) in set(int(x) for x in qa_frames):
                if img_path and img_path.exists():
                    overlay_path = frame_dir / f"frame_{frame_id}.png"
                    overlay_canonical_path = overlays_dir / f"frame_{frame_id}_overlay_canonical.png"
                    base = Image.open(img_path).convert("RGB")
                    draw = ImageDraw.Draw(base)
                    for c in pixel_contours:
                        if len(c) >= 2:
                            draw.line([tuple(p) for p in c] + [tuple(c[0])], fill=(255, 0, 0), width=2)
                    if len(valid_pts) >= 2:
                        draw.line(valid_pts + [valid_pts[0]], fill=(0, 255, 0), width=2)
                    canonical_pts, _ = _project_polygon_wk_to_pixel(frame_id, world_canonical_wk, ctx, canonical_z_wk)
                    if len(canonical_pts) >= 3:
                        draw.line(canonical_pts + [canonical_pts[0]], fill=(0, 120, 255), width=2)
                    base.save(overlay_canonical_path)
                    qa_overlays.append((frame_id, overlay_path))
                    qa_overlays_canonical.append((frame_id, overlay_canonical_path))
                    if p90 is not None:
                        qa_roundtrip_p90[frame_id] = p90
                    iou = _polygon_iou(mask, canonical_pts) if canonical_pts else 0.0
                    qa_canonical_iou[frame_id] = iou
    
        except Exception as exc:
            per_frame_rows.append(
                {
                    "frame_id": frame_id,
                    "mask_area_px": 0.0,
                    "status": "backproject_failed",
                    "reason": f"exception:{type(exc).__name__}",
                    "has_pixel": 0,
                    "has_world_raw": 0,
                    "has_world_simplified": 0,
                    "mask_area_px_resized": 0.0,
                    "filter_stage_used": "exception",
                    "valid_world_pts": 0,
                    "dtm_hit_ratio": 0.0,
                    "valid_ratio": 0.0,
                    "p50": "",
                    "p90": "",
                }
            )
            reason_counter[f"exception:{type(exc).__name__}"] += 1
            continue
    write_csv(
        tables_dir / "per_frame_landing.csv",
        per_frame_rows,
        [
            "frame_id",
            "mask_area_px",
            "status",
            "reason",
            "has_pixel",
            "has_world_raw",
            "has_world_simplified",
            "mask_area_px_resized",
            "filter_stage_used",
            "valid_world_pts",
            "dtm_hit_ratio",
            "valid_ratio",
            "stripe_count",
            "canonical_status",
            "canonical_area_m2",
        ],
    )

    write_csv(
        tables_dir / "roundtrip_px_errors.csv",
        roundtrip_rows,
        ["frame_id", "p50", "p90", "valid_ratio", "reason"],
    )

    _montage(qa_overlays, images_dir / "qa_montage_roundtrip.png")
    _montage(qa_overlays_canonical, images_dir / "qa_montage_canonical.png")
    _pie_chart(reason_counter, images_dir / "landing_reason_summary.png")

    try:
        raw_union_items_disk = _load_raw_union_items_from_frames(frames_dir)
        merge_debug_source = "memory"
        if raw_union_items_disk:
            raw_union_items = raw_union_items_disk
            merge_debug_source = "disk"

        merged_items, merged_raw_items, canonical_metrics, merge_debug = _merge_from_raw_union_items(
            raw_union_items, ctx, merge_cfg, frame_start
        )
        merge_debug["source"] = merge_debug_source

        merged_items_utm = []
        merged_items_wk = []
        merged_bbox_report = []
        bbox_bad = False
        for item in merged_items:
            geom_utm = item.get("geometry")
            if geom_utm is None or geom_utm.is_empty:
                continue
            frame_ref = str(item.get("frame_id_ref") or f"{frame_start:010d}")
            item_utm = dict(item)
            item_utm["coord_space"] = "utm32"
            merged_items_utm.append(item_utm)
            geom_wk = _poly_wu_to_wk(geom_utm, 0.0, ctx, frame_ref)
            if geom_wk is not None and not geom_wk.is_empty:
                item_wk = dict(item)
                item_wk["geometry"] = geom_wk
                item_wk["coord_space"] = "wk"
                merged_items_wk.append(item_wk)

            minx, miny, maxx, maxy = geom_utm.bounds
            bbox = {"e_min": float(minx), "e_max": float(maxx), "n_min": float(miny), "n_max": float(maxy)}
            merged_bbox_report.append(
                {
                    "candidate_id": item["candidate_id"],
                    "bbox_utm32": bbox,
                    "center_utm32": [float(item_utm.get("center_x", 0.0)), float(item_utm.get("center_y", 0.0))],
                    "support_frames": int(item.get("support_frames", 0)),
                    "frame_id_ref": frame_ref,
                    "angle_deg": float(item.get("angle_deg", 0.0)),
                    "length_m": float(item.get("length_m", 0.0)),
                    "width_m": float(item.get("width_m", 0.0)),
                }
            )
            if not (100000 <= minx <= 900000 and 100000 <= maxx <= 900000):
                bbox_bad = True
            if not (1000000 <= miny <= 9000000 and 1000000 <= maxy <= 9000000):
                bbox_bad = True

        if bbox_bad:
            write_json(
                run_dir / "decision.json",
                {
                    "status": "FAIL",
                    "reason": "utm_bbox_invalid_merged",
                    "merged_bbox": merged_bbox_report,
                },
            )
            return 0
    except Exception as exc:
        write_json(
            run_dir / "decision.json",
            {
                "status": "FAIL",
                "reason": "merge_failed",
                "error": str(exc),
            },
        )
        return 0

    try:
        import geopandas as gpd

        if merged_items_utm:
            gdf = gpd.GeoDataFrame(merged_items_utm, geometry="geometry", crs="EPSG:32632")
        else:
            gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:32632")
        gdf.to_file(merged_dir / "crosswalk_candidates_canonical_utm32.gpkg", layer="crosswalk_candidates", driver="GPKG")
        gdf.to_file(merged_dir / "crosswalk_candidates_utm32.gpkg", layer="crosswalk_candidates", driver="GPKG")

        if merged_raw_items:
            gdf_raw = gpd.GeoDataFrame(merged_raw_items, geometry="geometry", crs="EPSG:32632")
        else:
            gdf_raw = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:32632")
        gdf_raw.to_file(merged_dir / "crosswalk_candidates_raw_utm32.gpkg", layer="crosswalk_candidates", driver="GPKG")

        if merged_items_wk:
            gdf_wk = gpd.GeoDataFrame(merged_items_wk, geometry="geometry")
        else:
            gdf_wk = gpd.GeoDataFrame({"geometry": []}, geometry="geometry")
        gdf_wk.to_file(merged_dir / "crosswalk_candidates_canonical_wk.gpkg", layer="crosswalk_candidates", driver="GPKG")
    except Exception as exc:
        write_json(merged_dir / "merge_failed.json", {"reason": str(exc)})

    merge_rows = [
        {
            "candidate_id": item["candidate_id"],
            "support_count": item["support_frames"],
            "area_m2": float(item["geometry"].area) if item.get("geometry") is not None else 0.0,
            "center_x": item.get("center_x", 0.0),
            "center_y": item.get("center_y", 0.0),
            "angle_deg": item.get("angle_deg", 0.0),
            "length_m": item.get("length_m", 0.0),
            "width_m": item.get("width_m", 0.0),
        }
        for item in merged_items_utm
    ]
    write_csv(
        merged_dir / "merge_stats.csv",
        merge_rows,
        ["candidate_id", "support_count", "area_m2", "center_x", "center_y", "angle_deg", "length_m", "width_m"],
    )
    write_csv(
        tables_dir / "canonical_vs_raw_metrics.csv",
        canonical_metrics,
        ["candidate_id", "support_frames", "dist_center_m", "iou", "coverage_raw"],
    )

    total_frames = frame_end - frame_start + 1
    frames_not_crosswalk = sum(1 for r in per_frame_rows if r["status"] == "not_crosswalk")
    frames_ok = sum(1 for r in per_frame_rows if r["status"] == "ok")
    frames_backproject_failed = sum(1 for r in per_frame_rows if r["status"] == "backproject_failed")
    n_pixel_present = len(pixel_present_frames)
    n_world_ok = frames_ok
    n_backproject_failed = max(0, n_pixel_present - n_world_ok)

    cycle_p90 = None
    cycle_rows = []
    cycle_hist_path = images_dir / "pixel_cycle_hist.png"
    qa_target = "0000000290"
    if qa_target not in set(pixel_present_frames) and pixel_present_frames:
        qa_target = max(pixel_present_frames, key=lambda fid: mask_area_by_frame.get(fid, 0.0))

    if qa_target in set(pixel_present_frames):
        rng = random.Random(20260130)
        mask_path = mask_dir / f"frame_{qa_target}.png"
        mask = _load_mask(mask_path)
        img_path = _find_frame_path(image_dir, qa_target)
        if img_path and img_path.exists():
            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size
            mask = _resize_mask(mask, (img_w, img_h))
            ys, xs = np.where(mask)
            if len(xs) > 0:
                idxs = rng.sample(range(len(xs)), min(500, len(xs)))
                pts = [(float(xs[i]), float(ys[i])) for i in idxs]
                reproj_pts = []
                dists = []
                camera_height = float(back_cfg.get("camera_height_m", 1.6))
                cam_z0 = float(ctx.pose_provider.get_t_w_c0(str(qa_target))[:3, 3][2]) - camera_height
                z0_use = float(dtm_median) if dtm_median is not None else cam_z0
                for u, v in pts:
                    pt, _ = _backproject_point(
                        qa_target,
                        float(u),
                        float(v),
                        ctx,
                        use_dtm,
                        int(back_cfg.get("dtm_iterations", 2)),
                        z0_use,
                        float(back_cfg.get("max_ray_t_m", 200.0)),
                    )
                    if pt is None:
                        continue
                    xyz = np.array([[pt[0], pt[1], pt[2]]], dtype=float)
                    uu, vv, valid = world_to_pixel_cam0(qa_target, xyz, ctx=ctx)
                    if len(valid) == 0 or not bool(valid[0]):
                        continue
                    du = float(abs(uu[0] - u))
                    dv = float(abs(vv[0] - v))
                    d = float(np.hypot(du, dv))
                    dists.append(d)
                    reproj_pts.append((float(uu[0]), float(vv[0])))
                if dists:
                    cycle_p90 = float(np.percentile(dists, 90))
                    cycle_p50 = float(np.percentile(dists, 50))
                    cycle_rows.append(
                        {
                            "frame_id": qa_target,
                            "cycle_p50": round(cycle_p50, 3),
                            "cycle_p90": round(cycle_p90, 3),
                            "n_samples": len(dists),
                        }
                    )
                    try:
                        import matplotlib.pyplot as plt

                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.hist(dists, bins=30)
                        ax.set_title(f"pixel_cycle p90={cycle_p90:.2f}")
                        fig.tight_layout()
                        fig.savefig(cycle_hist_path)
                        plt.close(fig)
                    except Exception:
                        pass
                    cycle_overlay_path = frames_dir / qa_target / f"frame_{qa_target}_cycle_overlay.png"
                    _draw_roundtrip_overlay(img, np.array(pts), reproj_pts, cycle_overlay_path)

    write_csv(tables_dir / "pixel_cycle.csv", cycle_rows, ["frame_id", "cycle_p50", "cycle_p90", "n_samples"])

    status = "WARN"
    ok_rate = (n_world_ok / max(1, n_pixel_present)) if n_pixel_present > 0 else 0.0
    qa_roundtrip_p90 = {str(k): float(v) for k, v in qa_roundtrip_p90.items() if v is not None}
    qa_canonical_iou = {str(k): float(v) for k, v in qa_canonical_iou.items() if v is not None}
    qa_p90s = [v for v in qa_roundtrip_p90.values() if v is not None]
    qa_p90_max = max(qa_p90s) if qa_p90s else None
    qa_iou_vals = list(qa_canonical_iou.values())
    qa_iou_min = min(qa_iou_vals) if qa_iou_vals else None
    if not canonical_metrics:
        status = "FAIL"
    else:
        bad = False
        for m in canonical_metrics:
            if float(m.get("dist_center_m", 0.0)) > 1.0:
                bad = True
            if float(m.get("coverage_raw", 0.0)) < 0.6:
                bad = True
        status = "PASS" if not bad else "WARN"

    decision = {
        "status": status,
        "frames_not_crosswalk": frames_not_crosswalk,
        "frames_ok": frames_ok,
        "frames_backproject_failed": frames_backproject_failed,
        "n_pixel_present": n_pixel_present,
        "n_world_ok": n_world_ok,
        "n_backproject_failed": n_backproject_failed,
        "ok_rate": round(ok_rate, 4),
        "pixel_cycle_p90": cycle_p90,
        "roundtrip_p90_qa": qa_roundtrip_p90,
        "qa_roundtrip_p90_max": qa_p90_max,
        "frames_ok_canonical": frames_ok_canonical,
        "merged_candidate_count": len(merged_items_utm),
        "qa_canonical_iou": qa_canonical_iou,
        "qa_canonical_iou_min": qa_iou_min,
        "stage12_run": str(stage12_run),
        "dtm_path": str(dtm_path) if dtm_path else "",
    }
    write_json(run_dir / "decision.json", decision)

    resolved_cfg = dict(cfg)
    resolved_cfg.update(
        {
            "resolved": {
                "run_id": run_id,
                "data_root": str(data_root),
                "image_dir": str(image_dir),
                "stage12_run": str(stage12_run),
                "mask_dir": str(mask_dir),
                "dtm_path": str(dtm_path) if dtm_path else "",
            }
        }
    )
    import yaml

    resolved_path = run_dir / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved_cfg, sort_keys=False), encoding="utf-8")
    (run_dir / "params_hash.txt").write_text(_hash_file(resolved_path), encoding="utf-8")

    reason_top = reason_counter.most_common(3)
    ok_hit = [float(r.get("dtm_hit_ratio", 0.0)) for r in per_frame_rows if r.get("status") == "ok"]
    dtm_hit_avg = float(np.mean(ok_hit)) if ok_hit else 0.0
    dtm_hit_med = float(np.median(ok_hit)) if ok_hit else 0.0
    dtm_hit_max = float(np.max(ok_hit)) if ok_hit else 0.0
    dtm_hit_warn = bool(ok_hit and dtm_hit_max < 0.3)
    report_lines = [
        "# World Candidates from Stage2 masks (0010 f250-500)",
        "",
        f"- status: {status}",
        f"- frames_not_crosswalk: {frames_not_crosswalk}",
        f"- frames_ok: {frames_ok}",
        f"- frames_backproject_failed: {frames_backproject_failed}",
        f"- n_pixel_present: {n_pixel_present}",
        f"- n_world_ok: {n_world_ok}",
        f"- n_backproject_failed: {n_backproject_failed}",
        f"- frames_ok_canonical: {frames_ok_canonical}",
        f"- ok_rate: {ok_rate:.3f}",
        f"- pixel_cycle_p90: {cycle_p90 if cycle_p90 is not None else 'NA'}",
        f"- merged_candidate_count: {len(merged_items_utm)}",
        f"- qa_canonical_iou_min: {qa_iou_min if qa_iou_min is not None else 'NA'}",
        "",
        "## mask_resize",
        *([f"- {note}" for note in resize_notes] if resize_notes else ["- none"]),
        "",
        "## backproject_failed_top3",
        *[f"- {k}: {v}" for k, v in reason_top],
        "",
        "## roundtrip_qa",
        *[f"- {k}: p90={v:.2f}" for k, v in qa_roundtrip_p90.items()],
        "",
        "## canonical_iou_qa",
        *[f"- {k}: iou={v:.3f}" for k, v in qa_canonical_iou.items()],
        "",
        "## canonical_vs_raw_metrics",
        *[
            f"- {m['candidate_id']}: dist_center_m={m['dist_center_m']:.2f} iou={m['iou']:.2f} coverage_raw={m['coverage_raw']:.2f}"
            for m in canonical_metrics
        ],
        "",
        "## merge_debug",
        f"- raw_union_items: {merge_debug.get('raw_union_items', 0)}",
        f"- clusters: {merge_debug.get('clusters', 0)}",
        f"- clusters_ge_min_support: {merge_debug.get('clusters_ge_min_support', 0)}",
        *[
            f"- {k}: {v}"
            for k, v in (merge_debug.get("reason_counts") or {}).items()
        ],
        "",
        "## dtm_hit_ratio",
        f"- ok_frames_avg: {dtm_hit_avg:.3f}",
        f"- ok_frames_median: {dtm_hit_med:.3f}",
        f"- ok_frames_max: {dtm_hit_max:.3f}",
        *(["- warning: dtm_hit_ratio low in ok frames (check UTM XY for DTM sampling)"] if dtm_hit_warn else []),
        "",
        "## merged_canonical_utm32",
        *[
            f"- {d['candidate_id']}: bbox={d['bbox_utm32']} center={d['center_utm32']} "
            f"support={d['support_frames']} frame_ref={d.get('frame_id_ref','')} "
            f"L/W={d['length_m']:.2f}/{d['width_m']:.2f} angle={d['angle_deg']:.1f}"
            for d in merged_bbox_report
        ],
        "",
        "## outputs",
        "- merged/crosswalk_candidates_canonical_utm32.gpkg",
        "- merged/crosswalk_candidates_raw_utm32.gpkg",
        "- merged/crosswalk_candidates_utm32.gpkg",
        "- images/qa_montage_roundtrip.png",
        "- images/qa_montage_canonical.png",
        "- overlays/frame_0000000290_overlay_canonical.png",
        "- tables/per_frame_landing.csv",
        "- tables/roundtrip_px_errors.csv",
        "- tables/canonical_vs_raw_metrics.csv",
        "- tables/pixel_cycle.csv",
        "- images/pixel_cycle_hist.png",
    ]
    write_text(run_dir / "report.md", "\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from shapely.geometry import GeometryCollection, LineString, MultiPoint, MultiPolygon, Polygon, shape
from shapely.ops import unary_union

from pipeline.datasets.kitti360_io import _find_oxts_dir, load_kitti360_pose
from pipeline.post.fill_holes import fill_small_holes_by_corridor
from pipeline.post.morph_close import close_candidates_in_corridor
from scripts.pipeline_common import (
    LOG,
    bbox_polygon,
    ensure_dir,
    ensure_overwrite,
    ensure_required_columns,
    load_yaml,
    now_ts,
    relpath,
    setup_logging,
    validate_output_crs,
    write_csv,
    write_json,
    write_text,
    write_gpkg_layer,
)

PRIMITIVE_FIELDS = [
    "evid_id",
    "source",
    "drive_id",
    "evid_type",
    "crs_epsg",
    "path_rel",
    "frame_start",
    "frame_end",
    "conf",
    "uncert",
    "meta_json",
]

WORLD_FIELDS = [
    "cand_id",
    "source",
    "drive_id",
    "class",
    "crs_epsg",
    "conf",
    "uncert",
    "evid_ref",
    "conflict",
    "attr_json",
]


def _load_drive_list(cfg: dict) -> List[str]:
    drives = [str(d) for d in (cfg.get("DRIVES") or []) if str(d).strip()]
    drives_file = str(cfg.get("DRIVES_FILE") or "").strip()
    if drives_file:
        path = Path(drives_file)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    drives.append(line.strip())
    seen = []
    for d in drives:
        if d not in seen:
            seen.append(d)
    return seen


def _resolve_dop20_root(cfg: dict) -> Tuple[Optional[Path], str]:
    cfg_root = str(cfg.get("DOP20_ROOT") or "").strip()
    if cfg_root:
        return Path(cfg_root), "config"
    env_root = os.environ.get("DOP20_ROOT", "").strip()
    if env_root:
        return Path(env_root), "env"
    for base in [Path("data"), Path("datasets")]:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_dir() and "dop20" in path.name.lower():
                return path, f"auto:{base}"
    return None, "unset"


def _safe_read_gpkg(path: Path) -> Optional[gpd.GeoDataFrame]:
    if not path.exists():
        return None
    try:
        return gpd.read_file(path)
    except Exception:
        return None


def _scan_tiles(root: Path) -> List[Path]:
    tiles = []
    for ext in ("*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.jp2"):
        tiles.extend(root.rglob(ext))
    return sorted(tiles)


def _build_tiles_index(tiles: List[Path], target_epsg: int) -> Tuple[gpd.GeoDataFrame, List[str]]:
    rows = []
    errors = []
    for path in tiles:
        try:
            with rasterio.open(path) as ds:
                bounds = ds.bounds
                crs = ds.crs
        except Exception as exc:
            errors.append(f"tile_open_failed:{path}:{exc}")
            continue
        if crs is None:
            errors.append(f"tile_missing_crs_assume:{path}")
            src_epsg = target_epsg
            utm_bounds = bounds
        else:
            src_epsg = crs.to_epsg() or 0
            try:
                utm_bounds = transform_bounds(crs, f"EPSG:{target_epsg}", *bounds, densify_pts=21)
            except Exception as exc:
                errors.append(f"tile_transform_failed:{path}:{exc}")
                continue
        rows.append(
            {
                "tile_id": path.stem,
                "path": str(path),
                "src_epsg": int(src_epsg),
                "minx": float(utm_bounds[0]),
                "miny": float(utm_bounds[1]),
                "maxx": float(utm_bounds[2]),
                "maxy": float(utm_bounds[3]),
                "geometry": bbox_polygon(utm_bounds),
            }
        )
    if not rows:
        gdf = gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs=f"EPSG:{target_epsg}")
    else:
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{target_epsg}")
    return gdf, errors


def _load_tiles_index(cfg: dict, dop20_root: Optional[Path], run_dir: Path) -> Tuple[gpd.GeoDataFrame, str, List[str]]:
    target_epsg = int(cfg.get("TARGET_EPSG", 32632))
    errors: List[str] = []
    index_hint = str(cfg.get("DOP20_INDEX") or "").strip()
    candidates = []
    if index_hint:
        candidates.append(Path(index_hint))
    if dop20_root:
        candidates.append(dop20_root / "dop20_tiles_index.gpkg")
        candidates.append(dop20_root / "dop20_tiles_index.json")
    candidates.append(Path("cache") / "dop20_tiles_index.gpkg")
    candidates.append(Path("cache") / "dop20_tiles_index.json")
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix.lower() == ".gpkg":
            gdf = _safe_read_gpkg(candidate)
            if gdf is None or gdf.empty:
                continue
            if dop20_root is not None and "path" in gdf.columns:
                sample = gdf["path"].dropna().astype(str)
                if not sample.empty and not str(dop20_root).lower() in str(sample.iloc[0]).lower():
                    continue
            gdf = gdf.set_crs(f"EPSG:{target_epsg}", allow_override=True)
            return gdf, f"reuse:{candidate}", errors
        if candidate.suffix.lower() == ".json":
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, list):
                continue
            if dop20_root is not None and payload:
                first_path = str(payload[0].get("path") or "")
                if first_path and str(dop20_root).lower() not in first_path.lower():
                    continue
            rows = []
            for item in payload:
                try:
                    bbox = (item["minx"], item["miny"], item["maxx"], item["maxy"])
                except Exception:
                    continue
                rows.append(
                    {
                        "tile_id": Path(str(item.get("path") or "")).stem,
                        "path": str(item.get("path") or ""),
                        "src_epsg": int(item.get("crs_epsg") or 0),
                        "minx": float(bbox[0]),
                        "miny": float(bbox[1]),
                        "maxx": float(bbox[2]),
                        "maxy": float(bbox[3]),
                        "geometry": bbox_polygon(bbox),
                    }
                )
            gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{target_epsg}")
            if not gdf.empty:
                return gdf, f"reuse_json:{candidate}", errors
    if dop20_root is None:
        return gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs=f"EPSG:{target_epsg}"), "missing_root", errors
    tiles_dir = dop20_root / "tiles_utm32"
    tile_root = tiles_dir if tiles_dir.exists() else dop20_root
    tiles = _scan_tiles(tile_root)
    gdf, build_errors = _build_tiles_index(tiles, target_epsg)
    errors.extend(build_errors)
    return gdf, f"scan:{tile_root}", errors


def _find_corridor_gpkg(drive_id: str, roots: List[Path]) -> Optional[Path]:
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.gpkg"):
            name = path.name.lower()
            if "corridor" not in name:
                continue
            if "osm_corridor" in name:
                continue
            if drive_id in str(path):
                return path
    return None


def _find_osm_corridor_gpkg(drive_id: str, roots: List[Path]) -> Optional[Path]:
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("osm_corridor_utm32.gpkg"):
            if drive_id in str(path):
                return path
    return None


def _trajectory_roi(
    data_root: Path,
    drive_id: str,
    buffer_m: float,
    stride: int,
) -> Tuple[Optional[object], str, List[str]]:
    errors: List[str] = []
    try:
        oxts_dir = _find_oxts_dir(data_root, drive_id)
    except Exception as exc:
        errors.append(f"oxts_missing:{exc}")
        return None, "missing", errors
    frames = sorted(oxts_dir.glob("*.txt"))
    if not frames:
        errors.append("oxts_empty")
        return None, "missing", errors
    if stride > 1:
        frames = frames[::stride]
    points: List[Tuple[float, float]] = []
    for frame in frames:
        frame_id = frame.stem
        try:
            x, y, _ = load_kitti360_pose(data_root, drive_id, frame_id)
        except Exception as exc:
            errors.append(f"pose_failed:{frame_id}:{exc}")
            continue
        points.append((x, y))
    if not points:
        errors.append("pose_empty")
        return None, "missing", errors
    if len(points) == 1:
        geom = MultiPoint(points).buffer(buffer_m)
    else:
        geom = LineString(points).buffer(buffer_m)
    return geom, "trajectory", errors


def _is_wgs84_bounds(bounds: Tuple[float, float, float, float]) -> bool:
    minx, miny, maxx, maxy = bounds
    return -180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0 and -90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0


def _load_osm_roads(
    osm_source: Path,
    roi_geom: object,
    target_epsg: int,
    allowlist: List[str],
) -> Tuple[gpd.GeoDataFrame, str]:
    roads = gpd.read_file(osm_source)
    source = f"local:{osm_source}"
    if roads.crs is None:
        bounds = tuple(roads.total_bounds) if not roads.empty else (0.0, 0.0, 0.0, 0.0)
        roads = roads.set_crs("EPSG:4326" if _is_wgs84_bounds(bounds) else f"EPSG:{target_epsg}")
        source = f"{source}|assume_crs"
    if roads.crs.to_epsg() != target_epsg:
        roads = roads.to_crs(f"EPSG:{target_epsg}")
        source = f"{source}|reproject"
    if "highway" in roads.columns and allowlist:
        roads = roads[roads["highway"].astype(str).isin(allowlist)]
    roads = roads[roads.intersects(roi_geom)]
    keep_cols = [c for c in ["highway", "name", "oneway"] if c in roads.columns]
    if keep_cols:
        roads = roads[keep_cols + ["geometry"]]
    return roads, source


def _build_osm_corridor(
    osm_roads: gpd.GeoDataFrame,
    width_table: Dict[str, float],
    width_scale: float,
    target_epsg: int,
) -> gpd.GeoDataFrame:
    if osm_roads.empty:
        return gpd.GeoDataFrame(columns=["class", "meta_json", "geometry"], geometry=[], crs=f"EPSG:{target_epsg}")
    default_w = float(width_table.get("default", 9.0))
    buffers = []
    for _, row in osm_roads.iterrows():
        highway = str(row.get("highway") or "").lower()
        width = float(width_table.get(highway, default_w)) * float(width_scale)
        buffers.append(row.geometry.buffer(width))
    union = unary_union(buffers)
    corridor = gpd.GeoDataFrame(
        [{"class": "osm_corridor", "meta_json": json.dumps({"width_table": width_table, "scale": width_scale}), "geometry": union}],
        geometry="geometry",
        crs=f"EPSG:{target_epsg}",
    )
    return corridor


def _corridor_roi(path: Path, buffer_m: float, target_epsg: int) -> Tuple[Optional[object], List[str]]:
    errors: List[str] = []
    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        errors.append(f"corridor_read_failed:{exc}")
        return None, errors
    if gdf.empty:
        errors.append("corridor_empty")
        return None, errors
    if gdf.crs is None:
        errors.append("corridor_missing_crs")
        return None, errors
    if gdf.crs.to_epsg() != target_epsg:
        gdf = gdf.to_crs(f"EPSG:{target_epsg}")
    geom = gdf.unary_union.buffer(buffer_m)
    return geom, errors


def _load_corridor_geom(path: Path, target_epsg: int) -> Tuple[Optional[object], List[str]]:
    errors: List[str] = []
    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        errors.append(f"corridor_read_failed:{exc}")
        return None, errors
    if gdf.empty:
        errors.append("corridor_empty")
        return None, errors
    if gdf.crs is None:
        errors.append("corridor_missing_crs")
        gdf = gdf.set_crs(f"EPSG:{target_epsg}", allow_override=True)
    if gdf.crs.to_epsg() != target_epsg:
        gdf = gdf.to_crs(f"EPSG:{target_epsg}")
    try:
        return gdf.unary_union, errors
    except Exception as exc:
        errors.append(f"corridor_union_failed:{exc}")
        return None, errors


def _write_raster(
    path: Path,
    array: np.ndarray,
    transform: rasterio.Affine,
    epsg: int,
    nodata: float,
    warnings: List[str],
) -> None:
    validate_output_crs(path, epsg, None, warnings)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 1 if array.ndim == 2 else array.shape[0]
    height = array.shape[-2]
    width = array.shape[-1]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=array.dtype,
        crs=f"EPSG:{epsg}",
        transform=transform,
        nodata=nodata,
    ) as dst:
        if count == 1:
            dst.write(array, 1)
        else:
            dst.write(array)


def _rgb_to_lab_mask(rgb: np.ndarray, dev_max: float) -> np.ndarray:
    try:
        import cv2

        img = np.moveaxis(rgb, 0, -1)
        img8 = np.clip(img * 255.0, 0, 255).astype("uint8")
        lab = cv2.cvtColor(img8, cv2.COLOR_RGB2LAB)
        a = lab[:, :, 1].astype("float32")
        b = lab[:, :, 2].astype("float32")
        return (np.abs(a - 128.0) <= dev_max) & (np.abs(b - 128.0) <= dev_max)
    except Exception:
        return np.ones(rgb.shape[1:], dtype=bool)


def _apply_corridor_gate(mask: np.ndarray, corridor_geom: object, transform: rasterio.Affine, gate_m: float) -> np.ndarray:
    if gate_m <= 0 or corridor_geom is None or corridor_geom.is_empty:
        return mask
    gate_geom = corridor_geom.buffer(gate_m)
    gate_mask = features.rasterize(
        [(gate_geom, 1)],
        out_shape=mask.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    return (mask.astype(bool) & gate_mask).astype("uint8")


def _road_mask_classic(
    mosaic: np.ndarray,
    roi_mask: np.ndarray,
    cfg: dict,
) -> Tuple[np.ndarray, Dict[str, object], Dict[str, object]]:
    max_val = float(mosaic.max()) if mosaic.size else 0.0
    scale = 255.0 if max_val <= 255 else 10000.0
    if scale <= 0:
        scale = 1.0
    rgb = mosaic[:3].astype("float32") / scale
    max_rgb = np.max(rgb, axis=0)
    min_rgb = np.min(rgb, axis=0)
    sat = (max_rgb - min_rgb) / (max_rgb + 1e-6)
    val = max_rgb
    lab_mask = _rgb_to_lab_mask(rgb, float(cfg.get("LAB_AB_DEV_MAX", 14.0)))
    mask = (
        (sat < float(cfg.get("HSV_S_MAX", 0.25)))
        & (val > float(cfg.get("HSV_V_MIN", 0.22)))
        & (val < float(cfg.get("HSV_V_MAX", 0.90)))
        & lab_mask
    )
    mask &= roi_mask

    morph_stats = {"open_applied": False, "close_applied": False}
    comp_stats = {"components": 0, "filtered": 0}
    open_pix = int(cfg.get("MORPH_OPEN_PIX", 3))
    close_pix = int(cfg.get("MORPH_CLOSE_PIX", 7))
    min_area_m2 = float(cfg.get("MIN_AREA_M2", 50.0))
    res_m = float(cfg.get("MOSAIC_RES_M", 0.5))
    min_area_px = max(1, int(min_area_m2 / (res_m * res_m)))

    try:
        import cv2

        if open_pix > 1:
            kernel = np.ones((open_pix, open_pix), dtype=np.uint8)
            mask = cv2.morphologyEx(mask.astype("uint8"), cv2.MORPH_OPEN, kernel) > 0
            morph_stats["open_applied"] = True
        if close_pix > 1:
            kernel = np.ones((close_pix, close_pix), dtype=np.uint8)
            mask = cv2.morphologyEx(mask.astype("uint8"), cv2.MORPH_CLOSE, kernel) > 0
            morph_stats["close_applied"] = True

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"), connectivity=8)
        comp_stats["components"] = int(num - 1)
        keep = np.zeros_like(mask, dtype=bool)
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= min_area_px:
                keep |= labels == i
            else:
                comp_stats["filtered"] += 1
        mask = keep
    except Exception:
        try:
            from scipy import ndimage as ndi

            if open_pix > 1:
                mask = ndi.binary_opening(mask, iterations=1)
                morph_stats["open_applied"] = True
            if close_pix > 1:
                mask = ndi.binary_closing(mask, iterations=1)
                morph_stats["close_applied"] = True
            labels, num = ndi.label(mask)
            comp_stats["components"] = int(num)
            keep = np.zeros_like(mask, dtype=bool)
            for i in range(1, num + 1):
                area = int(np.sum(labels == i))
                if area >= min_area_px:
                    keep |= labels == i
                else:
                    comp_stats["filtered"] += 1
            mask = keep
        except Exception:
            comp_stats["components"] = 0
            comp_stats["filtered"] = 0
    meta = {
        "scale": scale,
        "hsv_s_max": float(cfg.get("HSV_S_MAX", 0.25)),
        "hsv_v_min": float(cfg.get("HSV_V_MIN", 0.22)),
        "hsv_v_max": float(cfg.get("HSV_V_MAX", 0.90)),
        "lab_ab_dev_max": float(cfg.get("LAB_AB_DEV_MAX", 14.0)),
        "min_area_px": min_area_px,
    }
    return mask.astype("uint8"), meta, {"morph": morph_stats, "components": comp_stats}


def _geom_to_polygons(geom) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        polys: List[Polygon] = []
        for g in geom.geoms:
            polys.extend(_geom_to_polygons(g))
        return polys
    return []


def _gdf_from_geom(
    geom,
    drive_id: str,
    target_epsg: int,
    uncert_m: float,
    min_area_m2: float,
) -> gpd.GeoDataFrame:
    rows = []
    for poly in _geom_to_polygons(geom):
        if poly.is_empty:
            continue
        area = float(poly.area)
        if area < min_area_m2:
            continue
        conf = min(0.8, 0.5 + 0.3 * min(1.0, area / max(min_area_m2 * 10.0, 1.0)))
        rows.append(
            {
                "cand_id": str(uuid4()),
                "source": "sat",
                "drive_id": drive_id,
                "class": "road_surface",
                "crs_epsg": target_epsg,
                "conf": conf,
                "uncert": uncert_m,
                "evid_ref": "[]",
                "conflict": "",
                "attr_json": json.dumps({"area_m2": round(area, 2)}),
                "geometry": poly,
            }
        )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{target_epsg}")
    ensure_required_columns(gdf, WORLD_FIELDS)
    return gdf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dop20_road_surface.yaml")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    run_id = now_ts()
    run_dir = Path("runs") / f"dop20_{run_id}"
    overwrite = bool(cfg.get("OVERWRITE", True))
    if overwrite:
        ensure_overwrite(run_dir)
    ensure_dir(run_dir)
    setup_logging(run_dir / "run.log")
    LOG.info("run_id=%s", run_id)

    warnings: List[str] = []
    errors: List[str] = []
    target_epsg = int(cfg.get("TARGET_EPSG", 32632))
    buffer_m = float(cfg.get("BUFFER_M", 100.0))

    data_root = Path(str(cfg.get("DATA_ROOT") or os.environ.get("POC_DATA_ROOT", "")))
    if not data_root.exists():
        LOG.error("POC_DATA_ROOT not set or invalid.")
        return 2

    dop20_root, dop20_source = _resolve_dop20_root(cfg)
    if dop20_root is None or not dop20_root.exists():
        LOG.error("DOP20 root not found (source=%s).", dop20_source)
        return 3

    tiles_gdf, tiles_source, tile_errors = _load_tiles_index(cfg, dop20_root, run_dir)
    if tiles_gdf.empty:
        LOG.error("DOP20 tiles index empty (source=%s).", tiles_source)
        return 4
    tiles_path = run_dir / "tiles" / "dop20_tiles_index.gpkg"
    write_gpkg_layer(tiles_path, "dop20_tiles", tiles_gdf, target_epsg, warnings, overwrite=True)

    drives = _load_drive_list(cfg)
    if not drives:
        LOG.error("No drives found.")
        return 5

    traj_stride = int(cfg.get("TRAJ_STRIDE", 5))
    mosaic_res_m = float(cfg.get("MOSAIC_RES_M", 0.5))
    min_area_m2 = float(cfg.get("MIN_AREA_M2", 50.0))
    simplify_m = float(cfg.get("SIMPLIFY_M", 0.5))
    method = str(cfg.get("METHOD", "classic")).lower()
    corridor_gate_m = float(cfg.get("CORRIDOR_GATE_M", 0.0))
    hsv_s_max = float(cfg.get("HSV_S_MAX", 0.25))
    hsv_v_min = float(cfg.get("HSV_V_MIN", 0.22))
    hsv_v_max = float(cfg.get("HSV_V_MAX", 0.90))
    lab_ab_dev_max = float(cfg.get("LAB_AB_DEV_MAX", 14.0))
    osm_source_hint = str(cfg.get("OSM_SOURCE") or "").strip()
    osm_allowlist = [str(x) for x in (cfg.get("OSM_HIGHWAY_ALLOWLIST") or [])]
    osm_width_table = cfg.get("OSM_WIDTH_TABLE") or {}
    osm_width_scale = float(cfg.get("OSM_WIDTH_SCALE", 1.0))
    closing_enable = bool(cfg.get("CLOSING_ENABLE", False))
    closing_only_corridor = bool(cfg.get("CLOSING_ONLY_IN_CORRIDOR", True))
    closing_pad_m = float(cfg.get("CLOSING_CORRIDOR_PAD_M", 3.0))
    closing_radius_m = float(cfg.get("CLOSING_RADIUS_M", 1.0))
    hole_fill_enable = bool(cfg.get("FILL_HOLES_ENABLE", False))
    hole_fill_max_m2 = float(cfg.get("HOLE_FILL_MAX_M2", 0.0))
    hole_fill_pad_m = float(cfg.get("HOLE_FILL_CORRIDOR_PAD_M", 3.0))
    hole_fill_corridor_only = bool(cfg.get("HOLE_FILL_ONLY_INTERSECT_CORRIDOR", True))

    all_candidates: List[gpd.GeoDataFrame] = []
    summary_drives: Dict[str, dict] = {}
    failure_reasons: Dict[str, int] = {}

    for drive_id in drives:
        drive_dir = run_dir / "drives" / drive_id
        roi_dir = ensure_dir(drive_dir / "roi")
        evidence_dir = ensure_dir(drive_dir / "evidence")
        candidates_dir = ensure_dir(drive_dir / "candidates")
        qa_dir = ensure_dir(drive_dir / "qa")

        roi_source = "missing"
        roi_errors: List[str] = []
        corridor_note = "roi"
        corridor_errors: List[str] = []
        roi_geom = None
        roi_geom, roi_source, roi_errors = _trajectory_roi(data_root, drive_id, buffer_m, traj_stride)
        if roi_geom is None:
            corridor_path = _find_corridor_gpkg(drive_id, [data_root])
            if corridor_path:
                roi_geom, roi_errors = _corridor_roi(corridor_path, buffer_m, target_epsg)
                if roi_geom is not None:
                    roi_source = f"corridor:{corridor_path}"
        if roi_geom is None:
            roi_geom = bbox_polygon(tiles_gdf.total_bounds)
            roi_source = "fallback_bbox"
            roi_errors.append("fallback_bbox_used")

        roi_gdf = gpd.GeoDataFrame(
            [{"drive_id": drive_id, "source": roi_source, "geometry": roi_geom}],
            geometry="geometry",
            crs=f"EPSG:{target_epsg}",
        )
        roi_path = roi_dir / "roi_buffer100_utm32.gpkg"
        write_gpkg_layer(roi_path, "roi", roi_gdf, target_epsg, warnings, overwrite=True)

        roi_bounds = roi_geom.bounds
        tile_hits = tiles_gdf[tiles_gdf.intersects(roi_geom)]
        tile_count = int(len(tile_hits))
        mosaic_path = evidence_dir / "dop20_mosaic_utm32.tif"
        mask_path = evidence_dir / "dop20_roadmask_utm32.tif"
        evidence_path = evidence_dir / "dop20_evidence_utm32.gpkg"
        candidates_raw_path = candidates_dir / "dop20_candidates_raw_utm32.gpkg"
        candidates_closed_path = candidates_dir / "dop20_candidates_closed_utm32.gpkg"
        candidates_path = candidates_dir / "dop20_candidates_utm32.gpkg"
        osm_dir = ensure_dir(drive_dir / "osm")
        osm_roads_path = osm_dir / "osm_roads_utm32.gpkg"
        osm_corridor_path = osm_dir / "osm_corridor_utm32.gpkg"

        mask_coverage = 0.0
        poly_area = 0.0
        components_count = 0
        status = "ok"
        fail_reason = ""
        mosaic_written = False
        mask_written = False

        if tile_count == 0:
            status = "fail"
            fail_reason = "no_tiles"
        else:
            vrt_list = []
            nodata = 0.0
            for _, row in tile_hits.iterrows():
                path = Path(str(row["path"]))
                if not path.exists():
                    continue
                try:
                    src = rasterio.open(path)
                except Exception as exc:
                    errors.append(f"tile_open_failed:{drive_id}:{path}:{exc}")
                    continue
                if src.nodata is not None:
                    nodata = float(src.nodata)
                src_crs = src.crs or f"EPSG:{target_epsg}"
                vrt = WarpedVRT(
                    src,
                    crs=f"EPSG:{target_epsg}",
                    resampling=Resampling.bilinear,
                    resolution=mosaic_res_m,
                    src_crs=src_crs,
                )
                vrt_list.append(vrt)
            if not vrt_list:
                status = "fail"
                fail_reason = "tile_open_failed"
            else:
                mosaic, transform = merge(
                    vrt_list,
                    bounds=roi_bounds,
                    res=mosaic_res_m,
                    nodata=nodata,
                )
                roi_mask = features.geometry_mask(
                    [roi_geom],
                    out_shape=(mosaic.shape[1], mosaic.shape[2]),
                    transform=transform,
                    invert=True,
                )
                mosaic[:, ~roi_mask] = nodata
                _write_raster(mosaic_path, mosaic, transform, target_epsg, nodata, warnings)
                mosaic_written = True

                if method != "classic":
                    status = "fail"
                    fail_reason = f"unsupported_method:{method}"
                else:
                    osm_note = "none"
                    corridor_geom = None
                    if osm_source_hint:
                        osm_path = Path(osm_source_hint)
                        if osm_path.exists():
                            try:
                                osm_roads, osm_note = _load_osm_roads(osm_path, roi_geom, target_epsg, osm_allowlist)
                                write_gpkg_layer(osm_roads_path, "osm_roads", osm_roads, target_epsg, warnings, overwrite=True)
                                osm_corridor = _build_osm_corridor(osm_roads, osm_width_table, osm_width_scale, target_epsg)
                                write_gpkg_layer(osm_corridor_path, "osm_corridor", osm_corridor, target_epsg, warnings, overwrite=True)
                                corridor_geom = osm_corridor.unary_union if not osm_corridor.empty else None
                                if corridor_geom is not None and not corridor_geom.is_empty:
                                    corridor_note = f"osm_corridor:{osm_corridor_path}"
                            except Exception as exc:
                                corridor_errors.append(f"osm_failed:{exc}")
                        else:
                            corridor_errors.append(f"osm_source_missing:{osm_path}")

                    road_mask, mask_meta, mask_stats = _road_mask_classic(mosaic, roi_mask, cfg)
                    if corridor_geom is not None and corridor_gate_m > 0:
                        road_mask = _apply_corridor_gate(road_mask, corridor_geom, transform, corridor_gate_m)
                    components_count = int(mask_stats["components"]["components"])
                    if roi_mask.any():
                        mask_coverage = float(road_mask[roi_mask].mean())
                    _write_raster(mask_path, road_mask.astype("uint8"), transform, target_epsg, 0, warnings)
                    mask_written = True

                    shapes = features.shapes(road_mask.astype("uint8"), mask=road_mask.astype(bool), transform=transform)
                    rows = []
                    evid_ids = []
                    for evid_type, path in [
                        ("dop20_mosaic", mosaic_path),
                        ("road_surface_mask", mask_path),
                    ]:
                        evid_id = str(uuid4())
                        evid_ids.append(evid_id)
                        rows.append(
                            {
                                "evid_id": evid_id,
                                "source": "sat",
                                "drive_id": drive_id,
                                "evid_type": evid_type,
                                "crs_epsg": target_epsg,
                                "path_rel": relpath(run_dir, path),
                                "frame_start": -1,
                                "frame_end": -1,
                                "conf": 1.0 if mosaic_written else 0.0,
                                "uncert": mosaic_res_m,
                                "meta_json": json.dumps(
                                    {
                                        "tiles": tile_count,
                                        "res_m": mosaic_res_m,
                                        "mask_coverage": round(mask_coverage, 6),
                                        "mask_stats": mask_stats,
                                        "mask_meta": mask_meta,
                                        "corridor_gate_m": corridor_gate_m,
                                        "osm_source": osm_note,
                                    }
                                ),
                                "geometry": roi_geom,
                            }
                        )
                    evidence_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{target_epsg}")
                    ensure_required_columns(evidence_gdf, PRIMITIVE_FIELDS)
                    write_gpkg_layer(evidence_path, "primitive_evidence", evidence_gdf, target_epsg, warnings, overwrite=True)

                    cand_rows = []
                    for geom, val in shapes:
                        if int(val) != 1:
                            continue
                        poly = shape(geom)
                        if poly.is_empty:
                            continue
                        area = float(poly.area)
                        if area < min_area_m2:
                            continue
                        if simplify_m > 0:
                            poly = poly.simplify(simplify_m)
                        poly_area += area
                        conf = min(0.8, 0.5 + 0.3 * min(1.0, area / max(min_area_m2 * 10.0, 1.0)))
                        cand_rows.append(
                            {
                                "cand_id": str(uuid4()),
                                "source": "sat",
                                "drive_id": drive_id,
                                "class": "road_surface",
                                "crs_epsg": target_epsg,
                                "conf": conf,
                                "uncert": mosaic_res_m,
                                "evid_ref": json.dumps(evid_ids),
                                "conflict": "",
                                "attr_json": json.dumps({"area_m2": round(area, 2)}),
                                "geometry": poly,
                            }
                        )
                    candidates_gdf = gpd.GeoDataFrame(cand_rows, geometry="geometry", crs=f"EPSG:{target_epsg}")
                    ensure_required_columns(candidates_gdf, WORLD_FIELDS)
                    write_gpkg_layer(
                        candidates_raw_path, "world_candidates", candidates_gdf, target_epsg, warnings, overwrite=True
                    )

                    cand_union = unary_union(list(candidates_gdf.geometry)) if not candidates_gdf.empty else None
                    post_corridor_geom = corridor_geom if corridor_geom is not None else roi_geom

                    closed_geom = cand_union
                    if closing_enable and cand_union is not None:
                        corridor_for_close = post_corridor_geom if closing_only_corridor else None
                        closed_geom = close_candidates_in_corridor(
                            cand_union,
                            corridor_for_close,
                            closing_radius_m,
                            closing_pad_m,
                        )
                    closed_gdf = _gdf_from_geom(
                        closed_geom, drive_id, target_epsg, mosaic_res_m, min_area_m2
                    )
                    closed_gdf["evid_ref"] = json.dumps(evid_ids)
                    write_gpkg_layer(
                        candidates_closed_path, "world_candidates", closed_gdf, target_epsg, warnings, overwrite=True
                    )

                    final_geom = closed_geom
                    if hole_fill_enable and closed_geom is not None:
                        corridor_for_fill = post_corridor_geom if hole_fill_corridor_only else None
                        final_geom = fill_small_holes_by_corridor(
                            closed_geom,
                            corridor_for_fill,
                            hole_fill_max_m2,
                            hole_fill_pad_m,
                        )
                    final_gdf = _gdf_from_geom(
                        final_geom, drive_id, target_epsg, mosaic_res_m, min_area_m2
                    )
                    final_gdf["evid_ref"] = json.dumps(evid_ids)
                    write_gpkg_layer(
                        candidates_path, "world_candidates", final_gdf, target_epsg, warnings, overwrite=True
                    )
                    if not final_gdf.empty:
                        all_candidates.append(final_gdf.assign(drive_id=drive_id))

        if status == "fail":
            failure_reasons[fail_reason or "unknown"] = failure_reasons.get(fail_reason or "unknown", 0) + 1
            evid_rows = []
            for evid_type, path in [
                ("dop20_mosaic", mosaic_path),
                ("road_surface_mask", mask_path),
            ]:
                evid_rows.append(
                    {
                        "evid_id": str(uuid4()),
                        "source": "sat",
                        "drive_id": drive_id,
                        "evid_type": evid_type,
                        "crs_epsg": target_epsg,
                        "path_rel": relpath(run_dir, path),
                        "frame_start": -1,
                        "frame_end": -1,
                        "conf": 0.0,
                        "uncert": mosaic_res_m,
                        "meta_json": json.dumps({"status": "failed", "reason": fail_reason}),
                        "geometry": roi_geom,
                    }
                )
            evidence_gdf = gpd.GeoDataFrame(evid_rows, geometry="geometry", crs=f"EPSG:{target_epsg}")
            ensure_required_columns(evidence_gdf, PRIMITIVE_FIELDS)
            write_gpkg_layer(
                evidence_path, "primitive_evidence", evidence_gdf, target_epsg, warnings, overwrite=True
            )
            empty_candidates = gpd.GeoDataFrame(columns=WORLD_FIELDS + ["geometry"], geometry=[], crs=f"EPSG:{target_epsg}")
            write_gpkg_layer(
                candidates_raw_path, "world_candidates", empty_candidates, target_epsg, warnings, overwrite=True
            )
            write_gpkg_layer(
                candidates_closed_path, "world_candidates", empty_candidates, target_epsg, warnings, overwrite=True
            )
            write_gpkg_layer(
                candidates_path, "world_candidates", empty_candidates, target_epsg, warnings, overwrite=True
            )

        qa_rows = [
            {
                "drive_id": drive_id,
                "roi_path": relpath(run_dir, roi_path),
                "mosaic_path": relpath(run_dir, mosaic_path) if mosaic_written else "",
                "mask_path": relpath(run_dir, mask_path) if mask_written else "",
                "candidates_path": relpath(run_dir, candidates_path),
                "candidates_raw_path": relpath(run_dir, candidates_raw_path),
                "candidates_closed_path": relpath(run_dir, candidates_closed_path),
                "post_corridor": corridor_note,
                "osm_roads_path": relpath(run_dir, osm_roads_path) if osm_roads_path.exists() else "",
                "osm_corridor_path": relpath(run_dir, osm_corridor_path) if osm_corridor_path.exists() else "",
                "corridor_gate_m": corridor_gate_m,
                "minx": roi_bounds[0],
                "miny": roi_bounds[1],
                "maxx": roi_bounds[2],
                "maxy": roi_bounds[3],
                "mask_coverage": round(mask_coverage, 6),
                "poly_area": round(poly_area, 2),
                "components_count": components_count,
                "tiles_hit": tile_count,
                "status": status,
                "fail_reason": fail_reason,
            }
        ]
        write_csv(qa_dir / "qa_index.csv", qa_rows, list(qa_rows[0].keys()))

        report_lines = [
            "# DOP20 Road Surface Report",
            "",
            f"- drive_id: {drive_id}",
            f"- status: {status}",
            f"- fail_reason: {fail_reason or 'none'}",
            f"- roi_source: {roi_source}",
            f"- tiles_hit: {tile_count}",
            "",
            "## Parameters",
            "```json",
            json.dumps(
                {
                    "buffer_m": buffer_m,
                    "mosaic_res_m": mosaic_res_m,
                    "method": method,
                    "hsv_s_max": hsv_s_max,
                    "hsv_v_min": hsv_v_min,
                    "hsv_v_max": hsv_v_max,
                    "lab_ab_dev_max": lab_ab_dev_max,
                    "morph_open_pix": int(cfg.get("MORPH_OPEN_PIX", 3)),
                    "morph_close_pix": int(cfg.get("MORPH_CLOSE_PIX", 7)),
                    "min_area_m2": min_area_m2,
                    "simplify_m": simplify_m,
                    "corridor_gate_m": corridor_gate_m,
                },
                indent=2,
            ),
            "```",
            "",
            "## CRS Checks",
        ]
        report_lines.extend([f"- {w}" for w in warnings] if warnings else ["- ok"])
        report_lines.extend(
            [
                "",
                "## Outputs",
                f"- {relpath(run_dir, roi_path)}",
                f"- {relpath(run_dir, evidence_path)}",
                f"- {relpath(run_dir, osm_roads_path)}" if osm_roads_path.exists() else "- osm_roads: none",
                f"- {relpath(run_dir, osm_corridor_path)}" if osm_corridor_path.exists() else "- osm_corridor: none",
                f"- {relpath(run_dir, candidates_raw_path)}",
                f"- {relpath(run_dir, candidates_closed_path)}",
                f"- {relpath(run_dir, candidates_path)}",
                f"- {relpath(run_dir, qa_dir / 'qa_index.csv')}",
                "",
                "## Metrics",
                f"- mask_coverage: {mask_coverage:.6f}",
                f"- poly_area_m2: {poly_area:.2f}",
                f"- components_count: {components_count}",
                "",
                "## Postprocess",
                f"- post_corridor: {corridor_note}",
                f"- corridor_gate_m: {corridor_gate_m}",
                f"- closing_enable: {closing_enable}",
                f"- closing_radius_m: {closing_radius_m}",
                f"- closing_pad_m: {closing_pad_m}",
                f"- hole_fill_enable: {hole_fill_enable}",
                f"- hole_fill_max_m2: {hole_fill_max_m2}",
                f"- hole_fill_pad_m: {hole_fill_pad_m}",
                f"- hole_fill_corridor_only: {hole_fill_corridor_only}",
                "",
                "## Failures",
            ]
        )
        report_lines.extend([f"- {e}" for e in roi_errors] if roi_errors else ["- none"])
        if corridor_errors:
            report_lines.extend([f"- corridor:{e}" for e in corridor_errors])
        report_lines.extend(
            [
                "",
                "## Known Limits",
                "- classic baseline may miss bright concrete or dark asphalt under shadows.",
                "- vegetation filtering relies on NDVI when NIR is present; RGB-only tiles use saturation heuristics.",
            ]
        )
        write_text(qa_dir / "report.md", "\n".join(report_lines))

        summary_drives[drive_id] = {
            "status": status,
            "fail_reason": fail_reason or "",
            "tiles_hit": tile_count,
            "mask_coverage": round(mask_coverage, 6),
            "poly_area_m2": round(poly_area, 2),
            "components_count": components_count,
            "roi_source": roi_source,
        }

    if all_candidates:
        merged = gpd.GeoDataFrame(pd.concat(all_candidates, ignore_index=True), geometry="geometry", crs=f"EPSG:{target_epsg}")
        merged_path = run_dir / "merged" / "dop20_candidates_utm32.gpkg"
        write_gpkg_layer(merged_path, "world_candidates", merged, target_epsg, warnings, overwrite=True)

    summary = {
        "run_id": run_id,
        "config": args.config,
        "dop20_root": str(dop20_root),
        "dop20_source": dop20_source,
        "tiles_source": tiles_source,
        "drives": summary_drives,
        "failure_reasons": failure_reasons,
        "tile_errors": tile_errors,
        "warnings": warnings,
        "errors": errors,
    }
    write_json(run_dir / "run_summary.json", summary)
    summary_md = [
        "# DOP20 Road Surface Summary",
        "",
        f"- run_id: {run_id}",
        f"- config: {args.config}",
        f"- dop20_root: {dop20_root}",
        f"- tiles_source: {tiles_source}",
        f"- drives: {len(summary_drives)}",
        "",
        "## Failures",
    ]
    if failure_reasons:
        summary_md.extend([f"- {k}: {v}" for k, v in failure_reasons.items()])
    else:
        summary_md.append("- none")
    summary_md.extend(["", "## Errors"])
    summary_md.extend([f"- {e}" for e in errors] if errors else ["- none"])
    summary_md.extend(["", "## Tile Errors"])
    summary_md.extend([f"- {e}" for e in tile_errors] if tile_errors else ["- none"])
    write_text(run_dir / "run_summary.md", "\n".join(summary_md))
    LOG.info("completed dop20 run: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

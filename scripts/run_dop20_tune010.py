from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
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


@dataclass
class MosaicBundle:
    mosaic: np.ndarray
    transform: rasterio.Affine
    nodata: float
    roi_mask: np.ndarray
    roi_area_m2: float
    res_m: float
    tile_count: int


def _trajectory_roi(data_root: Path, drive_id: str, buffer_m: float, stride: int) -> Tuple[object, List[str]]:
    errors: List[str] = []
    oxts_dir = _find_oxts_dir(data_root, drive_id)
    frames = sorted(oxts_dir.glob("*.txt"))
    if stride > 1:
        frames = frames[::stride]
    points: List[Tuple[float, float]] = []
    for frame in frames:
        frame_id = frame.stem
        try:
            x, y, _ = load_kitti360_pose(data_root, drive_id, frame_id)
        except Exception as exc:  # pragma: no cover - data dependent
            errors.append(f"pose_failed:{frame_id}:{exc}")
            continue
        points.append((x, y))
    if not points:
        return bbox_polygon((0.0, 0.0, 0.0, 0.0)), errors
    if len(points) == 1:
        geom = MultiPoint(points).buffer(buffer_m)
    else:
        geom = LineString(points).buffer(buffer_m)
    return geom, errors


def _scan_tiles(root: Path) -> List[Path]:
    tiles: List[Path] = []
    for ext in ("*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.jp2"):
        tiles.extend(root.rglob(ext))
    return sorted(tiles)


def _build_tiles_index(tiles: Iterable[Path], target_epsg: int) -> Tuple[gpd.GeoDataFrame, List[str]]:
    rows = []
    errors: List[str] = []
    for path in tiles:
        try:
            with rasterio.open(path) as ds:
                bounds = ds.bounds
                crs = ds.crs
        except Exception as exc:  # pragma: no cover - raster IO
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
            except Exception as exc:  # pragma: no cover - transform
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


def _resolve_tiles(dop20_root: Path, target_epsg: int) -> Tuple[gpd.GeoDataFrame, str, List[str]]:
    tiles_dir = dop20_root / "tiles_utm32"
    tile_root = tiles_dir if tiles_dir.exists() else dop20_root
    tiles = _scan_tiles(tile_root)
    gdf, errors = _build_tiles_index(tiles, target_epsg)
    return gdf, f"scan:{tile_root}", errors


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


def _build_mosaic_bundle(
    tiles_gdf: gpd.GeoDataFrame,
    roi_geom: object,
    dop20_root: Path,
    res_m: float,
    target_epsg: int,
    warnings: List[str],
    mosaic_path: Path,
) -> MosaicBundle:
    tile_hits = tiles_gdf[tiles_gdf.intersects(roi_geom)]
    vrt_list = []
    nodata = 0.0
    for _, row in tile_hits.iterrows():
        path = Path(str(row["path"]))
        if not path.exists():
            continue
        src = rasterio.open(path)
        if src.nodata is not None:
            nodata = float(src.nodata)
        src_crs = src.crs or f"EPSG:{target_epsg}"
        vrt = WarpedVRT(
            src,
            crs=f"EPSG:{target_epsg}",
            resampling=Resampling.bilinear,
            resolution=res_m,
            src_crs=src_crs,
        )
        vrt_list.append(vrt)
    if not vrt_list:
        raise RuntimeError("no_tiles")
    mosaic, transform = merge(vrt_list, bounds=roi_geom.bounds, res=res_m, nodata=nodata)
    roi_mask = features.geometry_mask(
        [roi_geom],
        out_shape=(mosaic.shape[1], mosaic.shape[2]),
        transform=transform,
        invert=True,
    )
    mosaic[:, ~roi_mask] = nodata
    _write_raster(mosaic_path, mosaic, transform, target_epsg, nodata, warnings)
    roi_area_m2 = float(roi_mask.sum()) * (res_m * res_m)
    return MosaicBundle(
        mosaic=mosaic,
        transform=transform,
        nodata=nodata,
        roi_mask=roi_mask,
        roi_area_m2=roi_area_m2,
        res_m=res_m,
        tile_count=int(len(tile_hits)),
    )


def _is_wgs84_bounds(bounds: Tuple[float, float, float, float]) -> bool:
    minx, miny, maxx, maxy = bounds
    return -180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0 and -90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0


def _load_osm_roads(osm_source: Path, roi_geom: object, target_epsg: int, allowlist: List[str]) -> Tuple[gpd.GeoDataFrame, str]:
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
    target_epsg: int,
    width_meta: Optional[Dict[str, object]] = None,
) -> gpd.GeoDataFrame:
    if osm_roads.empty:
        return gpd.GeoDataFrame(columns=["class", "meta_json", "geometry"], geometry=[], crs=f"EPSG:{target_epsg}")
    default_w = float(width_table.get("default", 9.0))
    buffers = []
    for _, row in osm_roads.iterrows():
        highway = str(row.get("highway") or "").lower()
        width = float(width_table.get(highway, default_w))
        buffers.append(row.geometry.buffer(width))
    union = unary_union(buffers)
    meta_payload = {"width_table": width_table}
    if width_meta:
        meta_payload.update(width_meta)
    corridor = gpd.GeoDataFrame(
        [
            {
                "class": "osm_corridor",
                "meta_json": json.dumps(meta_payload),
                "geometry": union,
            }
        ],
        geometry="geometry",
        crs=f"EPSG:{target_epsg}",
    )
    return corridor


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


def _mask_classic(bundle: MosaicBundle, params: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, object]]:
    mosaic = bundle.mosaic
    res_m = bundle.res_m
    max_val = float(mosaic.max()) if mosaic.size else 0.0
    scale = 255.0 if max_val <= 255 else 10000.0
    if scale <= 0:
        scale = 1.0
    rgb = mosaic[:3].astype("float32") / scale
    max_rgb = np.max(rgb, axis=0)
    min_rgb = np.min(rgb, axis=0)
    sat = (max_rgb - min_rgb) / (max_rgb + 1e-6)
    val = max_rgb
    lab_mask = _rgb_to_lab_mask(rgb, float(params["LAB_AB_DEV_MAX"]))
    mask = (
        (sat < float(params["HSV_S_MAX"]))
        & (val > float(params["HSV_V_MIN"]))
        & (val < float(params["HSV_V_MAX"]))
        & lab_mask
    )
    mask &= bundle.roi_mask

    open_pix = int(params["MORPH_OPEN_PIX"])
    close_pix = int(params["MORPH_CLOSE_PIX"])
    min_area_m2 = float(params["MIN_AREA_M2"])
    min_area_px = max(1, int(min_area_m2 / (res_m * res_m)))
    components = 0
    filtered = 0

    try:
        import cv2

        if open_pix > 1:
            kernel = np.ones((open_pix, open_pix), dtype=np.uint8)
            mask = cv2.morphologyEx(mask.astype("uint8"), cv2.MORPH_OPEN, kernel) > 0
        if close_pix > 1:
            kernel = np.ones((close_pix, close_pix), dtype=np.uint8)
            mask = cv2.morphologyEx(mask.astype("uint8"), cv2.MORPH_CLOSE, kernel) > 0
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"), connectivity=8)
        components = int(num - 1)
        keep = np.zeros_like(mask, dtype=bool)
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= min_area_px:
                keep |= labels == i
            else:
                filtered += 1
        mask = keep
    except Exception:
        pass

    meta = {
        "scale": scale,
        "min_area_px": min_area_px,
        "components_raw": components,
        "filtered_raw": filtered,
    }
    return mask.astype("uint8"), meta


def _apply_corridor_gate(mask: np.ndarray, corridor_geom: object, bundle: MosaicBundle, gate_m: float) -> np.ndarray:
    if gate_m < 0:
        return mask
    gate_geom = corridor_geom.buffer(gate_m)
    gate_mask = features.rasterize(
        [(gate_geom, 1)],
        out_shape=bundle.roi_mask.shape,
        transform=bundle.transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    gated = mask.astype(bool) & gate_mask & bundle.roi_mask
    return gated.astype("uint8")


def _penalty_upper(x: float, target_max: float, hard_max: float) -> float:
    if x <= target_max:
        return 0.0
    if x >= hard_max:
        return 100.0 + (x - hard_max) * 10.0
    return (x - target_max) / max(hard_max - target_max, 1e-6)


def _penalty_lower(x: float, target_min: float, hard_min: float) -> float:
    if x >= target_min:
        return 0.0
    if x <= hard_min:
        return 100.0 + (hard_min - x) * 10.0
    return (target_min - x) / max(target_min - hard_min, 1e-6)


def _penalty_range(x: float, target: Tuple[float, float], hard: Tuple[float, float]) -> float:
    if target[0] <= x <= target[1]:
        return 0.0
    if x < target[0]:
        return _penalty_lower(x, target[0], hard[0])
    return _penalty_upper(x, target[1], hard[1])


def _score(metrics: Dict[str, float]) -> float:
    return (
        4.0 * _penalty_range(metrics["mask_coverage"], (0.06, 0.22), (0.04, 0.30))
        + 4.0 * _penalty_lower(metrics["corridor_coverage"], 0.70, 0.60)
        + 3.0 * _penalty_upper(metrics["outside_ratio"], 0.12, 0.18)
        + 2.0 * _penalty_upper(metrics["components_count"], 180.0, 300.0)
        + 1.5 * _penalty_upper(metrics["largest_roi_ratio"], 0.25, 0.35)
    )


def _metrics_from_masks(
    mask: np.ndarray,
    corridor_geom: object,
    bundle: MosaicBundle,
) -> Dict[str, float]:
    res_m = bundle.res_m
    roi_mask = bundle.roi_mask
    transform = bundle.transform
    out_shape = roi_mask.shape
    corridor_mask = features.rasterize(
        [(corridor_geom, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    mask_bool = mask.astype(bool)
    roi_area = float(roi_mask.sum()) * (res_m * res_m)
    mask_area = float((mask_bool & roi_mask).sum()) * (res_m * res_m)
    corridor_area = float(corridor_mask.sum()) * (res_m * res_m)
    inter_area = float((mask_bool & corridor_mask).sum()) * (res_m * res_m)
    outside_area = float((mask_bool & ~corridor_mask & roi_mask).sum()) * (res_m * res_m)
    mask_coverage = mask_area / max(roi_area, 1e-6)
    corridor_coverage = inter_area / max(corridor_area, 1e-6)
    outside_ratio = outside_area / max(roi_area, 1e-6)
    return {
        "mask_area_m2": mask_area,
        "roi_area_m2": roi_area,
        "corridor_area_m2": corridor_area,
        "mask_coverage": mask_coverage,
        "corridor_coverage": corridor_coverage,
        "outside_ratio": outside_ratio,
    }


def _candidates_from_mask(
    mask: np.ndarray,
    bundle: MosaicBundle,
    params: Dict[str, float],
    drive_id: str,
    target_epsg: int,
) -> Tuple[gpd.GeoDataFrame, int, float]:
    shapes_iter = features.shapes(mask.astype("uint8"), mask=mask.astype(bool), transform=bundle.transform)
    min_area_m2 = float(params["MIN_AREA_M2"])
    simplify_m = float(params["SIMPLIFY_TOL_M"])
    rows = []
    largest = 0.0
    for geom, val in shapes_iter:
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
        largest = max(largest, area)
        conf = min(0.8, 0.5 + 0.3 * min(1.0, area / max(min_area_m2 * 10.0, 1.0)))
        rows.append(
            {
                "cand_id": str(uuid4()),
                "source": "sat",
                "drive_id": drive_id,
                "class": "road_surface",
                "crs_epsg": target_epsg,
                "conf": conf,
                "uncert": bundle.res_m,
                "evid_ref": "[]",
                "conflict": "",
                "attr_json": json.dumps({"area_m2": round(area, 2)}),
                "geometry": poly,
            }
        )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{target_epsg}")
    ensure_required_columns(gdf, WORLD_FIELDS)
    return gdf, len(gdf), largest


def _hole_stats_from_geom(geom) -> Tuple[int, float, float]:
    if geom is None or geom.is_empty:
        return 0, 0.0, 0.0
    polys: List[Polygon] = []
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            if isinstance(g, Polygon):
                polys.append(g)
            elif isinstance(g, MultiPolygon):
                polys.extend(list(g.geoms))
    holes_count = 0
    holes_area = 0.0
    area_total = 0.0
    for p in polys:
        if p.is_empty:
            continue
        area_total += float(p.area)
        holes_count += len(p.interiors)
        for ring in p.interiors:
            hole_poly = Polygon(ring)
            holes_area += float(hole_poly.area)
    holes_ratio = holes_area / max(area_total, 1e-6)
    return holes_count, holes_area, holes_ratio


def _candidate_union(gdf: gpd.GeoDataFrame):
    if gdf.empty:
        return None
    try:
        return unary_union(list(gdf.geometry))
    except Exception:
        return unary_union([g for g in gdf.geometry if g is not None])


def _vector_coverage_and_outside(cand_geom, corridor_geom, roi_geom) -> Tuple[float, float]:
    if cand_geom is None or cand_geom.is_empty:
        return 0.0, 0.0
    roi_area = float(roi_geom.area)
    corridor_area = float(corridor_geom.area)
    inter_corridor = cand_geom.intersection(corridor_geom)
    corridor_cov = float(inter_corridor.area) / max(corridor_area, 1e-6)
    outside = cand_geom.difference(corridor_geom)
    outside_ratio = float(outside.area) / max(roi_area, 1e-6)
    return corridor_cov, outside_ratio


def _geom_to_polygons(geom) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    polys: List[Polygon] = []
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            if isinstance(g, Polygon):
                polys.append(g)
            elif isinstance(g, MultiPolygon):
                polys.extend(list(g.geoms))
    return [p for p in polys if p is not None and not p.is_empty]


def _gdf_from_geom(
    geom,
    drive_id: str,
    target_epsg: int,
    uncert_m: float,
    min_area_m2: float,
) -> gpd.GeoDataFrame:
    rows = []
    for poly in _geom_to_polygons(geom):
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


def _stop_ok(metrics: Dict[str, float]) -> bool:
    return (
        metrics["corridor_coverage"] >= 0.70
        and metrics["outside_ratio"] <= 0.12
        and metrics["mask_coverage"] <= 0.22
        and metrics["components_count"] <= 180
    )


def main() -> int:
    cfg = load_yaml(Path("configs/dop20_tune010.yaml"))
    run_id = now_ts()
    run_dir = Path("runs") / f"dop20_tune010_{run_id}"
    if bool(cfg.get("OVERWRITE", True)):
        ensure_overwrite(run_dir)
    ensure_dir(run_dir)
    setup_logging(run_dir / "run.log")

    target_epsg = int(cfg.get("TARGET_EPSG", 32632))
    drive_id = str(cfg.get("DRIVE_ID") or "")
    if not drive_id:
        LOG.error("DRIVE_ID missing.")
        return 2

    data_root = Path(str(cfg.get("DATA_ROOT") or os.environ.get("POC_DATA_ROOT", "")))
    if not data_root.exists():
        LOG.error("POC_DATA_ROOT not set or invalid.")
        return 3

    dop20_root = Path(str(cfg.get("DOP20_ROOT") or ""))
    if not dop20_root.exists():
        LOG.error("DOP20_ROOT not found: %s", dop20_root)
        return 4

    drive_dir = run_dir / "drives" / drive_id
    roi_dir = ensure_dir(drive_dir / "roi")
    osm_dir = ensure_dir(drive_dir / "osm")
    evidence_dir = ensure_dir(drive_dir / "evidence")
    candidates_dir = ensure_dir(drive_dir / "candidates")
    qa_dir = ensure_dir(drive_dir / "qa")
    autotune_dir = ensure_dir(run_dir / "autotune")

    warnings: List[str] = []
    errors: List[str] = []

    roi_geom, roi_errors = _trajectory_roi(
        data_root,
        drive_id,
        buffer_m=float(cfg.get("ROI_BUFFER_M", 100.0)),
        stride=int(cfg.get("TRAJ_STRIDE", 5)),
    )
    roi_gdf = gpd.GeoDataFrame([{"drive_id": drive_id, "geometry": roi_geom}], geometry="geometry", crs=f"EPSG:{target_epsg}")
    roi_path = roi_dir / "roi_buffer100_utm32.gpkg"
    write_gpkg_layer(roi_path, "roi", roi_gdf, target_epsg, warnings, overwrite=True)

    osm_source = Path(str(cfg.get("OSM_SOURCE") or ""))
    if not osm_source.exists():
        errors.append(f"osm_source_missing:{osm_source}")
        LOG.error("OSM_SOURCE not found: %s", osm_source)
        return 5
    osm_allow = [str(x) for x in (cfg.get("OSM_HIGHWAY_ALLOWLIST") or [])]
    osm_roads, osm_source_note = _load_osm_roads(osm_source, roi_geom, target_epsg, osm_allow)
    osm_roads_path = osm_dir / "osm_roads_utm32.gpkg"
    write_gpkg_layer(osm_roads_path, "osm_roads", osm_roads, target_epsg, warnings, overwrite=True)

    width_scale = float(cfg.get("OSM_WIDTH_SCALE", 1.0))
    width_table_raw = {str(k): float(v) for k, v in (cfg.get("OSM_WIDTH_TABLE") or {}).items()}
    width_table = {k: v * width_scale for k, v in width_table_raw.items()}
    corridor_gdf = _build_osm_corridor(
        osm_roads,
        width_table,
        target_epsg,
        width_meta={"width_scale": width_scale, "width_table_raw": width_table_raw},
    )
    if corridor_gdf.empty:
        errors.append("osm_corridor_empty")
        LOG.error("OSM corridor is empty.")
        return 6
    corridor_geom = corridor_gdf.geometry.iloc[0]
    corridor_path = osm_dir / "osm_corridor_utm32.gpkg"
    write_gpkg_layer(corridor_path, "osm_corridor", corridor_gdf, target_epsg, warnings, overwrite=True)

    tiles_gdf, tiles_source, tile_errors = _resolve_tiles(dop20_root, target_epsg)
    if tiles_gdf.empty:
        LOG.error("DOP20 tiles index empty.")
        return 7
    tiles_path = run_dir / "tiles" / "dop20_tiles_index.gpkg"
    write_gpkg_layer(tiles_path, "dop20_tiles", tiles_gdf, target_epsg, warnings, overwrite=True)

    mosaic_path = evidence_dir / "dop20_mosaic_utm32.tif"
    try:
        bundle = _build_mosaic_bundle(
            tiles_gdf,
            roi_geom,
            dop20_root,
            res_m=float(cfg.get("MOSAIC_RES_M", 0.25)),
            target_epsg=target_epsg,
            warnings=warnings,
            mosaic_path=mosaic_path,
        )
    except RuntimeError as exc:
        LOG.error("mosaic_failed:%s", exc)
        return 8

    init_params = {str(k): float(v) for k, v in (cfg.get("INIT_PARAMS") or {}).items()}
    grid1 = cfg.get("GRID_STAGE1") or {}
    grid2 = cfg.get("GRID_STAGE2") or {}
    max_trials = int(cfg.get("MAX_TRIALS", 40))

    trials: List[Dict[str, object]] = []
    best = None
    best_mask = None
    best_candidates = None
    initial_metrics = None
    hole_fill_summary: Dict[str, object] = {}

    def run_trial(param_overrides: Dict[str, float], stage: str, trial_id: int) -> Tuple[Dict[str, object], np.ndarray, gpd.GeoDataFrame]:
        params = dict(init_params)
        params.update({k: float(v) for k, v in param_overrides.items()})
        mask, mask_meta = _mask_classic(bundle, params)
        gate_m = float(params.get("CORRIDOR_GATE_M", 0.0))
        mask = _apply_corridor_gate(mask, corridor_geom, bundle, gate_m)
        mask_meta["corridor_gate_m"] = gate_m
        cand_gdf, comp_cnt, largest_area = _candidates_from_mask(mask, bundle, params, drive_id, target_epsg)
        metrics = _metrics_from_masks(mask, corridor_geom, bundle)
        metrics["components_count"] = float(comp_cnt)
        metrics["largest_roi_ratio"] = float(largest_area) / max(metrics["roi_area_m2"], 1e-6)
        score_val = float(_score(metrics))
        row = {
            "trial_id": trial_id,
            "stage": stage,
            **{k: params[k] for k in init_params.keys()},
            **metrics,
            "score": score_val,
            "status": "ok",
        }
        row["mask_meta_json"] = json.dumps(mask_meta)
        return row, mask, cand_gdf

    trial_id = 0
    # Stage 1: HSV/LAB sweep
    for s_max in grid1.get("HSV_S_MAX", [init_params["HSV_S_MAX"]]):
        for v_min in grid1.get("HSV_V_MIN", [init_params["HSV_V_MIN"]]):
            for v_max in grid1.get("HSV_V_MAX", [init_params["HSV_V_MAX"]]):
                for lab_dev in grid1.get("LAB_AB_DEV_MAX", [init_params["LAB_AB_DEV_MAX"]]):
                    trial_id += 1
                    row, mask, cand = run_trial(
                        {
                            "HSV_S_MAX": s_max,
                            "HSV_V_MIN": v_min,
                            "HSV_V_MAX": v_max,
                            "LAB_AB_DEV_MAX": lab_dev,
                        },
                        stage="stage1",
                        trial_id=trial_id,
                    )
                    trials.append(row)
                    if initial_metrics is None:
                        initial_metrics = row
                    if best is None or row["score"] < best["score"]:
                        best, best_mask, best_candidates = row, mask, cand
                    if _stop_ok(row):
                        break
                    if trial_id >= max_trials:
                        break
                if (best and _stop_ok(best)) or trial_id >= max_trials:
                    break
            if (best and _stop_ok(best)) or trial_id >= max_trials:
                break
        if (best and _stop_ok(best)) or trial_id >= max_trials:
            break

    # Stage 2: morphology/min area around best HSV/LAB
    if best is not None and not _stop_ok(best) and trial_id < max_trials:
        fixed = {
            "HSV_S_MAX": float(best["HSV_S_MAX"]),
            "HSV_V_MIN": float(best["HSV_V_MIN"]),
            "HSV_V_MAX": float(best["HSV_V_MAX"]),
            "LAB_AB_DEV_MAX": float(best["LAB_AB_DEV_MAX"]),
        }
        for open_pix in grid2.get("MORPH_OPEN_PIX", [init_params["MORPH_OPEN_PIX"]]):
            for close_pix in grid2.get("MORPH_CLOSE_PIX", [init_params["MORPH_CLOSE_PIX"]]):
                for min_area in grid2.get("MIN_AREA_M2", [init_params["MIN_AREA_M2"]]):
                    for gate_m in grid2.get("CORRIDOR_GATE_M", [init_params.get("CORRIDOR_GATE_M", 0.0)]):
                        if trial_id >= max_trials:
                            break
                        trial_id += 1
                        overrides = dict(fixed)
                        overrides.update(
                            {
                                "MORPH_OPEN_PIX": open_pix,
                                "MORPH_CLOSE_PIX": close_pix,
                                "MIN_AREA_M2": min_area,
                                "CORRIDOR_GATE_M": gate_m,
                            }
                        )
                        row, mask, cand = run_trial(overrides, stage="stage2", trial_id=trial_id)
                        trials.append(row)
                        if row["score"] < best["score"]:
                            best, best_mask, best_candidates = row, mask, cand
                        if _stop_ok(best):
                            break
                    if _stop_ok(best) or trial_id >= max_trials:
                        break
                if _stop_ok(best) or trial_id >= max_trials:
                    break
            if _stop_ok(best) or trial_id >= max_trials:
                break

    if best is None or best_mask is None or best_candidates is None:
        LOG.error("autotune_failed:no_trials")
        return 9

    trials_df = pd.DataFrame(trials)

    best_params = {k: float(best[k]) for k in init_params.keys()}
    with (autotune_dir / "best_params.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(best_params, f, sort_keys=False, allow_unicode=False)

    # Vector postprocess: corridor-constrained closing, then corridor-constrained hole fill.
    raw_candidates = best_candidates.copy()
    min_area_m2 = float(best_params.get("MIN_AREA_M2", 120.0))
    raw_union = _candidate_union(raw_candidates)
    raw_holes_count, raw_holes_area, raw_holes_ratio = _hole_stats_from_geom(raw_union)
    raw_corridor_cov_vec, raw_outside_vec = _vector_coverage_and_outside(raw_union, corridor_geom, roi_geom)

    closing_enable = bool(cfg.get("CLOSING_ENABLE", False))
    closing_only_corridor = bool(cfg.get("CLOSING_ONLY_IN_CORRIDOR", True))
    closing_pad_m = float(cfg.get("CLOSING_CORRIDOR_PAD_M", 3.0))
    closing_radius_default = float(cfg.get("CLOSING_RADIUS_M", 1.0))
    closing_candidates_cfg = [float(x) for x in (cfg.get("CLOSING_RADIUS_CANDIDATES_M") or [])]
    closing_radii = sorted(set(closing_candidates_cfg + ([closing_radius_default] if closing_enable else [0.0])))
    if not closing_radii:
        closing_radii = [0.0]

    corridor_for_close = corridor_geom if closing_only_corridor else raw_union
    fill_enable = bool(cfg.get("FILL_HOLES_ENABLE", True))
    hole_fill_max = float(cfg.get("HOLE_FILL_MAX_M2", 180.0))
    hole_fill_pad = float(cfg.get("HOLE_FILL_CORRIDOR_PAD_M", 3.0))
    hole_fill_corridor_only = bool(cfg.get("HOLE_FILL_ONLY_INTERSECT_CORRIDOR", True))
    corridor_for_fill = corridor_geom if hole_fill_corridor_only else None

    closing_trials: List[Dict[str, object]] = []

    def _close_geom(radius_m: float):
        if not closing_enable or radius_m <= 0.0:
            return raw_union
        return close_candidates_in_corridor(raw_union, corridor_for_close, radius_m, closing_pad_m)

    def _fill_geom(geom_in, max_area_m2: float):
        if geom_in is None or geom_in.is_empty:
            return geom_in
        if not fill_enable or max_area_m2 <= 0.0:
            return geom_in
        return fill_small_holes_by_corridor(
            geom_in,
            corridor_for_fill,
            max_hole_area_m2=max_area_m2,
            corridor_pad_m=hole_fill_pad,
        )

    def _closing_score(
        holes_ratio: float,
        holes_count: int,
        outside_ratio: float,
        corridor_cov: float,
        outside_best: float,
        corridor_best: float,
    ) -> float:
        holes_count_norm = float(holes_count) / max(float(raw_holes_count), 1.0)
        outside_pen = 0.0
        if outside_ratio > outside_best + 0.002:
            outside_pen = 100.0 + (outside_ratio - (outside_best + 0.002)) * 1000.0
        corridor_pen = 0.0
        if corridor_cov < corridor_best - 0.01:
            corridor_pen = 50.0 + ((corridor_best - 0.01) - corridor_cov) * 500.0
        return 3.0 * holes_ratio + 2.0 * holes_count_norm + 4.0 * outside_pen + 2.0 * corridor_pen

    best_close = None
    best_closed_geom = raw_union
    best_final_geom = raw_union
    best_hole_fill_max = hole_fill_max
    best_rollback_used = False

    for radius_m in closing_radii:
        closed_geom = _close_geom(radius_m)
        if closing_only_corridor and corridor_geom is not None and not corridor_geom.is_empty:
            closed_geom = closed_geom.intersection(corridor_geom)
        closed_holes_count, closed_holes_area, closed_holes_ratio = _hole_stats_from_geom(closed_geom)
        closed_corridor_cov, closed_outside = _vector_coverage_and_outside(closed_geom, corridor_geom, roi_geom)

        hole_fill_used_max = hole_fill_max
        rollback_used = False
        final_geom = _fill_geom(closed_geom, hole_fill_used_max)
        if closing_only_corridor and corridor_geom is not None and not corridor_geom.is_empty:
            final_geom = final_geom.intersection(corridor_geom)
        final_holes_count, final_holes_area, final_holes_ratio = _hole_stats_from_geom(final_geom)
        final_corridor_cov, final_outside = _vector_coverage_and_outside(final_geom, corridor_geom, roi_geom)
        outside_delta = final_outside - raw_outside_vec
        corridor_delta = final_corridor_cov - raw_corridor_cov_vec

        # One rollback attempt if filling causes spill or corridor loss.
        if outside_delta > 0.002 or corridor_delta < -0.01:
            hole_fill_used_max = max(60.0, hole_fill_max * 0.6)
            final_geom = _fill_geom(closed_geom, hole_fill_used_max)
            if closing_only_corridor and corridor_geom is not None and not corridor_geom.is_empty:
                final_geom = final_geom.intersection(corridor_geom)
            final_holes_count, final_holes_area, final_holes_ratio = _hole_stats_from_geom(final_geom)
            final_corridor_cov, final_outside = _vector_coverage_and_outside(final_geom, corridor_geom, roi_geom)
            outside_delta = final_outside - raw_outside_vec
            corridor_delta = final_corridor_cov - raw_corridor_cov_vec
            rollback_used = True

        score_close = _closing_score(
            final_holes_ratio,
            final_holes_count,
            final_outside,
            final_corridor_cov,
            outside_best=raw_outside_vec,
            corridor_best=raw_corridor_cov_vec,
        )

        closing_trials.append(
            {
                "trial_id": f"closing_r{radius_m}",
                "stage": "closing",
                "CLOSING_RADIUS_M": radius_m,
                "hole_fill_max_m2": hole_fill_used_max,
                "closed_holes_count": closed_holes_count,
                "closed_holes_area_m2": closed_holes_area,
                "closed_holes_ratio": closed_holes_ratio,
                "closed_outside_ratio_vec": closed_outside,
                "closed_corridor_cov_vec": closed_corridor_cov,
                "final_holes_count": final_holes_count,
                "final_holes_area_m2": final_holes_area,
                "final_holes_ratio": final_holes_ratio,
                "final_outside_ratio_vec": final_outside,
                "final_corridor_cov_vec": final_corridor_cov,
                "outside_delta": outside_delta,
                "corridor_delta": corridor_delta,
                "score": score_close,
                "status": "ok",
            }
        )

        if best_close is None or score_close < float(best_close["score"]):
            best_close = closing_trials[-1]
            best_closed_geom = closed_geom
            best_final_geom = final_geom
            best_hole_fill_max = hole_fill_used_max
            best_rollback_used = rollback_used

    # Rebuild closed/final candidates from selected geometries.
    closed_candidates = _gdf_from_geom(best_closed_geom, drive_id, target_epsg, bundle.res_m, min_area_m2)
    filled_candidates = _gdf_from_geom(best_final_geom, drive_id, target_epsg, bundle.res_m, min_area_m2)

    closed_holes_count, closed_holes_area, closed_holes_ratio = _hole_stats_from_geom(best_closed_geom)
    closed_corridor_cov_vec, closed_outside_vec = _vector_coverage_and_outside(best_closed_geom, corridor_geom, roi_geom)
    filled_holes_count, filled_holes_area, filled_holes_ratio = _hole_stats_from_geom(best_final_geom)
    filled_corridor_cov_vec, filled_outside_vec = _vector_coverage_and_outside(best_final_geom, corridor_geom, roi_geom)
    outside_delta = filled_outside_vec - raw_outside_vec
    corridor_delta = filled_corridor_cov_vec - raw_corridor_cov_vec

    chosen_radius = float(best_close["CLOSING_RADIUS_M"]) if best_close is not None else 0.0
    best_params_out = dict(best_params)
    best_params_out["CLOSING_RADIUS_M"] = chosen_radius
    best_params_out["CLOSING_PAD_M"] = closing_pad_m
    best_params_out["HOLE_FILL_MAX_M2"] = best_hole_fill_max
    with (autotune_dir / "best_params.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(best_params_out, f, sort_keys=False, allow_unicode=False)

    hole_fill_summary = {
        "enabled": fill_enable,
        "max_hole_area_m2": best_hole_fill_max,
        "corridor_pad_m": hole_fill_pad,
        "corridor_only": hole_fill_corridor_only,
        "rollback_used": best_rollback_used,
        "closing": {
            "enabled": closing_enable,
            "corridor_only": closing_only_corridor,
            "corridor_pad_m": closing_pad_m,
            "radius_candidates_m": closing_radii,
            "chosen_radius_m": chosen_radius,
            "best_score": float(best_close["score"]) if best_close is not None else None,
        },
        "raw": {
            "holes_count": raw_holes_count,
            "holes_area_m2": raw_holes_area,
            "holes_area_ratio": raw_holes_ratio,
            "outside_ratio_vec": raw_outside_vec,
            "corridor_coverage_vec": raw_corridor_cov_vec,
        },
        "closed": {
            "holes_count": closed_holes_count,
            "holes_area_m2": closed_holes_area,
            "holes_area_ratio": closed_holes_ratio,
            "outside_ratio_vec": closed_outside_vec,
            "corridor_coverage_vec": closed_corridor_cov_vec,
        },
        "filled": {
            "holes_count": filled_holes_count,
            "holes_area_m2": filled_holes_area,
            "holes_area_ratio": filled_holes_ratio,
            "outside_ratio_vec": filled_outside_vec,
            "corridor_coverage_vec": filled_corridor_cov_vec,
        },
        "outside_delta": outside_delta,
        "corridor_delta": corridor_delta,
    }

    if closing_trials:
        trials_df = pd.concat([trials_df, pd.DataFrame(closing_trials)], ignore_index=True)
    trials_df.to_csv(autotune_dir / "trials.csv", index=False, encoding="utf-8")
    trials_total = int(len(trials_df))

    mask_path = evidence_dir / "dop20_roadmask_utm32.tif"
    _write_raster(mask_path, best_mask.astype("uint8"), bundle.transform, target_epsg, 0.0, warnings)

    evid_ids = [str(uuid4()), str(uuid4())]
    evidence_rows = [
        {
            "evid_id": evid_ids[0],
            "source": "sat",
            "drive_id": drive_id,
            "evid_type": "dop20_mosaic",
            "crs_epsg": target_epsg,
            "path_rel": relpath(run_dir, mosaic_path),
            "frame_start": -1,
            "frame_end": -1,
            "conf": 1.0,
            "uncert": bundle.res_m,
            "meta_json": json.dumps({"tiles": bundle.tile_count, "res_m": bundle.res_m}),
            "geometry": roi_geom,
        },
        {
            "evid_id": evid_ids[1],
            "source": "sat",
            "drive_id": drive_id,
            "evid_type": "road_surface_mask",
            "crs_epsg": target_epsg,
            "path_rel": relpath(run_dir, mask_path),
            "frame_start": -1,
            "frame_end": -1,
            "conf": 1.0,
            "uncert": bundle.res_m,
            "meta_json": json.dumps(
                {
                    "best_params": best_params_out,
                    "best_metrics": {
                        "mask_coverage": best["mask_coverage"],
                        "corridor_coverage": best["corridor_coverage"],
                        "outside_ratio": best["outside_ratio"],
                        "components_count": best["components_count"],
                        "largest_roi_ratio": best["largest_roi_ratio"],
                    },
                }
            ),
            "geometry": roi_geom,
        },
    ]
    evidence_gdf = gpd.GeoDataFrame(evidence_rows, geometry="geometry", crs=f"EPSG:{target_epsg}")
    ensure_required_columns(evidence_gdf, PRIMITIVE_FIELDS)
    evidence_path = evidence_dir / "dop20_evidence_utm32.gpkg"
    write_gpkg_layer(evidence_path, "primitive_evidence", evidence_gdf, target_epsg, warnings, overwrite=True)

    raw_out = raw_candidates.copy()
    raw_out["evid_ref"] = json.dumps(evid_ids)
    raw_candidates_path = candidates_dir / "dop20_candidates_raw_utm32.gpkg"
    write_gpkg_layer(raw_candidates_path, "world_candidates", raw_out, target_epsg, warnings, overwrite=True)

    closed_out = closed_candidates.copy()
    closed_out["evid_ref"] = json.dumps(evid_ids)
    closed_candidates_path = candidates_dir / "dop20_candidates_closed_utm32.gpkg"
    write_gpkg_layer(closed_candidates_path, "world_candidates", closed_out, target_epsg, warnings, overwrite=True)

    filled_out = filled_candidates.copy()
    filled_out["evid_ref"] = json.dumps(evid_ids)
    candidates_path = candidates_dir / "dop20_candidates_utm32.gpkg"
    write_gpkg_layer(candidates_path, "world_candidates", filled_out, target_epsg, warnings, overwrite=True)

    qa_row = {
        "drive_id": drive_id,
        "roi_path": relpath(run_dir, roi_path),
        "osm_roads_path": relpath(run_dir, osm_roads_path),
        "osm_corridor_path": relpath(run_dir, corridor_path),
        "mosaic_path": relpath(run_dir, mosaic_path),
        "mask_path": relpath(run_dir, mask_path),
        "candidates_path": relpath(run_dir, candidates_path),
        "candidates_raw_path": relpath(run_dir, raw_candidates_path),
        "candidates_closed_path": relpath(run_dir, closed_candidates_path),
        "minx": roi_geom.bounds[0],
        "miny": roi_geom.bounds[1],
        "maxx": roi_geom.bounds[2],
        "maxy": roi_geom.bounds[3],
        "mask_coverage": best["mask_coverage"],
        "corridor_coverage": best["corridor_coverage"],
        "outside_ratio": best["outside_ratio"],
        "components_count": best["components_count"],
        "largest_roi_ratio": best["largest_roi_ratio"],
        "tiles_hit": bundle.tile_count,
        "osm_source": osm_source_note,
        "tiles_source": tiles_source,
        "closing_radius_m": hole_fill_summary.get("closing", {}).get("chosen_radius_m"),
        "holes_count_raw": hole_fill_summary.get("raw", {}).get("holes_count"),
        "holes_area_m2_raw": hole_fill_summary.get("raw", {}).get("holes_area_m2"),
        "holes_area_ratio_raw": hole_fill_summary.get("raw", {}).get("holes_area_ratio"),
        "holes_count_closed": hole_fill_summary.get("closed", {}).get("holes_count"),
        "holes_area_m2_closed": hole_fill_summary.get("closed", {}).get("holes_area_m2"),
        "holes_area_ratio_closed": hole_fill_summary.get("closed", {}).get("holes_area_ratio"),
        "holes_count_filled": hole_fill_summary.get("filled", {}).get("holes_count"),
        "holes_area_m2_filled": hole_fill_summary.get("filled", {}).get("holes_area_m2"),
        "holes_area_ratio_filled": hole_fill_summary.get("filled", {}).get("holes_area_ratio"),
        "outside_ratio_vec_raw": hole_fill_summary.get("raw", {}).get("outside_ratio_vec"),
        "outside_ratio_vec_closed": hole_fill_summary.get("closed", {}).get("outside_ratio_vec"),
        "outside_ratio_vec_filled": hole_fill_summary.get("filled", {}).get("outside_ratio_vec"),
        "corridor_cov_vec_raw": hole_fill_summary.get("raw", {}).get("corridor_coverage_vec"),
        "corridor_cov_vec_closed": hole_fill_summary.get("closed", {}).get("corridor_coverage_vec"),
        "corridor_cov_vec_filled": hole_fill_summary.get("filled", {}).get("corridor_coverage_vec"),
        "hole_fill_outside_delta": hole_fill_summary.get("outside_delta"),
        "hole_fill_corridor_delta": hole_fill_summary.get("corridor_delta"),
        "hole_fill_enabled": hole_fill_summary.get("enabled"),
    }
    write_csv(qa_dir / "qa_index.csv", [qa_row], list(qa_row.keys()))

    init_note = ""
    if initial_metrics is not None:
        init_note = (
            f"- initial_mask_coverage: {initial_metrics['mask_coverage']:.4f}\n"
            f"- initial_corridor_coverage: {initial_metrics['corridor_coverage']:.4f}\n"
            f"- initial_outside_ratio: {initial_metrics['outside_ratio']:.4f}\n"
            f"- initial_score: {initial_metrics['score']:.4f}\n"
        )

    report_lines = [
        "# DOP20 Tune010 Report",
        "",
        f"- drive_id: {drive_id}",
        f"- osm_source: {osm_source_note}",
        f"- tiles_source: {tiles_source}",
        f"- trials: {trials_total}",
        "",
        "## Initial vs Best",
        init_note.rstrip() if init_note else "- initial: n/a",
        f"- best_mask_coverage: {best['mask_coverage']:.4f}",
        f"- best_corridor_coverage: {best['corridor_coverage']:.4f}",
        f"- best_outside_ratio: {best['outside_ratio']:.4f}",
        f"- best_components_count: {best['components_count']:.0f}",
        f"- best_largest_roi_ratio: {best['largest_roi_ratio']:.4f}",
        f"- best_score: {best['score']:.4f}",
        "",
        "## Best Params",
        "```yaml",
        yaml.safe_dump(best_params_out, sort_keys=False, allow_unicode=False).rstrip(),
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
            f"- {relpath(run_dir, osm_roads_path)}",
            f"- {relpath(run_dir, corridor_path)}",
            f"- {relpath(run_dir, mosaic_path)}",
            f"- {relpath(run_dir, mask_path)}",
            f"- {relpath(run_dir, evidence_path)}",
            f"- {relpath(run_dir, raw_candidates_path)}",
            f"- {relpath(run_dir, closed_candidates_path)}",
            f"- {relpath(run_dir, candidates_path)}",
            f"- {relpath(run_dir, autotune_dir / 'trials.csv')}",
            f"- {relpath(run_dir, autotune_dir / 'best_params.yaml')}",
        ]
    )
    if hole_fill_summary.get("enabled") or hole_fill_summary.get("closing", {}).get("enabled"):
        raw = hole_fill_summary.get("raw", {})
        closed = hole_fill_summary.get("closed", {})
        filled = hole_fill_summary.get("filled", {})
        closing = hole_fill_summary.get("closing", {})
        report_lines.extend(
            [
                "",
                "## Closing + Hole Filling",
                f"- closing_enabled: {closing.get('enabled')}",
                f"- closing_radius_m: {closing.get('chosen_radius_m')}",
                f"- closing_pad_m: {closing.get('corridor_pad_m')}",
                f"- max_hole_area_m2: {hole_fill_summary.get('max_hole_area_m2')}",
                f"- corridor_pad_m: {hole_fill_summary.get('corridor_pad_m')}",
                f"- corridor_only: {hole_fill_summary.get('corridor_only')}",
                f"- rollback_used: {hole_fill_summary.get('rollback_used')}",
                f"- holes_count_raw: {raw.get('holes_count')}",
                f"- holes_count_closed: {closed.get('holes_count')}",
                f"- holes_count_filled: {filled.get('holes_count')}",
                f"- holes_area_m2_raw: {raw.get('holes_area_m2')}",
                f"- holes_area_m2_closed: {closed.get('holes_area_m2')}",
                f"- holes_area_m2_filled: {filled.get('holes_area_m2')}",
                f"- holes_area_ratio_raw: {raw.get('holes_area_ratio')}",
                f"- holes_area_ratio_closed: {closed.get('holes_area_ratio')}",
                f"- holes_area_ratio_filled: {filled.get('holes_area_ratio')}",
                f"- outside_ratio_vec_raw: {raw.get('outside_ratio_vec')}",
                f"- outside_ratio_vec_closed: {closed.get('outside_ratio_vec')}",
                f"- outside_ratio_vec_filled: {filled.get('outside_ratio_vec')}",
                f"- corridor_cov_vec_raw: {raw.get('corridor_coverage_vec')}",
                f"- corridor_cov_vec_closed: {closed.get('corridor_coverage_vec')}",
                f"- corridor_cov_vec_filled: {filled.get('corridor_coverage_vec')}",
                f"- outside_delta: {hole_fill_summary.get('outside_delta')}",
                f"- corridor_delta: {hole_fill_summary.get('corridor_delta')}",
            ]
        )
        if float(hole_fill_summary.get("outside_delta") or 0.0) > 0.002:
            report_lines.append("- warning: outside_ratio increased > 0.002; consider lowering closing radius or HOLE_FILL_MAX_M2.")
        if float(hole_fill_summary.get("corridor_delta") or 0.0) < -0.01:
            report_lines.append("- warning: corridor_coverage dropped > 0.01; consider lowering closing radius.")
    write_text(qa_dir / "report.md", "\n".join(report_lines))

    summary = {
        "run_id": run_id,
        "drive_id": drive_id,
        "dop20_root": str(dop20_root),
        "tiles_source": tiles_source,
        "osm_source": osm_source_note,
        "trials": trials_total,
        "best": {
            "params": best_params_out,
            "metrics": {
                "mask_coverage": float(best["mask_coverage"]),
                "corridor_coverage": float(best["corridor_coverage"]),
                "outside_ratio": float(best["outside_ratio"]),
                "components_count": float(best["components_count"]),
                "largest_roi_ratio": float(best["largest_roi_ratio"]),
                "score": float(best["score"]),
            },
        },
        "initial": {
            "mask_coverage": float(initial_metrics["mask_coverage"]) if initial_metrics is not None else None,
            "corridor_coverage": float(initial_metrics["corridor_coverage"]) if initial_metrics is not None else None,
            "outside_ratio": float(initial_metrics["outside_ratio"]) if initial_metrics is not None else None,
            "score": float(initial_metrics["score"]) if initial_metrics is not None else None,
        },
        "hole_fill": hole_fill_summary,
        "roi_errors": roi_errors,
        "tile_errors": tile_errors,
        "warnings": warnings,
        "errors": errors,
    }
    write_json(run_dir / "run_summary.json", summary)
    summary_md = [
        "# DOP20 Tune010 Summary",
        "",
        f"- run_id: {run_id}",
        f"- drive_id: {drive_id}",
        f"- trials: {trials_total}",
        f"- best_score: {best['score']:.4f}",
        f"- best_corridor_coverage: {best['corridor_coverage']:.4f}",
        f"- best_outside_ratio: {best['outside_ratio']:.4f}",
        "",
        "## Notes",
        "- OSM corridor is a soft prior QA constraint, not ground truth.",
    ]
    write_text(run_dir / "run_summary.md", "\n".join(summary_md))
    LOG.info("completed tune010 run: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape

from pipeline.lidar_semantic.accum_points_world import AccumResult, accumulate_world_points
from pipeline.lidar_semantic.build_rasters import RasterBundle, build_rasters
from pipeline.lidar_semantic.build_roi_corridor import RoiResult, build_roi_corridor, roi_to_gdf
from pipeline.lidar_semantic.classify_road import RoadResult, classify_road
from pipeline.lidar_semantic.detect_crosswalk import CrosswalkResult, detect_crosswalks
from pipeline.lidar_semantic.export_pointcloud import write_las
from pipeline.post.morph_close import close_candidates_in_corridor
from pipeline.utils.config_resolve import get_params_hash, load_yaml, resolve_config, update_resolved_config
from pipeline.utils.postcheck import postcheck
from scripts.pipeline_common import (
    LOG,
    ensure_dir,
    ensure_overwrite,
    ensure_required_columns,
    now_ts,
    relpath,
    setup_logging,
    validate_output_crs,
    write_csv,
    write_json,
    write_text,
    write_gpkg_layer,
)

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
class MarkingResult:
    marking_score: np.ndarray
    marking_mask: np.ndarray
    markings_points_mask: np.ndarray
    markings_polygons: gpd.GeoDataFrame
    stats: Dict[str, float]
    score_thresh: float


def _write_raster(path: Path, array: np.ndarray, bundle: RasterBundle, epsg: int, warnings: List[str]) -> None:
    validate_output_crs(path, epsg, None, warnings)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = array.astype(np.float32)
    nodata = np.nan
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=bundle.height,
        width=bundle.width,
        count=1,
        dtype="float32",
        crs=f"EPSG:{epsg}",
        transform=bundle.transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def _mask_raster(path: Path, mask: np.ndarray, bundle: RasterBundle, epsg: int, warnings: List[str]) -> None:
    validate_output_crs(path, epsg, None, warnings)
    path.parent.mkdir(parents=True, exist_ok=True)
    nodata = 255
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=bundle.height,
        width=bundle.width,
        count=1,
        dtype="uint8",
        crs=f"EPSG:{epsg}",
        transform=bundle.transform,
        nodata=nodata,
    ) as dst:
        dst.write(mask.astype("uint8"), 1)


def _write_meta(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _group_slices(lin_idx: np.ndarray):
    order = np.argsort(lin_idx, kind="mergesort")
    lin_sorted = lin_idx[order]
    uniq, start, counts = np.unique(lin_sorted, return_index=True, return_counts=True)
    return order, uniq, start, counts


def _intensity_ground_p95(
    points_xyz: np.ndarray,
    intensity: np.ndarray,
    bundle: RasterBundle,
    ground_band_dz_m: float,
) -> np.ndarray:
    height = bundle.height
    width = bundle.width
    p95 = np.full((height, width), np.nan, dtype=np.float32)
    if points_xyz.size == 0:
        return p95
    valid = bundle.point_valid
    if not np.any(valid):
        return p95
    ix = bundle.point_ix[valid]
    iy = bundle.point_iy[valid]
    z = points_xyz[valid, 2]
    inten = intensity[valid]
    h_ref = bundle.point_height_p10[valid]
    ground = np.isfinite(h_ref) & (z <= (h_ref + float(ground_band_dz_m)))
    if not np.any(ground):
        return p95
    ixg = ix[ground]
    iyg = iy[ground]
    ig = inten[ground]
    lin = iyg.astype(np.int64) * int(width) + ixg.astype(np.int64)
    order, uniq, start, counts = _group_slices(lin)
    ig_sorted = ig[order]
    ix_sorted = ixg[order]
    iy_sorted = iyg[order]
    for u, s, c in zip(uniq, start, counts):
        if c <= 0:
            continue
        sl = slice(s, s + c)
        cx = int(ix_sorted[s])
        cy = int(iy_sorted[s])
        p95[cy, cx] = float(np.percentile(ig_sorted[sl], 95))
    return p95


def _box_mean(values: np.ndarray, valid_mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return np.where(valid_mask, values, np.nan)
    v = np.where(valid_mask, values, 0.0).astype(np.float32)
    m = valid_mask.astype(np.float32)
    pad = int(radius_px)
    v_pad = np.pad(v, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
    m_pad = np.pad(m, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
    v_int = np.pad(v_pad, ((1, 0), (1, 0)), mode="constant", constant_values=0.0).cumsum(axis=0).cumsum(axis=1)
    m_int = np.pad(m_pad, ((1, 0), (1, 0)), mode="constant", constant_values=0.0).cumsum(axis=0).cumsum(axis=1)
    h, w = values.shape
    y0 = np.arange(0, h)
    y1 = y0 + 2 * pad
    x0 = np.arange(0, w)
    x1 = x0 + 2 * pad
    out = np.zeros((h, w), dtype=np.float32)
    for i, (yy0, yy1) in enumerate(zip(y0, y1)):
        sum_row = v_int[yy1 + 1, x1 + 1] - v_int[yy0, x1 + 1] - v_int[yy1 + 1, x0] + v_int[yy0, x0]
        cnt_row = m_int[yy1 + 1, x1 + 1] - m_int[yy0, x1 + 1] - m_int[yy1 + 1, x0] + m_int[yy0, x0]
        out[i, :] = np.where(cnt_row > 0, sum_row / cnt_row, 0.0)
    return out


def _mask_to_polygons(mask: np.ndarray, bundle: RasterBundle, min_area_m2: float) -> gpd.GeoDataFrame:
    geoms = []
    for geom, val in features.shapes(mask.astype("uint8"), mask=mask.astype(bool), transform=bundle.transform):
        if int(val) != 1:
            continue
        poly = shape(geom)
        if poly.is_empty:
            continue
        if float(poly.area) < min_area_m2:
            continue
        geoms.append(poly)
    if not geoms:
        return gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32632")
    return gpd.GeoDataFrame([{"geometry": g} for g in geoms], geometry="geometry", crs="EPSG:32632")


def _polys_to_world_candidates(
    polys: gpd.GeoDataFrame,
    drive_id: str,
    cls: str,
    epsg: int,
    res_m: float,
    params_hash: str,
) -> gpd.GeoDataFrame:
    if polys is None or polys.empty:
        return gpd.GeoDataFrame(columns=WORLD_FIELDS + ["geometry"], geometry=[], crs=f"EPSG:{epsg}")
    rows = []
    for geom in polys.geometry:
        if geom is None or geom.is_empty:
            continue
        rows.append(
            {
                "cand_id": str(np.random.default_rng().integers(0, 2**63 - 1)),
                "source": "lidar",
                "drive_id": drive_id,
                "class": cls,
                "crs_epsg": epsg,
                "conf": 0.7,
                "uncert": res_m,
                "evid_ref": "[]",
                "conflict": "",
                "attr_json": json.dumps({"area_m2": round(float(geom.area), 2), "params_hash": params_hash}, ensure_ascii=True),
                "geometry": geom,
            }
        )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{epsg}")
    ensure_required_columns(gdf, WORLD_FIELDS)
    return gdf


def _select_marking_threshold(
    score: np.ndarray,
    road_mask: np.ndarray,
    ratio_min: float,
    ratio_max: float,
) -> Tuple[float, float, float]:
    road_scores = score[road_mask.astype(bool)]
    road_scores = road_scores[np.isfinite(road_scores)]
    if road_scores.size == 0:
        return 1.0, 0.0, 100.0
    target = float((ratio_min + ratio_max) * 0.5)
    target = max(1e-6, min(target, 0.999))
    thr = float(np.quantile(road_scores, 1.0 - target))
    ratio = float(np.mean(road_scores >= thr))
    if ratio < ratio_min:
        thr = float(np.quantile(road_scores, 1.0 - float(ratio_min)))
        ratio = float(np.mean(road_scores >= thr))
    elif ratio > ratio_max:
        thr = float(np.quantile(road_scores, 1.0 - float(ratio_max)))
        ratio = float(np.mean(road_scores >= thr))
    pctl = float((1.0 - ratio) * 100.0)
    return thr, ratio, pctl


def _compute_markings_v2(
    bundle: RasterBundle,
    points_xyz: np.ndarray,
    intensity: np.ndarray,
    road_points_mask: np.ndarray,
    road_mask: np.ndarray,
    corridor_geom: object,
    cfg: Dict[str, object],
) -> Tuple[MarkingResult, np.ndarray, np.ndarray]:
    ground_p95 = _intensity_ground_p95(points_xyz, intensity, bundle, float(cfg["GROUND_BAND_DZ_M"]))
    valid = np.isfinite(ground_p95)
    radius_px = int(round(float(cfg["BG_WIN_RADIUS_M"]) / float(bundle.res_m)))
    bg_mean = _box_mean(np.nan_to_num(ground_p95, nan=0.0), valid, radius_px)
    score_raw = np.maximum(0.0, ground_p95 - bg_mean)
    road_bool = road_mask.astype(bool)
    score_raw = np.where(road_bool, score_raw, 0.0)
    score_vals = score_raw[road_bool]
    if score_vals.size == 0:
        score_norm = np.zeros_like(score_raw, dtype=np.float32)
    else:
        p99 = float(np.percentile(score_vals, 99.0))
        p99 = max(p99, 1e-6)
        score_norm = np.clip(score_raw / p99, 0.0, 1.0).astype(np.float32)

    thr, ratio, pctl = _select_marking_threshold(
        score_norm,
        road_bool,
        float(cfg["MARKING_AREA_RATIO_MIN"]),
        float(cfg["MARKING_AREA_RATIO_MAX"]),
    )

    cfg["MARKING_SCORE_TH"] = float(thr)
    cfg["MARKING_SCORE_PCTL_CHOSEN"] = float(pctl)

    marking_mask = (score_norm >= float(thr)) & road_bool

    min_area_m2 = max(0.5, bundle.res_m * bundle.res_m * 2.0)
    polys = _mask_to_polygons(marking_mask, bundle, min_area_m2=min_area_m2)
    if not polys.empty and float(cfg["MARKING_CLOSE_RADIUS_M"]) > 0:
        union = polys.unary_union
        closed = close_candidates_in_corridor(union, corridor_geom, float(cfg["MARKING_CLOSE_RADIUS_M"]), corridor_pad_m=0.0)
        if closed is not None and not closed.is_empty:
            marking_mask = features.rasterize(
                [(closed, 1)],
                out_shape=(bundle.height, bundle.width),
                transform=bundle.transform,
                fill=0,
                dtype="uint8",
            ).astype(bool)
            polys = _mask_to_polygons(marking_mask, bundle, min_area_m2=min_area_m2)
        else:
            marking_mask = np.zeros_like(marking_mask, dtype=bool)
            polys = gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32632")

    ix = bundle.point_ix
    iy = bundle.point_iy
    valid_pts = bundle.point_valid
    road_cell = np.zeros_like(valid_pts, dtype=bool)
    mark_cell = np.zeros_like(valid_pts, dtype=bool)
    if np.any(valid_pts):
        road_cell[valid_pts] = road_bool[iy[valid_pts], ix[valid_pts]]
        mark_cell[valid_pts] = marking_mask[iy[valid_pts], ix[valid_pts]]

    ground_band = valid_pts & np.isfinite(bundle.point_height_p10) & (
        points_xyz[:, 2] <= (bundle.point_height_p10 + float(cfg["GROUND_BAND_DZ_M"]))
    )
    markings_points_mask = road_cell & mark_cell & ground_band

    marking_area = float(polys.geometry.area.sum()) if not polys.empty else 0.0
    road_area = float(np.sum(road_bool) * (bundle.res_m * bundle.res_m))
    stats = {
        "marking_area_m2": marking_area,
        "marking_cover_on_road": marking_area / max(road_area, 1e-6),
        "marking_points_ratio": float(markings_points_mask.sum()) / max(float(road_points_mask.sum()), 1.0),
        "score_thresh": float(thr),
        "score_ratio_on_road": float(ratio),
    }
    return (
        MarkingResult(
            marking_score=score_norm.astype(np.float32),
            marking_mask=marking_mask.astype("uint8"),
            markings_points_mask=markings_points_mask,
            markings_polygons=polys,
            stats=stats,
            score_thresh=float(thr),
        ),
        ground_p95,
        bg_mean,
    )


def _write_pointcloud(
    path: Path,
    points_xyz: np.ndarray,
    intensity: np.ndarray,
    classification: np.ndarray,
    epsg: int,
) -> List[Path]:
    return write_las(path, points_xyz, intensity, classification, epsg)


def _points_in_polys(points_xyz: np.ndarray, polys: gpd.GeoDataFrame) -> np.ndarray:
    if points_xyz.size == 0 or polys is None or polys.empty:
        return np.zeros((points_xyz.shape[0],), dtype=bool)
    union = polys.unary_union
    try:
        from shapely.prepared import prep

        u = prep(union)
        if hasattr(u, "contains_xy"):
            return np.array([bool(u.contains_xy(x, y)) for x, y in points_xyz[:, :2]], dtype=bool)
        from shapely.geometry import Point

        return np.array([bool(u.contains(Point(x, y))) for x, y in points_xyz[:, :2]], dtype=bool)
    except Exception:
        return np.zeros((points_xyz.shape[0],), dtype=bool)


def _select_drive_0010(drives: List[str]) -> str:
    for d in drives:
        if "_0010_" in d:
            return d
    raise RuntimeError("no_0010_drive_found")


def _load_drives(data_root: Path, drives_file: str) -> List[str]:
    path = Path(drives_file)
    if path.exists():
        drives = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if drives:
            return drives
    raw_root = data_root / "data_3d_raw"
    if raw_root.exists():
        drives = sorted([p.name for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("2013_05_28_drive_")])
        return drives
    return []


def _apply_degrade(cfg: Dict[str, object], step: int) -> Optional[str]:
    if step == 0 and float(cfg["VOXEL_SIZE_M"]) < 0.08:
        cfg["VOXEL_SIZE_M"] = 0.08
        return "degrade:voxel=0.08"
    if step == 1 and float(cfg["VOXEL_SIZE_M"]) < 0.10:
        cfg["VOXEL_SIZE_M"] = 0.10
        return "degrade:voxel=0.10"
    if step == 2 and float(cfg["RASTER_RES_M"]) < 0.30:
        cfg["RASTER_RES_M"] = 0.30
        return "degrade:raster=0.30"
    if step == 3 and float(cfg["RASTER_RES_M"]) < 0.50:
        cfg["RASTER_RES_M"] = 0.50
        return "degrade:raster=0.50"
    if step == 4 and int(cfg["MAX_FRAMES"]) > 2000:
        cfg["MAX_FRAMES"] = 2000
        return "degrade:max_frames=2000"
    if step == 5 and int(cfg["MAX_FRAMES"]) > 1500:
        cfg["MAX_FRAMES"] = 1500
        return "degrade:max_frames=1500"
    if step == 6 and float(cfg["BG_WIN_RADIUS_M"]) > 2.0:
        cfg["BG_WIN_RADIUS_M"] = 2.0
        return "degrade:bg_win=2.0"
    return None


def main() -> int:
    base_cfg = load_yaml(Path("configs/lidar_semantic_v2_0010.yaml"))
    run_id = now_ts()
    run_dir = Path("runs") / f"lidar_semantic_v2_0010_{run_id}"
    if bool(base_cfg.get("OVERWRITE", True)):
        ensure_overwrite(run_dir)
    ensure_dir(run_dir)
    setup_logging(run_dir / "run.log")

    cfg = resolve_config(base_cfg, run_dir)
    params_hash = get_params_hash(cfg)

    warnings: List[str] = []
    errors: List[str] = []
    drive_notes: List[str] = []

    data_root = Path(str(cfg["KITTI_ROOT"]))
    if not data_root.exists():
        write_text(run_dir / "run_summary.md", f"# lidar_semantic_v2_0010\n\n- status: fail\n- reason: KITTI_ROOT_missing\n")
        return 2

    drives = _load_drives(data_root, str(cfg.get("GOLDEN_DRIVES_FILE")))
    drive_id = _select_drive_0010(drives)
    if len([d for d in drives if d == drive_id]) != 1:
        raise RuntimeError("drive_selection_failed")

    target_epsg = int(cfg["TARGET_EPSG"])
    budget_s = float(cfg["TIME_BUDGET_H"]) * 3600.0
    start_t = time.perf_counter()

    drive_dir = run_dir / "drives" / drive_id
    roi_dir = ensure_dir(drive_dir / "roi")
    ras_dir = ensure_dir(drive_dir / "rasters")
    pc_dir = ensure_dir(drive_dir / "pointcloud")
    vec_dir = ensure_dir(drive_dir / "vectors")
    qa_dir = ensure_dir(drive_dir / "qa")
    docs_dir = ensure_dir(run_dir / "docs")

    roi_res: RoiResult = build_roi_corridor(
        data_root=data_root,
        drive_id=drive_id,
        buffer_m=float(cfg["ROI_BUFFER_M"]),
        stride=int(cfg["TRAJ_STRIDE"]),
        target_epsg=target_epsg,
    )
    if roi_res.errors:
        warnings.extend([f"roi:{e}" for e in roi_res.errors])
    roi_gdf = roi_to_gdf(roi_res.roi_geom, drive_id, target_epsg)
    roi_path = roi_dir / "roi_corridor_utm32.gpkg"
    write_gpkg_layer(roi_path, "roi", roi_gdf, target_epsg, warnings, overwrite=True)

    accum: Optional[AccumResult] = None
    degrade_steps = 7
    step = 0
    while True:
        try:
            accum = accumulate_world_points(
                data_root=data_root,
                drive_id=drive_id,
                roi_geom=roi_res.roi_geom,
                mode=str(cfg["LIDAR_WORLD_MODE"]),
                cam_id="image_00",
                stride=int(cfg["STRIDE"]),
                max_frames=int(cfg["MAX_FRAMES"]),
                voxel_size_m=float(cfg["VOXEL_SIZE_M"]),
            )
            break
        except MemoryError:
            if step >= degrade_steps:
                raise
            note = _apply_degrade(cfg, step)
            if note:
                drive_notes.append(note)
                params_hash = update_resolved_config(cfg, run_dir)
            step += 1
        except Exception as exc:
            errors.append(f"accum:{exc}")
            if step >= degrade_steps:
                raise
            note = _apply_degrade(cfg, step)
            if note:
                drive_notes.append(note)
                params_hash = update_resolved_config(cfg, run_dir)
            step += 1
    if accum is None:
        raise RuntimeError("accum_failed")

    warnings.extend(accum.errors)

    bundle = build_rasters(
        accum.points_xyz,
        accum.intensity,
        roi_geom=roi_res.roi_geom,
        res_m=float(cfg["RASTER_RES_M"]),
        ground_band_dz_m=float(cfg["GROUND_BAND_DZ_M"]),
    )
    road = classify_road(
        bundle,
        accum.points_xyz,
        corridor_geom=roi_res.roi_geom,
        ground_band_dz_m=float(cfg["GROUND_BAND_DZ_M"]),
        min_density=float(cfg["MIN_DENSITY"]),
        roughness_max_m=float(cfg["ROAD_ROUGHNESS_MAX_M"]),
        close_radius_m=float(cfg["ROAD_CLOSE_RADIUS_M"]),
    )

    marking, ground_p95, bg_mean = _compute_markings_v2(
        bundle,
        accum.points_xyz,
        accum.intensity,
        road.road_points_mask,
        road.road_mask,
        roi_res.roi_geom,
        cfg,
    )
    params_hash = update_resolved_config(cfg, run_dir)

    crosswalk = detect_crosswalks(
        marking.markings_polygons,
        corridor_geom=roi_res.roi_geom,
        merge_radius_m=float(cfg["CROSSWALK_MERGE_RADIUS_M"]),
        w_min_m=float(cfg["CROSSWALK_W_MIN_M"]),
        w_max_m=float(cfg["CROSSWALK_W_MAX_M"]),
        l_min_m=float(cfg["CROSSWALK_L_MIN_M"]),
        l_max_m=float(cfg["CROSSWALK_L_MAX_M"]),
        area_min_m2=float(cfg["CROSSWALK_AREA_MIN_M2"]),
        area_max_m2=float(cfg["CROSSWALK_AREA_MAX_M2"]),
        min_components=int(cfg["CROSSWALK_MIN_COMPONENTS"]),
    )

    # Rasters
    height_path = ras_dir / "height_p10_utm32.tif"
    inten_p95_path = ras_dir / "intensity_ground_p95_utm32.tif"
    bg_mean_path = ras_dir / "intensity_bg_mean_utm32.tif"
    mark_score_path = ras_dir / "marking_score_utm32.tif"
    mark_mask_path = ras_dir / "marking_mask_utm32.tif"
    road_mask_path = ras_dir / "road_mask_utm32.tif"

    _write_raster(height_path, bundle.height_p10, bundle, target_epsg, warnings)
    _write_raster(inten_p95_path, ground_p95, bundle, target_epsg, warnings)
    _write_raster(bg_mean_path, bg_mean, bundle, target_epsg, warnings)
    _write_raster(mark_score_path, marking.marking_score, bundle, target_epsg, warnings)
    _mask_raster(mark_mask_path, marking.marking_mask, bundle, target_epsg, warnings)
    _mask_raster(road_mask_path, road.road_mask, bundle, target_epsg, warnings)

    _write_meta(
        mark_score_path.with_suffix(".meta.json"),
        {"params_hash": params_hash, "type": "marking_score", "score_thresh": marking.score_thresh},
    )
    _write_meta(
        mark_mask_path.with_suffix(".meta.json"),
        {"params_hash": params_hash, "type": "marking_mask", "score_thresh": marking.score_thresh},
    )

    # Vectors
    road_vec = _polys_to_world_candidates(road.road_polygons, drive_id, "road_surface", target_epsg, bundle.res_m, params_hash)
    mark_vec = _polys_to_world_candidates(marking.markings_polygons, drive_id, "lane_marking", target_epsg, bundle.res_m, params_hash)
    cross_vec = _polys_to_world_candidates(crosswalk.crosswalks, drive_id, "crosswalk", target_epsg, bundle.res_m, params_hash)

    road_vec_path = vec_dir / "road_surface_utm32.gpkg"
    mark_vec_path = vec_dir / "markings_utm32.gpkg"
    cross_vec_path = vec_dir / "crosswalk_utm32.gpkg"
    write_gpkg_layer(road_vec_path, "world_candidates", road_vec, target_epsg, warnings, overwrite=True)
    write_gpkg_layer(mark_vec_path, "world_candidates", mark_vec, target_epsg, warnings, overwrite=True)
    write_gpkg_layer(cross_vec_path, "world_candidates", cross_vec, target_epsg, warnings, overwrite=True)

    # Pointclouds
    road_mask = road.road_points_mask
    non_road_mask = ~road_mask
    mark_mask = marking.markings_points_mask
    cross_mask = _points_in_polys(accum.points_xyz, crosswalk.crosswalks) & mark_mask

    road_path = pc_dir / "road_surface_points_utm32.laz"
    non_path = pc_dir / "non_road_points_utm32.laz"
    mark_path = pc_dir / "markings_points_utm32.laz"
    cross_path = pc_dir / "crosswalk_points_utm32.laz"

    if np.any(road_mask):
        _write_pointcloud(road_path, accum.points_xyz[road_mask], accum.intensity[road_mask], np.full(int(road_mask.sum()), 11, dtype=np.uint8), target_epsg)
    if np.any(non_road_mask):
        _write_pointcloud(non_path, accum.points_xyz[non_road_mask], accum.intensity[non_road_mask], np.full(int(non_road_mask.sum()), 1, dtype=np.uint8), target_epsg)
    if np.any(mark_mask):
        _write_pointcloud(mark_path, accum.points_xyz[mark_mask], accum.intensity[mark_mask], np.full(int(mark_mask.sum()), 1, dtype=np.uint8), target_epsg)
    if np.any(cross_mask):
        _write_pointcloud(cross_path, accum.points_xyz[cross_mask], accum.intensity[cross_mask], np.full(int(cross_mask.sum()), 1, dtype=np.uint8), target_epsg)

    _write_meta(
        mark_path.with_suffix(".meta.json"),
        {"params_hash": params_hash, "type": "markings_points"},
    )
    _write_meta(
        cross_vec_path.with_suffix(".meta.json"),
        {"params_hash": params_hash, "type": "crosswalk"},
    )

    road_cover = float(road.stats.get("road_cover", 0.0))
    marking_cover = float(marking.stats.get("marking_cover_on_road", 0.0))
    marking_ratio = float(marking.stats.get("marking_points_ratio", 0.0))
    cross_cnt = int(crosswalk.stats.get("crosswalk_count", 0.0))

    qa_row = {
        "drive_id": drive_id,
        "status": "ok",
        "params_hash": params_hash,
        "road_cover": round(road_cover, 6),
        "marking_cover_on_road": round(marking_cover, 6),
        "marking_points_ratio": round(marking_ratio, 6),
        "crosswalk_count": cross_cnt,
        "roi_path": relpath(run_dir, roi_path),
        "road_surface_path": relpath(run_dir, road_vec_path),
        "markings_path": relpath(run_dir, mark_vec_path),
        "crosswalk_path": relpath(run_dir, cross_vec_path),
        "markings_points_path": relpath(run_dir, mark_path),
        "marking_score_path": relpath(run_dir, mark_score_path),
        "marking_mask_path": relpath(run_dir, mark_mask_path),
        "degrade_notes": ";".join(drive_notes),
    }
    write_csv(qa_dir / "qa_index.csv", [qa_row], list(qa_row.keys()))

    report_lines = [
        "# LiDAR Semantic V2 0010 Report",
        "",
        f"- drive_id: {drive_id}",
        f"- params_hash: {params_hash}",
        f"- resolved_config: {relpath(run_dir, run_dir / 'resolved_config.yaml')}",
        "",
        "## Metrics",
        f"- road_cover: {road_cover:.6f}",
        f"- marking_cover_on_road: {marking_cover:.6f}",
        f"- marking_points_ratio: {marking_ratio:.6f}",
        f"- crosswalk_count: {cross_cnt}",
        "",
        "## Degrade Notes",
    ]
    report_lines.extend([f"- {n}" for n in drive_notes] if drive_notes else ["- none"])
    report_lines.extend(
        [
            "",
            "## Outputs",
            f"- {relpath(run_dir, roi_path)}",
            f"- {relpath(run_dir, height_path)}",
            f"- {relpath(run_dir, inten_p95_path)}",
            f"- {relpath(run_dir, bg_mean_path)}",
            f"- {relpath(run_dir, mark_score_path)}",
            f"- {relpath(run_dir, mark_mask_path)}",
            f"- {relpath(run_dir, road_vec_path)}",
            f"- {relpath(run_dir, mark_vec_path)}",
            f"- {relpath(run_dir, cross_vec_path)}",
            f"- {relpath(run_dir, mark_path)}",
            "",
            "## Warnings",
        ]
    )
    report_lines.extend([f"- {w}" for w in warnings] if warnings else ["- none"])
    write_text(qa_dir / "report.md", "\n".join(report_lines))

    summary = {
        "run_id": run_id,
        "drive_id": drive_id,
        "params_hash": params_hash,
        "resolved_config": str(run_dir / "resolved_config.yaml"),
        "road_cover": round(road_cover, 6),
        "marking_cover_on_road": round(marking_cover, 6),
        "marking_points_ratio": round(marking_ratio, 6),
        "crosswalk_count": cross_cnt,
        "degrade_notes": drive_notes,
        "warnings": warnings,
        "errors": errors,
    }
    write_json(run_dir / "run_summary.json", summary)
    summary_md = [
        "# LiDAR Semantic V2 0010 Summary",
        "",
        f"- run_id: {run_id}",
        f"- drive_id: {drive_id}",
        f"- params_hash: {params_hash}",
        f"- road_cover: {road_cover:.6f}",
        f"- marking_cover_on_road: {marking_cover:.6f}",
        f"- marking_points_ratio: {marking_ratio:.6f}",
        f"- crosswalk_count: {cross_cnt}",
    ]
    write_text(run_dir / "run_summary.md", "\n".join(summary_md))

    qgis_doc = docs_dir / "qgis_lidar_semantic_v2_0010.md"
    qgis_doc.write_text(
        "\n".join(
            [
                "# QGIS quick check (LiDAR semantic v2 0010)",
                "",
                "1) Load point layers:",
                f"- {relpath(run_dir, road_path)}",
                f"- {relpath(run_dir, non_path)}",
                f"- {relpath(run_dir, mark_path)}",
                f"- {relpath(run_dir, cross_path)}",
                "2) Load raster layers:",
                f"- {relpath(run_dir, mark_score_path)}",
                f"- {relpath(run_dir, mark_mask_path)}",
                "3) Load vector layers:",
                f"- {relpath(run_dir, road_vec_path)}",
                f"- {relpath(run_dir, mark_vec_path)}",
                f"- {relpath(run_dir, cross_vec_path)}",
            ]
        ),
        encoding="utf-8",
    )

    # Postcheck
    ok, reason = postcheck(run_dir, qa_dir / "report.md", mark_score_path.with_suffix(".meta.json"))
    if not ok:
        write_text(run_dir / "postcheck_fail.txt", reason)
        return 3

    elapsed = time.perf_counter() - start_t
    write_text(qa_dir / "elapsed.txt", f"{elapsed:.2f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

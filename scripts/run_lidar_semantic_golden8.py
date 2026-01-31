from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio import features

from pipeline.lidar_semantic.accum_points_world import AccumResult, accumulate_world_points
from pipeline.lidar_semantic.build_rasters import RasterBundle, build_rasters
from pipeline.lidar_semantic.build_roi_corridor import RoiResult, build_roi_corridor, roi_to_gdf
from pipeline.lidar_semantic.classify_road import RoadResult, classify_road
from pipeline.lidar_semantic.detect_crosswalk import CrosswalkResult, detect_crosswalks
from pipeline.lidar_semantic.export_pointcloud import classify_codes, write_las
from pipeline.lidar_semantic.extract_markings import MarkingResult, extract_markings
from pipeline.qa.lidar_semantic_report import write_drive_report, write_qa_index, write_run_summary
from scripts.pipeline_common import (
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
class DegradePlan:
    voxel_size_m: float
    raster_res_m: float
    max_frames: int
    notes: List[str]


def _detect_data_root(cfg: Dict) -> Path:
    candidates: List[str] = []
    env_root = os.environ.get("POC_DATA_ROOT", "").strip()
    if env_root:
        candidates.append(env_root)
    cfg_root = str(cfg.get("DATA_ROOT") or "").strip()
    if cfg_root:
        candidates.append(cfg_root)
    candidates.extend([str(p) for p in (cfg.get("DATA_ROOT_CANDIDATES") or [])])
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    # Fall back to a common default without interrupting the pipeline.
    return Path(r"E:\KITTI360\KITTI-360")


def _load_drives(cfg: Dict, data_root: Path) -> List[str]:
    drives_file = Path(str(cfg.get("GOLDEN_DRIVES_FILE") or "configs/golden_drives.txt"))
    if drives_file.exists():
        drives = [ln.strip() for ln in drives_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if drives:
            return drives
    raw_root = data_root / "data_3d_raw"
    if raw_root.exists():
        drives = sorted([p.name for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("2013_05_28_drive_")])
        if drives:
            return drives
    return []


def _pick_tune_drive(drives: List[str], hint: str) -> Optional[str]:
    hint = str(hint or "").strip()
    if hint:
        for d in drives:
            if hint in d:
                return d
    return drives[0] if drives else None


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


def _polys_to_world_candidates(polys: gpd.GeoDataFrame, drive_id: str, cls: str, epsg: int, res_m: float) -> gpd.GeoDataFrame:
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
                "attr_json": json.dumps({"area_m2": round(float(geom.area), 2)}, ensure_ascii=True),
                "geometry": geom,
            }
        )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{epsg}")
    ensure_required_columns(gdf, WORLD_FIELDS)
    return gdf


def _degrade_plan(cfg: Dict, elapsed_s: float, remaining: int, budget_s: float) -> DegradePlan:
    base_voxel = float(cfg.get("VOXEL_SIZE_M", 0.05))
    base_res = float(cfg.get("RASTER_RES_M", 0.2))
    base_max_frames = int(cfg.get("MAX_FRAMES", 0))
    notes: List[str] = []
    if remaining <= 0:
        return DegradePlan(base_voxel, base_res, base_max_frames, notes)
    remain_s = max(0.0, budget_s - elapsed_s)
    per_drive = remain_s / float(remaining)
    voxel = base_voxel
    res = base_res
    max_frames = base_max_frames
    if per_drive < 900:
        voxel = max(voxel, 0.08)
        notes.append("degrade:voxel=0.08")
    if per_drive < 600:
        voxel = max(voxel, 0.10)
        res = max(res, 0.30)
        notes.append("degrade:raster=0.30")
    if per_drive < 300:
        res = max(res, 0.50)
        max_frames = 2000 if max_frames == 0 else min(max_frames, 2000)
        notes.append("degrade:max_frames=2000")
    elif per_drive < 450:
        max_frames = 3000 if max_frames == 0 else min(max_frames, 3000)
        notes.append("degrade:max_frames=3000")
    return DegradePlan(voxel, res, max_frames, notes)


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


def _score(stats: Dict[str, float]) -> float:
    return (
        3.0 * _penalty_range(stats["road_cover"], (0.55, 0.90), (0.35, 0.97))
        + 3.0 * _penalty_range(stats["marking_cover_on_road"], (0.005, 0.06), (0.0, 0.12))
        + 2.0 * _penalty_range(stats["marking_points_ratio"], (0.002, 0.05), (0.0, 0.10))
        + 1.0 * _penalty_upper(stats["crosswalk_count"], 10.0, 25.0)
    )


def _run_pipeline(
    points_xyz: np.ndarray,
    intensity: np.ndarray,
    roi_geom: object,
    corridor_geom: object,
    params: Dict[str, float],
    res_m: float,
) -> Tuple[RasterBundle, RoadResult, MarkingResult, CrosswalkResult]:
    bundle = build_rasters(
        points_xyz,
        intensity,
        roi_geom=roi_geom,
        res_m=res_m,
        ground_band_dz_m=float(params["GROUND_BAND_DZ_M"]),
    )
    road = classify_road(
        bundle,
        points_xyz,
        corridor_geom=corridor_geom,
        ground_band_dz_m=float(params["GROUND_BAND_DZ_M"]),
        min_density=float(params["MIN_DENSITY"]),
        roughness_max_m=float(params["ROAD_ROUGHNESS_MAX_M"]),
        close_radius_m=float(params["ROAD_CLOSE_RADIUS_M"]),
    )
    marking = extract_markings(
        bundle,
        points_xyz,
        intensity,
        road_points_mask=road.road_points_mask,
        road_mask=road.road_mask,
        corridor_geom=corridor_geom,
        intensity_pctl=float(params["MARKING_INTENSITY_PCTL"]),
        height_max_m=float(params["MARKING_HEIGHT_MAX_M"]),
        close_radius_m=float(params["MARKING_CLOSE_RADIUS_M"]),
    )
    crosswalk = detect_crosswalks(
        marking.markings_polygons,
        corridor_geom=corridor_geom,
        merge_radius_m=float(params["CROSSWALK_MERGE_RADIUS_M"]),
        w_min_m=float(params["CROSSWALK_W_MIN_M"]),
        w_max_m=float(params["CROSSWALK_W_MAX_M"]),
        l_min_m=float(params["CROSSWALK_L_MIN_M"]),
        l_max_m=float(params["CROSSWALK_L_MAX_M"]),
        area_min_m2=float(params["CROSSWALK_AREA_MIN_M2"]),
        area_max_m2=float(params["CROSSWALK_AREA_MAX_M2"]),
        min_components=int(params["CROSSWALK_MIN_COMPONENTS"]),
    )
    return bundle, road, marking, crosswalk


def _tune_params(
    run_dir: Path,
    tune_drive: str,
    cfg: Dict,
    data_root: Path,
    roi_geom: object,
    warnings: List[str],
) -> Dict[str, float]:
    autotune_dir = ensure_dir(run_dir / "autotune")
    cache_dir = ensure_dir(autotune_dir / "cache")
    cache_npz = cache_dir / f"{tune_drive}_points_world.npz"

    if cache_npz.exists():
        cached = np.load(cache_npz)
        points_xyz = cached["points_xyz"]
        intensity = cached["intensity"]
    else:
        accum = accumulate_world_points(
            data_root=data_root,
            drive_id=tune_drive,
            roi_geom=roi_geom,
            mode=str(cfg.get("LIDAR_WORLD_MODE", "fullpose")),
            cam_id="image_00",
            stride=int(cfg.get("STRIDE", 1)),
            max_frames=int(cfg.get("MAX_FRAMES", 0)),
            voxel_size_m=float(cfg.get("VOXEL_SIZE_M", 0.05)),
        )
        points_xyz = accum.points_xyz
        intensity = accum.intensity
        np.savez_compressed(cache_npz, points_xyz=points_xyz, intensity=intensity)
        if accum.errors:
            warnings.extend([f"tune_accum:{e}" for e in accum.errors])

    base_params = {
        "GROUND_BAND_DZ_M": float(cfg.get("GROUND_BAND_DZ_M", 0.15)),
        "MIN_DENSITY": float(cfg.get("MIN_DENSITY", 8)),
        "ROAD_ROUGHNESS_MAX_M": float(cfg.get("ROAD_ROUGHNESS_MAX_M", 0.05)),
        "ROAD_CLOSE_RADIUS_M": float(cfg.get("ROAD_CLOSE_RADIUS_M", 0.6)),
        "MARKING_INTENSITY_PCTL": float(cfg.get("MARKING_INTENSITY_PCTL", 98.5)),
        "MARKING_HEIGHT_MAX_M": float(cfg.get("MARKING_HEIGHT_MAX_M", 0.06)),
        "MARKING_CLOSE_RADIUS_M": float(cfg.get("MARKING_CLOSE_RADIUS_M", 0.4)),
        "CROSSWALK_MERGE_RADIUS_M": float(cfg.get("CROSSWALK_MERGE_RADIUS_M", 1.0)),
        "CROSSWALK_W_MIN_M": float(cfg.get("CROSSWALK_W_MIN_M", 2.5)),
        "CROSSWALK_W_MAX_M": float(cfg.get("CROSSWALK_W_MAX_M", 8.0)),
        "CROSSWALK_L_MIN_M": float(cfg.get("CROSSWALK_L_MIN_M", 3.0)),
        "CROSSWALK_L_MAX_M": float(cfg.get("CROSSWALK_L_MAX_M", 20.0)),
        "CROSSWALK_AREA_MIN_M2": float(cfg.get("CROSSWALK_AREA_MIN_M2", 10.0)),
        "CROSSWALK_AREA_MAX_M2": float(cfg.get("CROSSWALK_AREA_MAX_M2", 200.0)),
        "CROSSWALK_MIN_COMPONENTS": int(cfg.get("CROSSWALK_MIN_COMPONENTS", 4)),
    }
    stage1 = cfg.get("TUNE_STAGE1") or {}
    stage2 = cfg.get("TUNE_STAGE2") or {}
    res_m = float(cfg.get("RASTER_RES_M", 0.2))

    trials: List[Dict[str, object]] = []
    best = None
    best_params = dict(base_params)

    trial_id = 0
    for dz in stage1.get("GROUND_BAND_DZ_M", [base_params["GROUND_BAND_DZ_M"]]):
        for rough in stage1.get("ROAD_ROUGHNESS_MAX_M", [base_params["ROAD_ROUGHNESS_MAX_M"]]):
            for close_r in stage1.get("ROAD_CLOSE_RADIUS_M", [base_params["ROAD_CLOSE_RADIUS_M"]]):
                trial_id += 1
                params = dict(base_params)
                params.update(
                    {
                        "GROUND_BAND_DZ_M": float(dz),
                        "ROAD_ROUGHNESS_MAX_M": float(rough),
                        "ROAD_CLOSE_RADIUS_M": float(close_r),
                    }
                )
                bundle, road, marking, crosswalk = _run_pipeline(points_xyz, intensity, roi_geom, roi_geom, params, res_m)
                stats = {
                    "road_cover": float(road.stats["road_cover"]),
                    "marking_cover_on_road": float(marking.stats["marking_cover_on_road"]),
                    "marking_points_ratio": float(marking.stats["marking_points_ratio"]),
                    "crosswalk_count": float(crosswalk.stats["crosswalk_count"]),
                }
                score = _score(stats)
                row = {"trial_id": trial_id, "stage": "stage1", **params, **stats, "score": score}
                trials.append(row)
                if best is None or score < float(best["score"]):
                    best = row
                    best_params = params

    # Stage 2: fix road params, scan marking params.
    road_fixed = {
        "GROUND_BAND_DZ_M": float(best_params["GROUND_BAND_DZ_M"]),
        "ROAD_ROUGHNESS_MAX_M": float(best_params["ROAD_ROUGHNESS_MAX_M"]),
        "ROAD_CLOSE_RADIUS_M": float(best_params["ROAD_CLOSE_RADIUS_M"]),
    }
    bundle, road, _, _ = _run_pipeline(points_xyz, intensity, roi_geom, roi_geom, best_params, res_m)
    for pctl in stage2.get("MARKING_INTENSITY_PCTL", [base_params["MARKING_INTENSITY_PCTL"]]):
        for hmax in stage2.get("MARKING_HEIGHT_MAX_M", [base_params["MARKING_HEIGHT_MAX_M"]]):
            trial_id += 1
            params = dict(best_params)
            params.update(road_fixed)
            params.update({"MARKING_INTENSITY_PCTL": float(pctl), "MARKING_HEIGHT_MAX_M": float(hmax)})
            # Reuse bundle/road but recompute markings + crosswalk.
            marking = extract_markings(
                bundle,
                points_xyz,
                intensity,
                road_points_mask=road.road_points_mask,
                road_mask=road.road_mask,
                corridor_geom=roi_geom,
                intensity_pctl=float(params["MARKING_INTENSITY_PCTL"]),
                height_max_m=float(params["MARKING_HEIGHT_MAX_M"]),
                close_radius_m=float(params["MARKING_CLOSE_RADIUS_M"]),
            )
            crosswalk = detect_crosswalks(
                marking.markings_polygons,
                corridor_geom=roi_geom,
                merge_radius_m=float(params["CROSSWALK_MERGE_RADIUS_M"]),
                w_min_m=float(params["CROSSWALK_W_MIN_M"]),
                w_max_m=float(params["CROSSWALK_W_MAX_M"]),
                l_min_m=float(params["CROSSWALK_L_MIN_M"]),
                l_max_m=float(params["CROSSWALK_L_MAX_M"]),
                area_min_m2=float(params["CROSSWALK_AREA_MIN_M2"]),
                area_max_m2=float(params["CROSSWALK_AREA_MAX_M2"]),
                min_components=int(params["CROSSWALK_MIN_COMPONENTS"]),
            )
            stats = {
                "road_cover": float(road.stats["road_cover"]),
                "marking_cover_on_road": float(marking.stats["marking_cover_on_road"]),
                "marking_points_ratio": float(marking.stats["marking_points_ratio"]),
                "crosswalk_count": float(crosswalk.stats["crosswalk_count"]),
            }
            score = _score(stats)
            row = {"trial_id": trial_id, "stage": "stage2", **params, **stats, "score": score}
            trials.append(row)
            if best is None or score < float(best["score"]):
                best = row
                best_params = params

    trials_df = pd.DataFrame(trials)
    trials_df.to_csv(autotune_dir / "trials.csv", index=False, encoding="utf-8")
    with (autotune_dir / "best_params.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(best_params, f, sort_keys=False, allow_unicode=False)
    return best_params


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


def _write_pointclouds(
    pointcloud_dir: Path,
    drive_id: str,
    points_xyz: np.ndarray,
    intensity: np.ndarray,
    road_points_mask: np.ndarray,
    markings_points_mask: np.ndarray,
    crosswalk_polys: gpd.GeoDataFrame,
    epsg: int,
    warnings: List[str],
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    cls = classify_codes(None, road_points_mask)
    semantic_path = pointcloud_dir / "semantic_points_utm32.laz"
    outs = write_las(semantic_path, points_xyz, intensity, cls, epsg)
    outputs["semantic_points_path"] = str(outs[0])
    if len(outs) > 1:
        outputs["semantic_points_las_path"] = str(outs[1])

    road_path = pointcloud_dir / "road_surface_points_utm32.laz"
    if np.any(road_points_mask):
        write_las(road_path, points_xyz[road_points_mask], intensity[road_points_mask], np.full(int(road_points_mask.sum()), 11, dtype=np.uint8), epsg)
        outputs["road_points_path"] = str(road_path)

    non_road_mask = ~road_points_mask
    non_path = pointcloud_dir / "non_road_points_utm32.laz"
    if np.any(non_road_mask):
        write_las(non_path, points_xyz[non_road_mask], intensity[non_road_mask], np.full(int(non_road_mask.sum()), 1, dtype=np.uint8), epsg)
        outputs["non_road_points_path"] = str(non_path)

    mark_path = pointcloud_dir / "markings_points_utm32.laz"
    if np.any(markings_points_mask):
        write_las(mark_path, points_xyz[markings_points_mask], intensity[markings_points_mask], np.full(int(markings_points_mask.sum()), 1, dtype=np.uint8), epsg)
        outputs["markings_points_path"] = str(mark_path)

    cross_mask = _points_in_polys(points_xyz, crosswalk_polys) & markings_points_mask
    cross_path = pointcloud_dir / "crosswalk_points_utm32.laz"
    if np.any(cross_mask):
        write_las(cross_path, points_xyz[cross_mask], intensity[cross_mask], np.full(int(cross_mask.sum()), 1, dtype=np.uint8), epsg)
        outputs["crosswalk_points_path"] = str(cross_path)

    if semantic_path.suffix.lower() == ".laz":
        warnings.append("pointcloud_note:laspy_missing_wrote_las_format")
    return outputs


def main() -> int:
    cfg = load_yaml(Path("configs/lidar_semantic_golden8.yaml"))
    run_id = now_ts()
    run_dir = Path("runs") / f"lidar_semantic_{run_id}"
    if bool(cfg.get("OVERWRITE", True)):
        ensure_overwrite(run_dir)
    ensure_dir(run_dir)
    setup_logging(run_dir / "run.log")

    warnings: List[str] = []
    errors: List[str] = []

    data_root = _detect_data_root(cfg)
    drives = _load_drives(cfg, data_root)
    tune_drive = _pick_tune_drive(drives, cfg.get("TUNE_DRIVE_HINT"))
    if not drives or tune_drive is None:
        summary = {
            "run_id": run_id,
            "status": "fail",
            "reason": "no_drives_found",
            "data_root": str(data_root),
        }
        write_run_summary(run_dir, summary, [])
        return 2

    target_epsg = int(cfg.get("TARGET_EPSG", 32632))
    budget_s = float(cfg.get("TIME_BUDGET_H", 6.0)) * 3600.0
    start_t = time.perf_counter()

    # Tune drive ROI + autotune.
    tune_roi = build_roi_corridor(
        data_root=data_root,
        drive_id=tune_drive,
        buffer_m=float(cfg.get("ROI_BUFFER_M", 20.0)),
        stride=int(cfg.get("TRAJ_STRIDE", 5)),
        target_epsg=target_epsg,
    )
    if tune_roi.errors:
        warnings.extend([f"tune_roi:{e}" for e in tune_roi.errors])
    best_params = _tune_params(run_dir, tune_drive, cfg, data_root, tune_roi.roi_geom, warnings)

    merged_dir = ensure_dir(run_dir / "merged")
    merged_road: List[gpd.GeoDataFrame] = []
    merged_mark: List[gpd.GeoDataFrame] = []
    merged_cross: List[gpd.GeoDataFrame] = []

    per_drive_rows: List[Dict[str, object]] = []

    for i, drive_id in enumerate(drives):
        drive_start = time.perf_counter()
        remaining = len(drives) - i
        degrade = _degrade_plan(cfg, elapsed_s=drive_start - start_t, remaining=remaining, budget_s=budget_s)

        drive_dir = run_dir / "drives" / drive_id
        roi_dir = ensure_dir(drive_dir / "roi")
        ras_dir = ensure_dir(drive_dir / "rasters")
        pc_dir = ensure_dir(drive_dir / "pointcloud")
        vec_dir = ensure_dir(drive_dir / "vectors")
        qa_dir = ensure_dir(drive_dir / "qa")

        drive_warnings: List[str] = []
        drive_warnings.extend(degrade.notes)
        try:
            roi_res: RoiResult = build_roi_corridor(
                data_root=data_root,
                drive_id=drive_id,
                buffer_m=float(cfg.get("ROI_BUFFER_M", 20.0)),
                stride=int(cfg.get("TRAJ_STRIDE", 5)),
                target_epsg=target_epsg,
            )
            drive_warnings.extend(roi_res.errors)
            roi_gdf = roi_to_gdf(roi_res.roi_geom, drive_id, target_epsg)
            roi_path = roi_dir / "roi_corridor_utm32.gpkg"
            write_gpkg_layer(roi_path, "roi", roi_gdf, target_epsg, drive_warnings, overwrite=True)

            accum: AccumResult = accumulate_world_points(
                data_root=data_root,
                drive_id=drive_id,
                roi_geom=roi_res.roi_geom,
                mode=str(cfg.get("LIDAR_WORLD_MODE", "fullpose")),
                cam_id="image_00",
                stride=int(cfg.get("STRIDE", 1)),
                max_frames=int(degrade.max_frames),
                voxel_size_m=float(degrade.voxel_size_m),
            )
            drive_warnings.extend(accum.errors)

            params = dict(best_params)
            params["MIN_DENSITY"] = float(cfg.get("MIN_DENSITY", 8))
            params.update(
                {
                    "CROSSWALK_MERGE_RADIUS_M": float(cfg.get("CROSSWALK_MERGE_RADIUS_M", 1.0)),
                    "CROSSWALK_W_MIN_M": float(cfg.get("CROSSWALK_W_MIN_M", 2.5)),
                    "CROSSWALK_W_MAX_M": float(cfg.get("CROSSWALK_W_MAX_M", 8.0)),
                    "CROSSWALK_L_MIN_M": float(cfg.get("CROSSWALK_L_MIN_M", 3.0)),
                    "CROSSWALK_L_MAX_M": float(cfg.get("CROSSWALK_L_MAX_M", 20.0)),
                    "CROSSWALK_AREA_MIN_M2": float(cfg.get("CROSSWALK_AREA_MIN_M2", 10.0)),
                    "CROSSWALK_AREA_MAX_M2": float(cfg.get("CROSSWALK_AREA_MAX_M2", 200.0)),
                    "CROSSWALK_MIN_COMPONENTS": int(cfg.get("CROSSWALK_MIN_COMPONENTS", 4)),
                    "MARKING_CLOSE_RADIUS_M": float(cfg.get("MARKING_CLOSE_RADIUS_M", 0.4)),
                }
            )

            bundle, road, marking, crosswalk = _run_pipeline(
                accum.points_xyz,
                accum.intensity,
                roi_res.roi_geom,
                roi_res.roi_geom,
                params,
                res_m=float(degrade.raster_res_m),
            )

            # Rasters
            height_path = ras_dir / "height_p10_utm32.tif"
            inten_path = ras_dir / "intensity_max_utm32.tif"
            road_mask_path = ras_dir / "road_mask_utm32.tif"
            mark_score_path = ras_dir / "marking_score_utm32.tif"
            mark_mask_path = ras_dir / "marking_mask_utm32.tif"
            _write_raster(height_path, bundle.height_p10, bundle, target_epsg, drive_warnings)
            _write_raster(inten_path, bundle.intensity_max, bundle, target_epsg, drive_warnings)
            _mask_raster(road_mask_path, road.road_mask, bundle, target_epsg, drive_warnings)
            _write_raster(mark_score_path, marking.marking_score, bundle, target_epsg, drive_warnings)
            _mask_raster(mark_mask_path, marking.marking_mask, bundle, target_epsg, drive_warnings)

            # Vectors
            road_vec = _polys_to_world_candidates(road.road_polygons, drive_id, "road_surface", target_epsg, bundle.res_m)
            mark_vec = _polys_to_world_candidates(marking.markings_polygons, drive_id, "lane_marking", target_epsg, bundle.res_m)
            cross_vec = _polys_to_world_candidates(crosswalk.crosswalks, drive_id, "crosswalk", target_epsg, bundle.res_m)
            road_vec_path = vec_dir / "road_surface_utm32.gpkg"
            mark_vec_path = vec_dir / "markings_utm32.gpkg"
            cross_vec_path = vec_dir / "crosswalk_utm32.gpkg"
            write_gpkg_layer(road_vec_path, "world_candidates", road_vec, target_epsg, drive_warnings, overwrite=True)
            write_gpkg_layer(mark_vec_path, "world_candidates", mark_vec, target_epsg, drive_warnings, overwrite=True)
            write_gpkg_layer(cross_vec_path, "world_candidates", cross_vec, target_epsg, drive_warnings, overwrite=True)

            # Pointclouds
            pc_outputs = _write_pointclouds(
                pc_dir,
                drive_id,
                accum.points_xyz,
                accum.intensity,
                road_points_mask=road.road_points_mask,
                markings_points_mask=marking.markings_points_mask,
                crosswalk_polys=crosswalk.crosswalks,
                epsg=target_epsg,
                warnings=drive_warnings,
            )

            road_cover = float(road.stats.get("road_cover", 0.0))
            marking_cover = float(marking.stats.get("marking_cover_on_road", 0.0))
            marking_ratio = float(marking.stats.get("marking_points_ratio", 0.0))
            cross_cnt = int(crosswalk.stats.get("crosswalk_count", 0.0))

            stats_row = {
                "drive_id": drive_id,
                "status": "ok",
                "roi_source": roi_res.roi_source,
                "frame_count": accum.frame_count,
                "point_count": int(accum.points_xyz.shape[0]),
                "road_cover": round(road_cover, 6),
                "marking_cover_on_road": round(marking_cover, 6),
                "marking_points_ratio": round(marking_ratio, 6),
                "crosswalk_count": cross_cnt,
                "roi_path": relpath(run_dir, roi_path),
                "road_surface_path": relpath(run_dir, road_vec_path),
                "markings_path": relpath(run_dir, mark_vec_path),
                "crosswalk_path": relpath(run_dir, cross_vec_path),
                "semantic_points_path": relpath(run_dir, Path(pc_outputs.get("semantic_points_path", ""))),
                "road_points_path": relpath(run_dir, Path(pc_outputs.get("road_points_path", ""))) if pc_outputs.get("road_points_path") else "",
                "markings_points_path": relpath(run_dir, Path(pc_outputs.get("markings_points_path", ""))) if pc_outputs.get("markings_points_path") else "",
                "degrade_notes": ";".join(degrade.notes),
            }
            per_drive_rows.append(stats_row)

            write_drive_report(qa_dir / "report.md", drive_id, stats_row, drive_warnings)
            write_csv(qa_dir / "qa_index.csv", [stats_row], list(stats_row.keys()))

            if not road_vec.empty:
                merged_road.append(road_vec)
            if not mark_vec.empty:
                merged_mark.append(mark_vec)
            if not cross_vec.empty:
                merged_cross.append(cross_vec)
        except Exception as exc:  # pragma: no cover - data dependent
            fail_row = {
                "drive_id": drive_id,
                "status": "fail",
                "reason": str(exc),
                "degrade_notes": ";".join(degrade.notes),
            }
            per_drive_rows.append(fail_row)
            write_drive_report(qa_dir / "report.md", drive_id, fail_row, drive_warnings)
            write_csv(qa_dir / "qa_index.csv", [fail_row], list(fail_row.keys()))
            errors.append(f"{drive_id}:{exc}")
        finally:
            drive_elapsed = time.perf_counter() - drive_start
            write_text(qa_dir / "elapsed.txt", f"{drive_elapsed:.2f}\n")

    merged_warnings: List[str] = []
    if merged_road:
        road_merged = gpd.GeoDataFrame(pd.concat(merged_road, ignore_index=True), crs=f"EPSG:{target_epsg}")
        road_path = merged_dir / "road_surface_utm32.gpkg"
        write_gpkg_layer(road_path, "world_candidates", road_merged, target_epsg, merged_warnings, overwrite=True)
    if merged_mark:
        mark_merged = gpd.GeoDataFrame(pd.concat(merged_mark, ignore_index=True), crs=f"EPSG:{target_epsg}")
        mark_path = merged_dir / "markings_utm32.gpkg"
        write_gpkg_layer(mark_path, "world_candidates", mark_merged, target_epsg, merged_warnings, overwrite=True)
    if merged_cross:
        cross_merged = gpd.GeoDataFrame(pd.concat(merged_cross, ignore_index=True), crs=f"EPSG:{target_epsg}")
        cross_path = merged_dir / "crosswalk_utm32.gpkg"
        write_gpkg_layer(cross_path, "world_candidates", cross_merged, target_epsg, merged_warnings, overwrite=True)

    qa_path = merged_dir / "qa_index.csv"
    write_qa_index(qa_path, per_drive_rows)

    ok_cnt = len([r for r in per_drive_rows if r.get("status") == "ok"])
    fail_cnt = len(per_drive_rows) - ok_cnt
    summary = {
        "run_id": run_id,
        "data_root": str(data_root),
        "target_epsg": target_epsg,
        "drives_total": len(per_drive_rows),
        "drives_ok": ok_cnt,
        "drives_fail": fail_cnt,
        "tune_drive": tune_drive,
        "best_params": best_params,
        "warnings": warnings + merged_warnings,
        "errors": errors,
    }
    write_run_summary(run_dir, summary, per_drive_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


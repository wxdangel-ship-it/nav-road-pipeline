from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import unary_union

from pipeline.datasets.kitti360_io import (
    _find_oxts_dir,
    _find_velodyne_dir,
    load_kitti360_calib,
    load_kitti360_lidar_points,
    load_kitti360_lidar_points_world_full,
    load_kitti360_pose,
)
from pipeline.lidar_semantic.build_rasters import build_rasters
from pipeline.lidar_semantic.export_pointcloud import write_las
from pipeline.projection.projector import project_points_cam0_to_image
from scripts.pipeline_common import (
    ensure_dir,
    ensure_overwrite,
    ensure_required_columns,
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


REQUIRED_KEYS = [
    "FRAME_START",
    "FRAME_END",
    "STRIDE",
    "ROI_BUFFER_M",
    "RASTER_RES_M",
    "VOXEL_SIZE_M",
    "TIME_BUDGET_H",
    "OVERWRITE",
    "TARGET_EPSG",
    "GROUND_BAND_DZ_M",
    "MIN_DENSITY",
    "ROAD_ROUGHNESS_MAX_M",
    "CLOSE_RADIUS_M",
    "OPEN_RADIUS_M",
    "MIN_POLY_AREA_M2",
    "SIMPLIFY_M",
    "PROJ_ENABLE",
    "PROJ_KEY_FRAMES",
    "PROJ_NONROAD_SAMPLE",
    "PROJ_GROUND_SAMPLE",
]


def _load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    import yaml

    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _normalize(cfg: Dict[str, object]) -> Dict[str, object]:
    def _norm(v):
        if isinstance(v, dict):
            return {k: _norm(v[k]) for k in sorted(v.keys())}
        if isinstance(v, list):
            return [_norm(x) for x in v]
        return v

    return _norm(cfg)


def _hash_cfg(cfg: Dict[str, object]) -> str:
    import hashlib

    raw = json.dumps(_normalize(cfg), sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _resolve_config(base: Dict[str, object], run_dir: Path) -> Dict[str, object]:
    cfg = dict(base)
    defaults = {
        "TARGET_EPSG": 32632,
        "FRAME_START": 250,
        "FRAME_END": 500,
        "STRIDE": 1,
        "ROI_BUFFER_M": 30.0,
        "RASTER_RES_M": 0.20,
        "VOXEL_SIZE_M": 0.05,
        "TIME_BUDGET_H": 6.0,
        "OVERWRITE": True,
        "GROUND_BAND_DZ_M": 0.20,
        "MIN_DENSITY": 3,
        "ROAD_ROUGHNESS_MAX_M": 0.20,
        "CLOSE_RADIUS_M": 0.8,
        "OPEN_RADIUS_M": 0.4,
        "MIN_POLY_AREA_M2": 30.0,
        "SIMPLIFY_M": 0.3,
        "PROJ_ENABLE": True,
        "PROJ_KEY_FRAMES": [250, 341, 500],
        "PROJ_NONROAD_SAMPLE": 20000,
        "PROJ_GROUND_SAMPLE": 80000,
    }
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys: {missing}")
    run_dir.mkdir(parents=True, exist_ok=True)
    import yaml

    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    params_hash = _hash_cfg(cfg)
    (run_dir / "params_hash.txt").write_text(params_hash + "\n", encoding="utf-8")
    return cfg


def _update_config(cfg: Dict[str, object], run_dir: Path) -> str:
    import yaml

    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    params_hash = _hash_cfg(cfg)
    (run_dir / "params_hash.txt").write_text(params_hash + "\n", encoding="utf-8")
    return params_hash


def _select_drive_0010(data_root: Path) -> str:
    drives_file = Path("configs/golden_drives.txt")
    if drives_file.exists():
        drives = [ln.strip() for ln in drives_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        raw_root = data_root / "data_3d_raw"
        drives = sorted([p.name for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("2013_05_28_drive_")])
    for d in drives:
        if "_0010_" in d:
            return d
    raise RuntimeError("no_0010_drive_found")


def _frame_ids(start: int, end: int, stride: int) -> List[str]:
    return [f"{i:010d}" for i in range(start, end + 1, max(1, stride))]


def _trajectory_roi(data_root: Path, drive_id: str, frame_ids: List[str], buffer_m: float) -> Tuple[Polygon, object]:
    pts: List[Tuple[float, float]] = []
    for fid in frame_ids:
        try:
            x, y, _ = load_kitti360_pose(data_root, drive_id, fid)
            pts.append((x, y))
        except Exception:
            continue
    if not pts:
        raise RuntimeError("no_pose_points")
    if len(pts) == 1:
        center = Point(pts[0])
        return center.buffer(float(buffer_m)), center
    line = LineString(pts)
    return line.buffer(float(buffer_m)), line


def _find_cam_pose_file(data_root: Path, drive_id: str) -> Optional[Path]:
    base_dirs = [
        data_root / "data_poses" / drive_id,
        data_root / "data_poses" / drive_id / "poses",
        data_root / "data_poses" / drive_id / "cam0",
        data_root / "data_poses" / drive_id / "pose",
    ]
    names = [
        "cam0_to_world.txt",
        "world_to_cam0.txt",
        "cam0_pose.txt",
        "pose_cam0.txt",
        "poses.txt",
        "pose.txt",
        "cam0.txt",
    ]
    for base in base_dirs:
        if not base.exists():
            continue
        for name in names:
            cand = base / name
            if cand.exists():
                return cand
        for cand in sorted(base.glob("*cam0*world*.txt")):
            return cand
        for cand in sorted(base.glob("*cam0*pose*.txt")):
            return cand
    return None


def _parse_pose_file(path: Path) -> Tuple[Dict[str, np.ndarray], str]:
    name = path.name.lower()
    direction = "unknown"
    if "to_world" in name:
        direction = "cam_to_world"
    if "world_to" in name:
        direction = "world_to_cam"
    out: Dict[str, np.ndarray] = {}
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        parts = [p for p in line.strip().split() if p]
        if not parts:
            continue
        frame_id = None
        nums = parts
        if len(parts) in {13, 17}:
            frame_id = parts[0]
            nums = parts[1:]
        elif len(parts) in {12, 16}:
            frame_id = f"{idx:010d}"
        else:
            continue
        try:
            vals = np.array([float(v) for v in nums], dtype=float)
        except ValueError:
            continue
        if vals.size == 12:
            mat = vals.reshape(3, 4)
            bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float)
            mat = np.vstack([mat, bottom])
        elif vals.size == 16:
            mat = vals.reshape(4, 4)
        else:
            continue
        key = str(frame_id)
        out[key] = mat
        if key.isdigit():
            out[f"{int(key):010d}"] = mat
    return out, direction


def _overlay_points(
    img_path: Path,
    u_ground: np.ndarray,
    v_ground: np.ndarray,
    in_ground: np.ndarray,
    u_non: np.ndarray,
    v_non: np.ndarray,
    in_non: np.ndarray,
    out_path: Path,
    title: str,
    params_hash: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = plt.imread(img_path)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.imshow(img)
    if np.any(in_non):
        ax.scatter(u_non[in_non], v_non[in_non], s=0.6, c="green", alpha=0.3)
    if np.any(in_ground):
        ax.scatter(u_ground[in_ground], v_ground[in_ground], s=1.0, c="red", alpha=0.7)
    ax.set_title(title)
    ax.text(0.01, 0.99, f"params_hash={params_hash}", transform=ax.transAxes, fontsize=6, color="white", va="top")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _find_image_dir(data_root: Path, drive_id: str, cam: str) -> Optional[Path]:
    candidates = [
        data_root / "data_2d_raw" / drive_id / cam / "data",
        data_root / "data_2d_raw" / drive_id / cam / "data_rect",
        data_root / drive_id / cam / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_image_path(img_dir: Path, frame_id: str) -> Optional[Path]:
    for ext in [".png", ".jpg", ".jpeg"]:
        p = img_dir / f"{frame_id}{ext}"
        if p.exists():
            return p
    return None


def _transform_points(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([points[:, :3], np.ones((points.shape[0], 1), dtype=points.dtype)])
    out = (mat @ pts_h.T).T
    return out[:, :3]


def _voxel_downsample(points: np.ndarray, intensity: np.ndarray, voxel: float) -> Tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return points, intensity
    key = np.floor(points / float(voxel)).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return points[idx], intensity[idx]


def _load_points_range(
    data_root: Path,
    drive_id: str,
    frame_ids: List[str],
    roi_geom: Polygon,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    pts_list: List[np.ndarray] = []
    inten_list: List[np.ndarray] = []
    errors: List[str] = []
    roi_bounds = roi_geom.bounds
    for fid in frame_ids:
        try:
            raw = load_kitti360_lidar_points(data_root, drive_id, fid)
            world = load_kitti360_lidar_points_world_full(data_root, drive_id, fid, cam_id="image_00")
            if raw.size == 0 or world.size == 0:
                continue
            mask = (
                (world[:, 0] >= roi_bounds[0])
                & (world[:, 0] <= roi_bounds[2])
                & (world[:, 1] >= roi_bounds[1])
                & (world[:, 1] <= roi_bounds[3])
            )
            world = world[mask]
            inten = raw[:, 3][mask]
            if world.size == 0:
                continue
            pts_list.append(world.astype(np.float32))
            inten_list.append(inten.astype(np.float32))
        except Exception as exc:
            errors.append(f"{fid}:{exc}")
    if not pts_list:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32), errors
    points = np.vstack(pts_list)
    intensity = np.concatenate(inten_list)
    points, intensity = _voxel_downsample(points, intensity, voxel_size)
    # precise clip to roi polygon
    try:
        from shapely.prepared import prep

        roi_p = prep(roi_geom)
        keep = np.array([roi_p.contains(Point(x, y)) for x, y in points[:, :2]], dtype=bool)
        if np.any(keep):
            points = points[keep]
            intensity = intensity[keep]
    except Exception:
        pass
    return points, intensity, errors


def _morph(mask: np.ndarray, op: str, radius_m: float, res_m: float) -> np.ndarray:
    r = int(round(radius_m / res_m))
    if r <= 0:
        return mask
    try:
        from scipy import ndimage as ndi

        struct = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
        if op == "close":
            return ndi.binary_closing(mask, structure=struct)
        if op == "open":
            return ndi.binary_opening(mask, structure=struct)
    except Exception:
        return mask
    return mask


def _largest_component(mask: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage as ndi

        labeled, num = ndi.label(mask)
        if num == 0:
            return mask
        best = None
        best_area = 0
        for i in range(1, num + 1):
            comp = labeled == i
            if not np.any(comp & keep_mask):
                continue
            area = int(np.sum(comp))
            if area > best_area:
                best_area = area
                best = comp
        return best if best is not None else mask
    except Exception:
        return mask


def _mask_to_polygons(mask: np.ndarray, transform: rasterio.Affine, min_area_m2: float) -> gpd.GeoDataFrame:
    geoms = []
    for geom, val in features.shapes(mask.astype("uint8"), mask=mask.astype(bool), transform=transform):
        if int(val) != 1:
            continue
        poly = shape(geom)
        if poly.is_empty:
            continue
        if float(poly.area) < min_area_m2:
            continue
        geoms.append(poly)
    return (
        gpd.GeoDataFrame([{"geometry": g} for g in geoms], geometry="geometry", crs="EPSG:32632")
        if geoms
        else gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32632")
    )


def _holes_area_ratio(polys: gpd.GeoDataFrame) -> float:
    if polys.empty:
        return 0.0
    holes = 0.0
    area = 0.0
    for geom in polys.geometry:
        if geom is None or geom.is_empty:
            continue
        area += float(geom.area)
        for ring in geom.interiors:
            holes += float(Polygon(ring).area)
    return holes / max(area, 1e-6)


def _quicklook(height: np.ndarray, mask: np.ndarray, transform: rasterio.Affine, out_path: Path) -> None:
    minx, miny, maxx, maxy = rasterio.transform.array_bounds(height.shape[0], height.shape[1], transform)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    vmin = float(np.nanpercentile(height, 2)) if np.isfinite(height).any() else 0.0
    vmax = float(np.nanpercentile(height, 98)) if np.isfinite(height).any() else 1.0
    ax.imshow(height, extent=(minx, maxx, miny, maxy), origin="upper", cmap="gray", vmin=vmin, vmax=vmax)
    mask_edges = np.logical_xor(mask, np.pad(mask, ((1, 0), (1, 0)), constant_values=False)[: mask.shape[0], : mask.shape[1]])
    ax.imshow(np.ma.masked_where(~mask_edges, mask_edges), extent=(minx, maxx, miny, maxy), origin="upper", cmap="autumn", alpha=0.6)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_meta(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> int:
    base_cfg = _load_yaml(Path("configs/lidar_road_surface_0010.yaml"))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"lidar_road_surface_0010_{run_id}"
    if bool(base_cfg.get("OVERWRITE", True)):
        ensure_overwrite(run_dir)
    ensure_dir(run_dir)
    setup_logging(run_dir / "run.log")

    cfg = _resolve_config(base_cfg, run_dir)
    params_hash = _hash_cfg(cfg)

    data_root = Path(str(base_cfg.get("KITTI_ROOT") or os.environ.get("POC_DATA_ROOT", r"E:\KITTI360\KITTI-360")))
    drive_id = _select_drive_0010(data_root)

    frames = _frame_ids(int(cfg["FRAME_START"]), int(cfg["FRAME_END"]), int(cfg["STRIDE"]))
    roi_geom, traj_geom = _trajectory_roi(data_root, drive_id, frames, float(cfg["ROI_BUFFER_M"]))

    roi_dir = ensure_dir(run_dir / "drives" / drive_id / "roi")
    ras_dir = ensure_dir(run_dir / "drives" / drive_id / "rasters")
    pc_dir = ensure_dir(run_dir / "drives" / drive_id / "pointcloud")
    vec_dir = ensure_dir(run_dir / "drives" / drive_id / "vectors")
    qa_dir = ensure_dir(run_dir / "drives" / drive_id / "qa")

    roi_gdf = gpd.GeoDataFrame([{"drive_id": drive_id, "geometry": roi_geom}], geometry="geometry", crs="EPSG:32632")
    roi_path = roi_dir / "roi_corridor_utm32.gpkg"
    write_gpkg_layer(roi_path, "roi", roi_gdf, 32632, [], overwrite=True)

    # Load points with degrade strategy
    notes: List[str] = []
    errors: List[str] = []
    stride = int(cfg["STRIDE"])
    voxel = float(cfg["VOXEL_SIZE_M"])
    res_m = float(cfg["RASTER_RES_M"])
    nonroad_sample = 1
    start_t = time.perf_counter()
    budget_s = float(cfg["TIME_BUDGET_H"]) * 3600.0

    for step in range(6):
        if step == 1 and stride < 2:
            stride = 2
            cfg["STRIDE"] = 2
            notes.append("degrade:stride=2")
        elif step == 2 and stride < 3:
            stride = 3
            cfg["STRIDE"] = 3
            notes.append("degrade:stride=3")
        elif step == 3 and voxel < 0.08:
            voxel = 0.08
            cfg["VOXEL_SIZE_M"] = 0.08
            notes.append("degrade:voxel=0.08")
        elif step == 4 and res_m < 0.30:
            res_m = 0.30
            cfg["RASTER_RES_M"] = 0.30
            notes.append("degrade:raster=0.30")
        elif step == 5 and nonroad_sample < 3:
            nonroad_sample = 3
            notes.append("degrade:nonroad_sample=3")
        if notes:
            params_hash = _update_config(cfg, run_dir)
        frames = _frame_ids(int(cfg["FRAME_START"]), int(cfg["FRAME_END"]), stride)
        points, intensity, errs = _load_points_range(data_root, drive_id, frames, roi_geom, voxel)
        errors.extend(errs)
        if points.size > 0:
            break
        if (time.perf_counter() - start_t) > budget_s:
            break

    if points.size == 0:
        write_text(qa_dir / "report.md", "no points loaded")
        return 2

    bundle = build_rasters(
        points,
        intensity,
        roi_geom=roi_geom,
        res_m=res_m,
        ground_band_dz_m=float(cfg["GROUND_BAND_DZ_M"]),
    )

    roi_mask = features.rasterize([(roi_geom, 1)], out_shape=(bundle.height, bundle.width), transform=bundle.transform, fill=0, dtype="uint8").astype(bool)
    density = bundle.density_all
    rough = bundle.roughness
    traj_buffer = traj_geom.buffer(6.0)
    traj_mask = features.rasterize([(traj_buffer, 1)], out_shape=roi_mask.shape, transform=bundle.transform, fill=0, dtype="uint8").astype(bool)

    def _compute(min_density: float, roughness_max: float, close_radius: float) -> Dict[str, object]:
        mask = (density >= float(min_density)) & (rough <= float(roughness_max)) & roi_mask
        mask = _morph(mask, "close", float(close_radius), res_m)
        mask = _morph(mask, "open", float(cfg["OPEN_RADIUS_M"]), res_m)
        mask = _largest_component(mask, traj_mask)
        road_area = float(np.sum(mask) * res_m * res_m)
        roi_area = float(roi_geom.area)
        traj_area = float(traj_buffer.area)
        traj_cover = float(np.sum(mask & traj_mask) * res_m * res_m) / max(traj_area, 1e-6)
        road_cover = road_area / max(roi_area, 1e-6)
        polys = _mask_to_polygons(mask, bundle.transform, float(cfg["MIN_POLY_AREA_M2"]))
        if not polys.empty and float(cfg["SIMPLIFY_M"]) > 0:
            polys["geometry"] = polys["geometry"].simplify(float(cfg["SIMPLIFY_M"]))
        holes_ratio = _holes_area_ratio(polys)
        return {
            "mask": mask,
            "polys": polys,
            "road_cover": road_cover,
            "traj_cover": traj_cover,
            "components_count": int(polys.shape[0]),
            "holes_ratio": holes_ratio,
            "min_density": min_density,
            "roughness_max": roughness_max,
            "close_radius": close_radius,
        }

    base_min = float(cfg["MIN_DENSITY"])
    base_rough = float(cfg["ROAD_ROUGHNESS_MAX_M"])
    base_close = float(cfg["CLOSE_RADIUS_M"])
    trials = []
    for close_r in [0.6, 0.8, 1.0]:
        trials.append(_compute(base_min, base_rough, close_r))
        trials.append(_compute(base_min * 0.8, base_rough * 0.8, close_r))
        trials.append(_compute(base_min * 1.2, base_rough * 1.2, close_r))

    def _score(t: Dict[str, object]) -> float:
        road_cover = float(t["road_cover"])
        traj_cover = float(t["traj_cover"])
        comp = float(t["components_count"])
        holes = float(t["holes_ratio"])
        score = 0.0
        if 0.45 <= road_cover <= 0.95:
            score += 1.0
        if traj_cover >= 0.85:
            score += 2.0
        if comp <= 80:
            score += 1.0
        if holes <= 0.15:
            score += 1.0
        return score + traj_cover

    best = max(trials, key=_score)
    road_mask = best["mask"]
    polys = best["polys"]
    road_cover = float(best["road_cover"])
    traj_cover = float(best["traj_cover"])
    holes_ratio = float(best["holes_ratio"])
    cfg["MIN_DENSITY"] = float(best["min_density"])
    cfg["ROAD_ROUGHNESS_MAX_M"] = float(best["roughness_max"])
    cfg["CLOSE_RADIUS_M"] = float(best["close_radius"])
    params_hash = _update_config(cfg, run_dir)
    notes.append("autotune:best_params_applied")

    # Points classification
    valid = bundle.point_valid
    ix = bundle.point_ix
    iy = bundle.point_iy
    road_cell = np.zeros_like(valid, dtype=bool)
    if np.any(valid):
        road_cell[valid] = road_mask[iy[valid], ix[valid]]
    ground_band = valid & np.isfinite(bundle.point_height_p10) & (points[:, 2] <= (bundle.point_height_p10 + float(cfg["GROUND_BAND_DZ_M"])))
    road_points = road_cell & ground_band
    non_road_points = ~road_points
    if nonroad_sample > 1:
        idx = np.arange(non_road_points.sum())
        keep = idx % nonroad_sample == 0
        non_road_points_idx = np.where(non_road_points)[0]
        non_road_mask = np.zeros_like(non_road_points, dtype=bool)
        non_road_mask[non_road_points_idx[keep]] = True
        non_road_points = non_road_mask

    # Write rasters
    def _write_raster(path: Path, arr: np.ndarray, nodata: float, dtype: str):
        validate_output_crs(path, 32632, None, [])
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=bundle.height,
            width=bundle.width,
            count=1,
            dtype=dtype,
            crs="EPSG:32632",
            transform=bundle.transform,
            nodata=nodata,
        ) as dst:
            dst.write(arr, 1)

    ground_path = ras_dir / "ground_z_p10_utm32.tif"
    rough_path = ras_dir / "roughness_utm32.tif"
    dens_path = ras_dir / "density_utm32.tif"
    mask_path = ras_dir / "road_surface_mask_utm32.tif"
    _write_raster(ground_path, bundle.height_p10.astype(np.float32), np.nan, "float32")
    _write_raster(rough_path, bundle.roughness.astype(np.float32), np.nan, "float32")
    _write_raster(dens_path, bundle.density_all.astype(np.float32), 0, "float32")
    _write_raster(mask_path, road_mask.astype(np.uint8), 255, "uint8")
    _write_meta(mask_path.with_suffix(".meta.json"), {"params_hash": params_hash})

    # Pointclouds
    road_path = pc_dir / "road_surface_points_utm32.laz"
    non_path = pc_dir / "non_road_points_utm32.laz"
    sem_path = pc_dir / "semantic_points_utm32.laz"
    if np.any(road_points):
        write_las(road_path, points[road_points], intensity[road_points], np.full(int(road_points.sum()), 11, dtype=np.uint8), 32632)
        _write_meta(road_path.with_suffix(".meta.json"), {"params_hash": params_hash})
    if np.any(non_road_points):
        write_las(non_path, points[non_road_points], intensity[non_road_points], np.full(int(non_road_points.sum()), 1, dtype=np.uint8), 32632)
        _write_meta(non_path.with_suffix(".meta.json"), {"params_hash": params_hash})
    classes = np.ones((points.shape[0],), dtype=np.uint8)
    classes[road_points] = 11
    write_las(sem_path, points, intensity, classes, 32632)
    _write_meta(sem_path.with_suffix(".meta.json"), {"params_hash": params_hash})

    # Vectors
    polys_out = gpd.GeoDataFrame(
        [
            {
                "cand_id": str(uuid4()),
                "source": "lidar",
                "drive_id": drive_id,
                "class": "road_surface",
                "crs_epsg": 32632,
                "conf": 0.7,
                "uncert": res_m,
                "evid_ref": "[]",
                "conflict": "",
                "attr_json": json.dumps({"params_hash": params_hash, "area_m2": round(float(g.area), 2)}),
                "geometry": g,
            }
            for g in polys.geometry
        ],
        geometry="geometry",
        crs="EPSG:32632",
    )
    ensure_required_columns(polys_out, WORLD_FIELDS)
    road_vec_path = vec_dir / "road_surface_utm32.gpkg"
    write_gpkg_layer(road_vec_path, "world_candidates", polys_out, 32632, [], overwrite=True)
    _write_meta(road_vec_path.with_suffix(".meta.json"), {"params_hash": params_hash})

    quicklook_path = qa_dir / "quicklook_overlay.png"
    _quicklook(bundle.height_p10, road_mask, bundle.transform, quicklook_path)

    # Fused point cloud projection to key frames (P0_E0_Y1_R)
    proj_rows = []
    if bool(cfg.get("PROJ_ENABLE", True)):
        try:
            calib = load_kitti360_calib(data_root, "image_00")
            img_dir = _find_image_dir(data_root, drive_id, "image_00")
            cam_pose_file = _find_cam_pose_file(data_root, drive_id)
            cam_pose_map, cam_pose_dir = ({}, "missing")
            if cam_pose_file is not None:
                cam_pose_map, cam_pose_dir = _parse_pose_file(cam_pose_file)
            if img_dir is not None and cam_pose_map:
                key_frames = [f"{int(f):010d}" for f in cfg.get("PROJ_KEY_FRAMES", [])]
                ground_pts = points[road_points]
                non_pts = points[non_road_points]
                if ground_pts.shape[0] > int(cfg["PROJ_GROUND_SAMPLE"]):
                    rng = np.random.default_rng(0)
                    sel = rng.choice(ground_pts.shape[0], size=int(cfg["PROJ_GROUND_SAMPLE"]), replace=False)
                    ground_pts = ground_pts[sel]
                if non_pts.shape[0] > int(cfg["PROJ_NONROAD_SAMPLE"]):
                    rng = np.random.default_rng(1)
                    sel = rng.choice(non_pts.shape[0], size=int(cfg["PROJ_NONROAD_SAMPLE"]), replace=False)
                    non_pts = non_pts[sel]
                for fid in key_frames:
                    img_path = _find_image_path(img_dir, fid)
                    if img_path is None or fid not in cam_pose_map:
                        proj_rows.append({"frame_id": fid, "status": "missing_frame"})
                        continue
                    try:
                        import matplotlib.pyplot as plt

                        img = plt.imread(img_path)
                    except Exception:
                        proj_rows.append({"frame_id": fid, "status": "image_read_fail"})
                        continue
                    h, w = img.shape[0], img.shape[1]
                    t_w_cam = cam_pose_map[fid]
                    if cam_pose_dir == "world_to_cam":
                        t_w_cam = np.linalg.inv(t_w_cam)
                    t_c_w = np.linalg.inv(t_w_cam)
                    g_cam = _transform_points(ground_pts, t_c_w)
                    ng_cam = _transform_points(non_pts, t_c_w)
                    try:
                        u_g, v_g, _, in_g = project_points_cam0_to_image(g_cam, calib, (h, w), use_rect=True, y_flip=True)
                        u_n, v_n, _, in_n = project_points_cam0_to_image(ng_cam, calib, (h, w), use_rect=True, y_flip=True)
                    except Exception:
                        proj_rows.append({"frame_id": fid, "status": "projection_fail"})
                        continue
                    in_ratio_g = float(np.sum(in_g)) / max(1, g_cam.shape[0])
                    v_norm = v_g[in_g] / max(1, float(h))
                    bottom_ratio = float(np.sum(v_norm > 0.60)) / max(1, v_norm.size)
                    title = f"frame {fid} | fused P0_E0_Y1_R | in_g={in_ratio_g:.2f} | bottom={bottom_ratio:.2f}"
                    _overlay_points(
                        img_path,
                        u_g,
                        v_g,
                        in_g,
                        u_n,
                        v_n,
                        in_n,
                        qa_dir / f"fused_overlay_frame_{fid}_all_ground.png",
                        title,
                        params_hash,
                    )
                    proj_rows.append(
                        {
                            "frame_id": fid,
                            "in_image_ratio_ground": in_ratio_g,
                            "ground_bottom_ratio": bottom_ratio,
                            "status": "ok",
                        }
                    )
            else:
                proj_rows.append({"frame_id": "NA", "status": "missing_cam_pose_or_image_dir"})
        except Exception as exc:
            proj_rows.append({"frame_id": "NA", "status": f"proj_exception:{exc}"})
    if proj_rows:
        write_csv(
            qa_dir / "fused_projection_stats.csv",
            proj_rows,
            list(proj_rows[0].keys()),
        )

    qa_row = {
        "drive_id": drive_id,
        "status": "ok",
        "params_hash": params_hash,
        "road_cover": round(road_cover, 6),
        "traj_cover": round(traj_cover, 6),
        "components_count": int(polys_out.shape[0]),
        "holes_area_ratio": round(holes_ratio, 6),
        "frame_start": int(cfg["FRAME_START"]),
        "frame_end": int(cfg["FRAME_END"]),
        "stride": int(cfg["STRIDE"]),
        "notes": ";".join(notes),
    }
    write_csv(qa_dir / "qa_index.csv", [qa_row], list(qa_row.keys()))
    report = [
        "# LiDAR Road Surface 0010 Report",
        "",
        f"- drive_id: {drive_id}",
        f"- frame_range: {int(cfg['FRAME_START'])}-{int(cfg['FRAME_END'])}",
        f"- stride: {int(cfg['STRIDE'])}",
        f"- params_hash: {params_hash}",
        f"- qa_pass: {str(0.45 <= road_cover <= 0.95 and traj_cover >= 0.85 and polys_out.shape[0] <= 80 and holes_ratio <= 0.15)}",
        "",
        "## Metrics",
        f"- road_cover: {road_cover:.6f}",
        f"- traj_cover: {traj_cover:.6f}",
        f"- components_count: {int(polys_out.shape[0])}",
        f"- holes_area_ratio: {holes_ratio:.6f}",
        "",
        "## Outputs",
        f"- {relpath(run_dir, road_path)}",
        f"- {relpath(run_dir, non_path)}",
        f"- {relpath(run_dir, sem_path)}",
        f"- {relpath(run_dir, mask_path)}",
        f"- {relpath(run_dir, road_vec_path)}",
        f"- {relpath(run_dir, quicklook_path)}",
        "",
        "## Degrade Notes",
    ]
    if proj_rows:
        report.extend(
            [
                f"- {relpath(run_dir, qa_dir / 'fused_projection_stats.csv')}",
                f"- {relpath(run_dir, qa_dir / 'fused_overlay_frame_0000000250_all_ground.png')}",
                f"- {relpath(run_dir, qa_dir / 'fused_overlay_frame_0000000341_all_ground.png')}",
                f"- {relpath(run_dir, qa_dir / 'fused_overlay_frame_0000000500_all_ground.png')}",
            ]
        )
    report.extend([f"- {n}" for n in notes] if notes else ["- none"])
    if errors:
        report.extend(["", "## Errors"] + [f"- {e}" for e in errors[:10]])
    write_text(qa_dir / "report.md", "\n".join(report))

    summary = {
        "run_id": run_id,
        "drive_id": drive_id,
        "params_hash": params_hash,
        "road_cover": road_cover,
        "traj_cover": traj_cover,
        "components_count": int(polys_out.shape[0]),
        "holes_area_ratio": holes_ratio,
        "frame_start": int(cfg["FRAME_START"]),
        "frame_end": int(cfg["FRAME_END"]),
        "stride": int(cfg["STRIDE"]),
        "notes": notes,
        "errors": errors,
    }
    write_json(run_dir / "run_summary.json", summary)
    # Postcheck
    report_text = (qa_dir / "report.md").read_text(encoding="utf-8")
    if params_hash not in report_text:
        return 3
    meta = mask_path.with_suffix(".meta.json")
    if not meta.exists() or params_hash not in meta.read_text(encoding="utf-8"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

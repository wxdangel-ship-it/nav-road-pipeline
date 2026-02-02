from __future__ import annotations

"""
Skill#2: surface_evidence_utm32
从融合点云产物生成地表证据（点云/DEM/矢量/BEV 标线），不触发 fusion。
"""

import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from pipeline.bev_dataset_export import bev_tiles_dir, build_bev_tiles_from_points
from pipeline.road_surface_evidence import (
    bbox_check_utm32,
    build_reference_surface_from_points_near_traj,
    binary_close,
    compute_candidate_stats,
    compute_candidate_stats_with_ref,
    compute_grid_shape,
    compute_slope,
    compute_z_min_grid,
    load_fusion_bounds,
    read_laz_points,
)
from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_json, write_text
from tools.manifest_hash_head import build_manifest
from tools.resolve_baseline import resolve_laz_paths

LOG = logging.getLogger("surface_evidence_skill")
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"

# =========================
# 参数区（按需修改）
# =========================
MODE = "single"  # single | batch
JOB_FILE = r"configs\jobs\surface_evidence\0010_f000_300.yaml"
BATCH_FILE = r"configs\jobs\surface_evidence\golden8_full.yaml"
OVERWRITE = False
DRY_RUN = False

CONFIG_FILE = r"configs\skills\surface_evidence_utm32.yaml"

MODE = os.environ.get("SURFACE_EVIDENCE_MODE", MODE)
JOB_FILE = os.environ.get("SURFACE_EVIDENCE_JOB_FILE", JOB_FILE)
BATCH_FILE = os.environ.get("SURFACE_EVIDENCE_BATCH_FILE", BATCH_FILE)
OVERWRITE = os.environ.get("SURFACE_EVIDENCE_OVERWRITE", str(int(OVERWRITE))) in {"1", "true", "True"}
DRY_RUN = os.environ.get("SURFACE_EVIDENCE_DRY_RUN", str(int(DRY_RUN))) in {"1", "true", "True"}
JOB_ID_FILTER = os.environ.get("SURFACE_EVIDENCE_JOB_ID", "").strip()


def _load_yaml(path: Path) -> Dict[str, object]:
    try:
        import yaml  # type: ignore
    except Exception:
        print("missing dependency: PyYAML. Install via: python -m pip install pyyaml")
        raise SystemExit(2)
    if not path.exists():
        raise FileNotFoundError(f"yaml_missing:{path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _expand_placeholders(text: str, ctx: Dict[str, str]) -> str:
    def repl_percent(match: re.Match[str]) -> str:
        var = match.group(1)
        return ctx.get(var, os.environ.get(var, ""))

    def repl_brace(match: re.Match[str]) -> str:
        var = match.group(1)
        return ctx.get(var, os.environ.get(var, ""))

    out = re.sub(r"%([^%]+)%", repl_percent, text)
    out = re.sub(r"\$\{([^}]+)\}", repl_brace, out)
    return out


def _resolve_path(raw: str, base_dir: Path, ctx: Dict[str, str]) -> Path:
    expanded = _expand_placeholders(raw, ctx)
    path = Path(expanded)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return Path(os.path.normpath(str(path)))


def _parse_pose_map(path: Path) -> Dict[str, np.ndarray]:
    pose_map = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 13:
            continue
        frame = parts[0]
        vals = list(map(float, parts[1:13]))
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :4] = np.asarray(vals, dtype=np.float64).reshape(3, 4)
        pose_map[frame] = mat
    return pose_map


def _load_world_to_utm32_transform(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"transform_missing:{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "dx": float(data.get("dx", 0.0)),
        "dy": float(data.get("dy", 0.0)),
        "dz": float(data.get("dz", 0.0)),
        "yaw_deg": float(data.get("yaw_deg", 0.0)),
        "scale": float(data.get("scale", 1.0)),
    }


def _apply_world_to_utm32(xy: np.ndarray, z: np.ndarray, tf: Dict[str, float]) -> np.ndarray:
    yaw = math.radians(float(tf.get("yaw_deg", 0.0)))
    scale = float(tf.get("scale", 1.0))
    dx = float(tf.get("dx", 0.0))
    dy = float(tf.get("dy", 0.0))
    dz = float(tf.get("dz", 0.0))
    c = math.cos(yaw)
    s = math.sin(yaw)
    r = np.array([[c, -s], [s, c]], dtype=np.float64)
    xy2 = (xy @ r.T) * scale
    out = np.zeros((xy.shape[0], 3), dtype=np.float64)
    out[:, 0] = xy2[:, 0] + dx
    out[:, 1] = xy2[:, 1] + dy
    out[:, 2] = z + dz
    return out


def _list_laz_paths(run_dir: Path) -> List[Path]:
    out_dir = run_dir / "outputs"
    parts = sorted(out_dir.glob("fused_points_utm32_part_*.laz"))
    if parts:
        return parts
    single = out_dir / "fused_points_utm32.laz"
    if single.exists():
        return [single]
    raise FileNotFoundError("fused_laz_not_found")


def _resolve_fusion_inputs(job: Dict[str, object]) -> Dict[str, object]:
    source = str(job.get("fusion_source", "")).strip()
    if not source:
        raise ValueError("fusion_source_missing")

    if source == "baseline_active":
        pointer = REPO_ROOT / "baselines" / "ACTIVE_LIDAR_FUSION_BASELINE.txt"
        baseline_dir = Path(pointer.read_text(encoding="utf-8").strip())
        laz_paths = resolve_laz_paths(baseline_dir)
        drive_id = str(job.get("drive_id") or "")
        match = re.search(r"drive_(\d{4})_sync", drive_id)
        if match:
            drive_num = match.group(1)
            laz_paths = [p for p in laz_paths if f"drive_{drive_num}_sync" in str(p)]
            evidence = baseline_dir / "evidence" / f"{drive_num}_sync"
        else:
            evidence = baseline_dir / "evidence"
        if not laz_paths:
            raise ValueError("baseline_active_no_laz_for_drive")
        bbox_geojson = evidence / "bbox_utm32.geojson"
        meta_json = evidence / "fused_points_utm32.meta.json"
        transform_json = evidence / "world_to_utm32_report.json"
        fusion_run_dir = laz_paths[0].parents[1]
        return {
            "source": source,
            "laz_paths": laz_paths,
            "bbox_geojson": bbox_geojson,
            "meta_json": meta_json,
            "transform_json": transform_json,
            "fusion_run_dir": fusion_run_dir,
            "baseline_dir": baseline_dir,
        }

    if source == "run_dir":
        run_dir = Path(str(job.get("fusion_run_dir") or "")).resolve()
        laz_paths = _list_laz_paths(run_dir)
        bbox_geojson = run_dir / "outputs" / "bbox_utm32.geojson"
        meta_json = run_dir / "outputs" / "fused_points_utm32.meta.json"
        transform_json = run_dir / "report" / "world_to_utm32_report.json"
        return {
            "source": source,
            "laz_paths": laz_paths,
            "bbox_geojson": bbox_geojson,
            "meta_json": meta_json,
            "transform_json": transform_json,
            "fusion_run_dir": run_dir,
        }

    if source == "explicit":
        raw = str(job.get("explicit_laz") or "").strip()
        if not raw:
            raise ValueError("explicit_laz_missing")
        laz_paths = sorted(Path(p) for p in Path(".").glob(raw))
        if not laz_paths:
            laz_paths = [Path(raw)]
        bbox_geojson = Path(str(job.get("explicit_bbox_geojson") or "")).resolve()
        transform_json = Path(str(job.get("explicit_transform_json") or "")).resolve()
        return {
            "source": source,
            "laz_paths": laz_paths,
            "bbox_geojson": bbox_geojson,
            "transform_json": transform_json,
            "fusion_run_dir": None,
        }

    raise ValueError(f"fusion_source_invalid:{source}")


def _read_bounds_from_geojson(path: Path) -> Tuple[float, float, float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    coords = data["features"][0]["geometry"]["coordinates"][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (min(xs), min(ys), max(xs), max(ys))


def _read_bounds_from_laz(laz_paths: Iterable[Path]) -> Tuple[float, float, float, float]:
    minx = math.inf
    miny = math.inf
    maxx = -math.inf
    maxy = -math.inf
    import laspy

    for p in laz_paths:
        with laspy.open(p) as reader:
            mins = reader.header.mins
            maxs = reader.header.maxs
            minx = min(minx, float(mins[0]))
            miny = min(miny, float(mins[1]))
            maxx = max(maxx, float(maxs[0]))
            maxy = max(maxy, float(maxs[1]))
    return (minx, miny, maxx, maxy)


def _write_raster(path: Path, bands: List[np.ndarray], transform, crs: str, nodata: List[float], dtypes: List[str]) -> None:
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": bands[0].shape[0],
        "width": bands[0].shape[1],
        "count": len(bands),
        "dtype": dtypes[0],
        "crs": crs,
        "transform": transform,
        "nodata": nodata[0],
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        for idx, band in enumerate(bands, start=1):
            dst.write(band.astype(dtypes[idx - 1]), idx)


def _extract_points_to_laz(
    laz_paths: Iterable[Path],
    z_base: np.ndarray,
    mask: np.ndarray,
    grid_bounds: Tuple[float, float, float, float],
    grid_res: float,
    dz_max: float,
    out_path: Path,
    chunk_points: int,
) -> Dict[str, object]:
    import laspy

    minx, miny, maxx, maxy = grid_bounds
    zmin = float(np.nanmin(z_base[np.isfinite(z_base)])) if np.isfinite(z_base).any() else 0.0
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001], dtype=np.float64)
    header.offsets = np.array([minx, miny, zmin], dtype=np.float64)
    try:
        from pyproj import CRS

        header.add_crs(CRS.from_epsg(32632))
    except Exception:
        pass

    points_total = 0
    points_ground = 0
    intensity_nonzero = 0
    intensity_max = 0

    with laspy.open(out_path, mode="w", header=header) as writer:
        for xs, ys, zs, intens in read_laz_points(laz_paths, chunk_points):
            points_total += int(xs.size)
            cols = np.floor((xs - minx) / grid_res).astype(np.int64)
            rows = np.floor((maxy - ys) / grid_res).astype(np.int64)
            valid = (cols >= 0) & (cols < mask.shape[1]) & (rows >= 0) & (rows < mask.shape[0])
            if not valid.any():
                continue
            cols = cols[valid]
            rows = rows[valid]
            xs = xs[valid]
            ys = ys[valid]
            zs = zs[valid]
            intens = intens[valid].astype(np.uint16)
            m = mask[rows, cols] > 0
            if not m.any():
                continue
            cols = cols[m]
            rows = rows[m]
            xs = xs[m]
            ys = ys[m]
            zs = zs[m]
            intens = intens[m]
            ref = z_base[rows, cols]
            dz = zs - ref
            keep = dz <= dz_max
            if not keep.any():
                continue
            xs = xs[keep]
            ys = ys[keep]
            zs = zs[keep]
            intens = intens[keep]
            if xs.size == 0:
                continue
            points_ground += int(xs.size)
            intensity_nonzero += int((intens > 0).sum())
            intensity_max = max(intensity_max, int(intens.max()))
            las = laspy.LasData(header)
            las.x = xs.astype(np.float64)
            las.y = ys.astype(np.float64)
            las.z = zs.astype(np.float64)
            las.intensity = intens.astype(np.uint16)
            writer.write_points(las.points)

    return {
        "points_total_read": points_total,
        "points_ground": points_ground,
        "intensity_nonzero": intensity_nonzero,
        "intensity_max": intensity_max,
    }


def _vectorize_mask(mask: np.ndarray, transform) -> Tuple[object, int]:
    import rasterio.features
    from shapely.geometry import shape
    from shapely.ops import unary_union

    shapes = list(rasterio.features.shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform))
    polys = [shape(geom) for geom, val in shapes if val == 1]
    if not polys:
        raise RuntimeError("mask_empty")
    merged = unary_union(polys)
    holes = 0
    try:
        if merged.geom_type == "Polygon":
            holes = len(merged.interiors)
        elif merged.geom_type == "MultiPolygon":
            holes = sum(len(p.interiors) for p in merged.geoms)
    except Exception:
        holes = 0
    return merged, holes


def _write_polygon(path: Path, geom, attrs: Dict[str, object]) -> None:
    import geopandas as gpd

    gdf = gpd.GeoDataFrame([attrs], geometry=[geom], crs="EPSG:32632")
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, layer="road_surface", driver="GPKG")


def _write_preview_geojson(path: Path, geom, simplify_m: float) -> None:
    import geopandas as gpd

    g = geom.simplify(simplify_m, preserve_topology=True)
    gdf = gpd.GeoDataFrame([{}], geometry=[g], crs="EPSG:32632")
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")


def _write_png_preview(path: Path, arr: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    img = arr.copy()
    if mask is not None:
        img = np.where(mask > 0, img, np.nan)
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="viridis")
    plt.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def _build_bev_rois(
    points_path: Path,
    crosswalk_gpkg: Path,
    out_dir: Path,
    res_m: float,
    buf_m: float,
    win_large_m: float,
    win_small_m: float,
    tophat_clip_max: int,
    chunk_points: int,
    preview_style: str = "legacy_gray",
) -> Dict[str, object]:
    import geopandas as gpd
    import rasterio
    import rasterio.transform
    from shapely.prepared import prep

    from pipeline.bev_markings import box_sum, load_polygon_bounds, rasterize_polygon_mask, read_laz_points

    if not crosswalk_gpkg.exists():
        return {"roi_count": 0}
    gdf = gpd.read_file(str(crosswalk_gpkg))
    if gdf.empty:
        return {"roi_count": 0}
    if gdf.crs is None or gdf.crs.to_epsg() != 32632:
        gdf = gdf.to_crs(epsg=32632)

    roi_dir = out_dir / "bev_rois_r005m"
    roi_dir.mkdir(parents=True, exist_ok=True)
    road_geom = None
    polygon_path = out_dir / "road_surface_polygon_utm32.gpkg"
    if polygon_path.exists():
        try:
            road_geom, _bounds = load_polygon_bounds(polygon_path, layer="road_surface")
        except Exception:
            road_geom = None
    index_items = []
    preview_gray_is_true = None
    roi_stats: List[Dict[str, object]] = []
    for idx, geom in enumerate(gdf.geometry):
        buf = geom.buffer(buf_m)
        minx, miny, maxx, maxy = buf.bounds
        width = int(math.ceil((maxx - minx) / res_m))
        height = int(math.ceil((maxy - miny) / res_m))
        if width <= 1 or height <= 1:
            continue
        intensity_max = np.zeros((height, width), dtype=np.uint16)
        density = np.zeros((height, width), dtype=np.uint32)
        prep_buf = prep(buf)
        for xs, ys, _zs, intens in read_laz_points(points_path, chunk_points):
            cols = np.floor((xs - minx) / res_m).astype(np.int64)
            rows = np.floor((maxy - ys) / res_m).astype(np.int64)
            valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
            if not valid.any():
                continue
            xs_v = xs[valid]
            ys_v = ys[valid]
            intens_v = intens[valid].astype(np.uint16)
            cols = cols[valid]
            rows = rows[valid]
            # 精确 in-buffer 过滤
            try:
                from shapely import points as shp_points
                from shapely import contains as shp_contains

                pts = shp_points(xs_v, ys_v)
                in_mask = shp_contains(buf, pts)
            except Exception:
                from shapely.geometry import Point

                in_mask = np.array([prep_buf.contains(Point(x, y)) for x, y in zip(xs_v, ys_v)])
            if not np.any(in_mask):
                continue
            cols = cols[in_mask]
            rows = rows[in_mask]
            intens_v = intens_v[in_mask]
            np.maximum.at(intensity_max, (rows, cols), intens_v)
            np.add.at(density, (rows, cols), 1)

        mask = (density > 0).astype(np.uint8)
        if road_geom is not None:
            poly_mask = rasterize_polygon_mask(road_geom, (minx, miny, maxx, maxy), res_m, (height, width))
            mask = ((mask > 0) & (poly_mask > 0)).astype(np.uint8)
        a = intensity_max.astype(np.float32) * mask.astype(np.float32)
        w = mask.astype(np.float32)
        r_l = int(round((win_large_m / 2.0) / res_m))
        r_s = int(round((win_small_m / 2.0) / res_m))
        sum_a = box_sum(a, r_l)
        sum_w = box_sum(w, r_l)
        mean_l = sum_a / np.maximum(sum_w, 1.0)
        top_hat = intensity_max.astype(np.float32) - mean_l
        top_hat[top_hat < 0] = 0
        top_hat_u16 = np.clip(top_hat, 0, tophat_clip_max).astype(np.uint16)
        top_hat_u16[mask == 0] = 0

        transform = rasterio.transform.from_origin(minx, maxy, res_m, res_m)
        tif_path = roi_dir / f"cw_{idx:03d}_bev.tif"
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype="uint16",
            crs="EPSG:32632",
            transform=transform,
            compress="deflate",
        ) as dst:
            dst.write(intensity_max.astype(np.uint16), 1)
            dst.set_band_description(1, "intensity_max")
            dst.write(np.clip(density, 0, 65535).astype(np.uint16), 2)
            dst.set_band_description(2, "density")
            dst.write(top_hat_u16, 3)
            dst.set_band_description(3, "top_hat_u16")

        # previews
        _write_png_preview(roi_dir / f"cw_{idx:03d}_top_hat.png", top_hat_u16.astype(np.float32))
        _write_png_preview(roi_dir / f"cw_{idx:03d}_raw_intensity.png", intensity_max.astype(np.float32), mask)
        try:
            import matplotlib.pyplot as plt

            base = np.clip(top_hat_u16.astype(np.float32) / float(tophat_clip_max) * 255.0, 0, 255).astype(np.uint8)
            plt.figure(figsize=(4, 4))
            plt.imshow(base, cmap="gray")
            plt.contour(mask, levels=[0.5], colors="green", linewidths=0.6)
            plt.axis("off")
            plt.savefig(roi_dir / f"cw_{idx:03d}_overlay.png", dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close()
        except Exception:
            pass

        if preview_style == "legacy_gray":
            vals = top_hat_u16[mask > 0].astype(np.float32)
            p2 = float(np.percentile(vals, 2)) if vals.size > 0 else 0.0
            p98 = float(np.percentile(vals, 98)) if vals.size > 0 else 1.0
            if p98 <= p2:
                p98 = p2 + 1.0
            scaled = np.clip((top_hat_u16.astype(np.float32) - p2) / (p98 - p2), 0.0, 1.0)
            gray_pctl = (scaled * 255.0).astype(np.uint8)
            gray_pctl[mask == 0] = 0
            gray_pctl_path = roi_dir / f"cw_{idx:03d}_top_hat_gray_pctl.png"
            try:
                import matplotlib.pyplot as plt

                plt.imsave(gray_pctl_path, gray_pctl.astype(np.uint8), cmap="gray", vmin=0, vmax=255)
            except Exception:
                pass

            fixed = np.clip(top_hat_u16.astype(np.float32) / 4000.0 * 255.0, 0.0, 255.0).astype(np.uint8)
            fixed[mask == 0] = 0
            fixed_path = roi_dir / f"cw_{idx:03d}_top_hat_gray_fixed.png"
            try:
                import matplotlib.pyplot as plt

                plt.imsave(fixed_path, fixed.astype(np.uint8), cmap="gray", vmin=0, vmax=255)
            except Exception:
                pass

            color_path = roi_dir / f"cw_{idx:03d}_top_hat_color_pctl.png"
            try:
                import matplotlib.pyplot as plt

                plt.imsave(color_path, gray_pctl.astype(np.uint8), cmap="viridis", vmin=0, vmax=255)
            except Exception:
                pass

            if preview_gray_is_true is None:
                try:
                    import matplotlib.image as mpimg

                    img = mpimg.imread(gray_pctl_path)
                    if img.ndim == 2:
                        preview_gray_is_true = True
                    elif img.ndim == 3 and img.shape[2] >= 3:
                        rgb = img[..., :3]
                        diff_rg = np.abs(rgb[..., 0] - rgb[..., 1]).max()
                        diff_gb = np.abs(rgb[..., 1] - rgb[..., 2]).max()
                        preview_gray_is_true = float(max(diff_rg, diff_gb)) == 0.0
                    else:
                        preview_gray_is_true = False
                except Exception:
                    preview_gray_is_true = False

        index_items.append(
            {
                "id": idx,
                "path": str(tif_path),
                "bbox": [minx, miny, maxx, maxy],
                "res_m": res_m,
            }
        )
        if vals.size > 0:
            roi_stats.append(
                {
                    "id": idx,
                    "top_hat_p50": float(np.percentile(vals, 50)),
                    "top_hat_p95": float(np.percentile(vals, 95)),
                    "top_hat_p98": float(np.percentile(vals, 98)),
                }
            )

    if index_items:
        import geopandas as gpd
        from shapely.geometry import box

        gdf_idx = gpd.GeoDataFrame(
            index_items,
            geometry=[box(*item["bbox"]) for item in index_items],
            crs="EPSG:32632",
        )
        gdf_idx.to_file(out_dir / "bev_rois_index.geojson", driver="GeoJSON")
    return {
        "roi_count": len(index_items),
        "roi_stats": roi_stats,
        "preview_style": preview_style,
        "preview_gray_is_true": preview_gray_is_true,
    }


def _build_surface(job: Dict[str, object], defaults: Dict[str, object]) -> Dict[str, object]:
    preset = defaults.get("preset", {})
    runtime = defaults.get("runtime", {})
    dem_cfg = defaults.get("dem", {})
    bev_cfg = defaults.get("bev", {})
    job_bev = job.get("bev", {}) if isinstance(job.get("bev", {}), dict) else {}
    tiles_cfg = job_bev.get("tiles", {}) if isinstance(job_bev.get("tiles", {}), dict) else {}
    top_hat_cfg = job_bev.get("top_hat", {}) if isinstance(job_bev.get("top_hat", {}), dict) else {}
    preview_cfg = job.get("preview", {}) if isinstance(job.get("preview", {}), dict) else {}

    def _coalesce(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    bev_res_m_tiles = float(
        _coalesce(tiles_cfg.get("res_m"), job_bev.get("res_m_tiles"), bev_cfg.get("res_m_tiles"))
    )
    bev_tile_size_px = int(
        _coalesce(tiles_cfg.get("tile_size_px"), job_bev.get("tile_size_px"), bev_cfg.get("tile_size_px"))
    )
    bev_tile_overlap_px = int(
        _coalesce(tiles_cfg.get("overlap_px"), job_bev.get("tile_overlap_px"), bev_cfg.get("tile_overlap_px"))
    )
    bev_keep_mask_area_m2 = float(
        _coalesce(tiles_cfg.get("keep_if_mask_area_m2_ge"), job_bev.get("tile_keep_mask_area_m2"), bev_cfg.get("tile_keep_mask_area_m2"))
    )
    bev_win_large_m = float(_coalesce(top_hat_cfg.get("win_large_m"), bev_cfg.get("win_large_m")))
    bev_win_small_m = float(_coalesce(top_hat_cfg.get("win_small_m"), bev_cfg.get("win_small_m")))
    bev_clip_max_u16 = int(_coalesce(top_hat_cfg.get("clip_max_u16"), bev_cfg.get("tophat_clip_max")))
    preview_style = str(_coalesce(preview_cfg.get("preview_style"), "legacy_gray"))

    drive_id = str(job["drive_id"])
    frame_start = int(job["frame_start"])
    frame_end = int(job["frame_end"])

    run_id = now_ts()
    run_dir = RUNS_DIR / f"surface_evidence_{drive_id}_{frame_start}_{frame_end}_{run_id}"
    if OVERWRITE:
        ensure_overwrite(run_dir)
    setup_logging(run_dir / "logs" / "run.log")
    LOG.info("run_start")

    fusion = _resolve_fusion_inputs(job)
    laz_paths: List[Path] = fusion["laz_paths"]

    if fusion.get("bbox_geojson") and Path(fusion["bbox_geojson"]).exists():
        bounds = _read_bounds_from_geojson(Path(fusion["bbox_geojson"]))
    else:
        bounds = _read_bounds_from_laz(laz_paths)

    bbox_ok = bbox_check_utm32(bounds)
    if not bbox_ok.get("ok"):
        raise RuntimeError(f"bbox_check_failed:{bbox_ok}")

    width, height, grid_bounds = compute_grid_shape(
        bounds, float(preset["grid_res_m"]), float(runtime["bbox_pad_m"])
    )
    minx, miny, maxx, maxy = grid_bounds

    use_traj = bool(preset.get("use_traj_xy_prior", True))
    traj_line = None
    traj_xy = None
    traj_z = None
    ref_info = {}
    layer_info = {"enabled": False}
    seed_info = {}

    if use_traj:
        kitti_root = str(job.get("kitti_root") or "")
        if not kitti_root:
            raise RuntimeError("kitti_root_missing")
        pose_path = Path(kitti_root) / "data_poses" / drive_id / "cam0_to_world.txt"
        if not pose_path.exists():
            raise RuntimeError("cam0_to_world_missing")
        tf_path = Path(fusion.get("transform_json") or "")
        if not tf_path.exists():
            raise RuntimeError("transform_json_missing")
        transform = _load_world_to_utm32_transform(tf_path)
        pose_map = _parse_pose_map(pose_path)
        traj = []
        for i in range(frame_start, frame_end + 1):
            fid_pad = f"{i:010d}"
            fid_plain = str(i)
            mat = None
            if fid_pad in pose_map:
                mat = pose_map[fid_pad]
            elif fid_plain in pose_map:
                mat = pose_map[fid_plain]
            if mat is None:
                continue
            traj.append([float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3])])
        if not traj:
            raise RuntimeError("traj_empty")
        traj = np.asarray(traj, dtype=np.float64)
        traj_utm = _apply_world_to_utm32(traj[:, :2], traj[:, 2], transform)
        traj_xy = traj_utm[:, :2]
        traj_z = traj_utm[:, 2]
        from shapely.geometry import LineString

        traj_line = LineString(traj_xy.tolist())
        ref_result = build_reference_surface_from_points_near_traj(
            laz_paths=laz_paths,
            traj_line=traj_line,
            traj_xy=traj_xy,
            traj_z=traj_z,
            grid_bounds=grid_bounds,
            ref_grid_res=float(preset["ref_grid_res_m"]),
            ref_local_radius_m=float(preset["ref_local_radius_m"]),
            ref_q_low=float(preset["ref_q_low"]),
            ref_min_pts=int(preset["ref_min_pts"]),
            ref_smooth_win_m=float(preset["ref_smooth_win_m"]),
            layer_reject_below_m=float(preset["layer_reject_below_traj_m"]),
            layer_reject_above_m=float(preset["layer_reject_above_traj_m"]),
            chunk_points=int(runtime["chunk_points"]),
        )
        z_ref = ref_result["z_ref"]
        ref_valid = ref_result["valid_mask"]
        ref_bounds = ref_result["grid_bounds"]
        ref_grid_res = float(ref_result["grid_res"])
        z_ref_fine = np.full((height, width), np.nan, dtype=np.float32)
        ref_h, ref_w = z_ref.shape
        ref_minx, ref_miny, ref_maxx, ref_maxy = ref_bounds
        for r in range(height):
            y = (maxy - (r + 0.5) * float(preset["grid_res_m"]))
            rr = int(math.floor((ref_maxy - y) / ref_grid_res))
            if rr < 0 or rr >= ref_h:
                continue
            for c in range(width):
                x = (minx + (c + 0.5) * float(preset["grid_res_m"]))
                cc = int(math.floor((x - ref_minx) / ref_grid_res))
                if cc < 0 or cc >= ref_w:
                    continue
                if ref_valid[rr, cc] > 0:
                    z_ref_fine[r, c] = z_ref[rr, cc]
        cand = compute_candidate_stats_with_ref(
            laz_paths, z_ref_fine, np.isfinite(z_ref_fine), grid_bounds, float(preset["grid_res_m"]),
            float(preset["dz_ground_max_m"]), int(runtime["chunk_points"])
        )
        slope = compute_slope(z_ref_fine, float(preset["grid_res_m"]))
        mask_pre, road_geom, mask_stats = compute_mask_from_stats(
            z_ref_fine,
            cand,
            slope,
            grid_bounds,
            float(preset["grid_res_m"]),
            int(preset["cell_min_ground_pts"]),
            float(preset["cell_max_roughness_m"]),
            float(preset["cell_max_slope"]),
            float(preset["island_area_min_m2"]),
            float(preset["hole_area_max_m2"]),
        )
        import rasterio
        import rasterio.features

        seed_geom = traj_line.buffer(float(preset["seed_buffer_m"]))
        seed_mask = rasterio.features.rasterize(
            [(seed_geom, 1)],
            out_shape=mask_pre.shape,
            transform=rasterio.transform.from_origin(minx, maxy, float(preset["grid_res_m"]), float(preset["grid_res_m"])),
            fill=0,
            dtype=np.uint8,
        )
        close_radius = float(preset.get("close_radius_m", 0.0))
        close_iter = int(preset.get("close_iter", 0))
        if close_radius > 0 and close_iter > 0:
            mask_pre = binary_close(mask_pre, int(round(close_radius / float(preset["grid_res_m"]))), close_iter)
        from pipeline.road_surface_evidence import keep_components_intersecting_seed

        mask, cc_info = keep_components_intersecting_seed(
            mask_pre, seed_mask, int(preset.get("max_keep_components", 5))
        )
        ref_info = {
            "ref_grid_res_m": float(preset["ref_grid_res_m"]),
            "ref_q_low": float(preset["ref_q_low"]),
            "ref_min_pts": int(preset["ref_min_pts"]),
            "ref_valid_ratio": float(ref_result["ref_valid_ratio"]),
            "ref_filled_ratio": float(ref_result["ref_filled_ratio"]),
            "ref_source_points_count": int(ref_result["used_points"]),
        }
        layer_info = {
            "enabled": True,
            "reject_below_m": float(preset["layer_reject_below_traj_m"]),
            "reject_above_m": float(preset["layer_reject_above_traj_m"]),
        }
        seed_info = {
            "seed_cells": int(seed_mask.sum()),
            "kept_component_ids": cc_info.get("kept_component_ids", []),
            "kept_area_cells": cc_info.get("kept_area_cells", 0),
        }
        z_base = z_ref_fine
        ref_valid_mask = (np.isfinite(z_ref_fine)).astype(np.uint8)
    else:
        z_min, _z_meta = compute_z_min_grid(
            laz_paths, bounds, float(preset["grid_res_m"]), float(runtime["bbox_pad_m"]),
            RUNS_DIR / "surface_cache", int(runtime["chunk_points"])
        )
        cand = compute_candidate_stats(
            laz_paths, z_min, grid_bounds, float(preset["grid_res_m"]),
            float(preset["dz_ground_max_m"]), RUNS_DIR / "surface_cache", int(runtime["chunk_points"])
        )
        slope = compute_slope(z_min, float(preset["grid_res_m"]))
        mask, road_geom, mask_stats = compute_mask_from_stats(
            z_min,
            cand,
            slope,
            grid_bounds,
            float(preset["grid_res_m"]),
            int(preset["cell_min_ground_pts"]),
            float(preset["cell_max_roughness_m"]),
            float(preset["cell_max_slope"]),
            float(preset["island_area_min_m2"]),
            float(preset["hole_area_max_m2"]),
        )
        z_base = z_min
        ref_valid_mask = (np.isfinite(z_min)).astype(np.uint8)

    # build z_std
    count = cand["cand_count"].astype(np.float32)
    mean = np.zeros_like(count, dtype=np.float32)
    std = np.zeros_like(count, dtype=np.float32)
    valid = count > 0
    mean[valid] = (cand["cand_sum_z"][valid] / count[valid]).astype(np.float32)
    var = np.zeros_like(count, dtype=np.float32)
    var[valid] = (cand["cand_sum_z2"][valid] / count[valid] - mean[valid] ** 2).astype(np.float32)
    var[var < 0] = 0
    std[valid] = np.sqrt(var[valid]).astype(np.float32)

    # outputs
    outputs = run_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    dem_path = outputs / "surface_dem_utm32.tif"
    quality_path = outputs / "surface_dem_quality_utm32.tif"

    import rasterio

    transform = rasterio.transform.from_origin(minx, maxy, float(preset["grid_res_m"]), float(preset["grid_res_m"]))
    dem = z_base.copy().astype(np.float32)
    dem[~np.isfinite(dem)] = -9999.0
    dem[mask == 0] = -9999.0
    density = cand["cand_count"].astype(np.uint32)
    density[mask == 0] = 0
    z_std = std.astype(np.float32)
    z_std[mask == 0] = -9999.0
    ref_valid_mask = ref_valid_mask.astype(np.uint8)

    _write_raster(dem_path, [dem], transform, "EPSG:32632", [-9999.0], ["float32"])
    _write_raster(
        quality_path,
        [density, z_std, mask.astype(np.uint8), ref_valid_mask],
        transform,
        "EPSG:32632",
        [0, -9999.0, 0, 0],
        ["uint32", "float32", "uint8", "uint8"],
    )

    _write_png_preview(outputs / "surface_dem_preview.png", dem, mask)

    polygon_path = outputs / "road_surface_polygon_utm32.gpkg"
    road_geom, holes_cnt = _vectorize_mask(mask.astype(np.uint8), transform)
    _write_polygon(
        polygon_path,
        road_geom,
        {
            "area_m2": float(road_geom.area),
            "grid_res_m": float(preset["grid_res_m"]),
            "dz_th_m": float(preset["dz_ground_max_m"]),
            "rough_th_m": float(preset["cell_max_roughness_m"]),
            "slope_th": float(preset["cell_max_slope"]),
            "min_pts": int(preset["cell_min_ground_pts"]),
            "preset_id": "surface_evidence",
        },
    )
    _write_preview_geojson(outputs / "road_surface_polygon_preview.geojson", road_geom, float(preset["grid_res_m"]) * 2.0)

    points_path = outputs / "road_surface_points_utm32.laz"
    stats = _extract_points_to_laz(
        laz_paths,
        z_base,
        mask.astype(np.uint8),
        grid_bounds,
        float(preset["grid_res_m"]),
        float(preset["dz_ground_max_m"]),
        points_path,
        int(runtime["chunk_points"]),
    )
    intensity_nonzero_ratio = stats["intensity_nonzero"] / float(stats["points_ground"] or 1)
    ratio_ground = stats["points_ground"] / float(stats["points_total_read"] or 1)
    write_json(
        outputs / "road_surface_points_utm32.meta.json",
        {
            "points_total_read": stats["points_total_read"],
            "points_road_surface": stats["points_ground"],
            "ratio_road_surface": ratio_ground,
            "intensity_max": stats["intensity_max"],
            "intensity_nonzero_ratio": intensity_nonzero_ratio,
            "bbox": {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy},
            "epsg": 32632,
        },
    )

    bev_stats = {}
    roi_stats = {}
    if job.get("output", {}).get("bev_markings", True):
        bev_stats = build_bev_tiles_from_points(
            [points_path],
            outputs,
            bev_res_m_tiles,
            bev_tile_size_px,
            bev_tile_overlap_px,
            float(runtime["bbox_pad_m"]),
            bev_win_large_m,
            bev_win_small_m,
            bev_clip_max_u16,
            polygon_path=polygon_path,
            tile_keep_mask_area_m2=bev_keep_mask_area_m2,
            chunk_points=int(runtime["chunk_points"]),
        )
        crosswalk_gpkg = Path(str(job.get("crosswalk_gpkg") or ""))
        if crosswalk_gpkg.exists():
            roi_stats = _build_bev_rois(
                points_path,
                crosswalk_gpkg,
                outputs,
                float(bev_cfg["res_m_roi"]),
                float(job.get("roi_buf_m", 30.0)),
                bev_win_large_m,
                bev_win_small_m,
                bev_clip_max_u16,
                int(runtime["chunk_points"]),
                preview_style=preview_style,
            )

    bev_snapshot = dict(bev_cfg)
    bev_snapshot["tiles"] = {
        "res_m": bev_res_m_tiles,
        "tile_size_px": bev_tile_size_px,
        "overlap_px": bev_tile_overlap_px,
        "keep_if_mask_area_m2_ge": bev_keep_mask_area_m2,
    }
    bev_snapshot["top_hat"] = {
        "win_large_m": bev_win_large_m,
        "win_small_m": bev_win_small_m,
        "clip_max_u16": bev_clip_max_u16,
    }
    params_snapshot = {
        "preset": preset,
        "dem": dem_cfg,
        "bev": bev_snapshot,
        "runtime": runtime,
    }
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "params.json", params_snapshot)

    mask_cells = float(mask.sum())
    dem_valid_ratio = float((dem != -9999.0).sum()) / mask_cells if mask_cells > 0 else 0.0
    roi_top_hat_p50 = None
    roi_top_hat_p95 = None
    roi_top_hat_p98 = None
    roi_top_hat_p98_over_p50 = None
    roi_stats_list = roi_stats.get("roi_stats", []) if isinstance(roi_stats, dict) else []
    if roi_stats_list:
        merged_p50 = [item.get("top_hat_p50", 0.0) for item in roi_stats_list]
        merged_p95 = [item.get("top_hat_p95", 0.0) for item in roi_stats_list]
        merged_p98 = [item.get("top_hat_p98", 0.0) for item in roi_stats_list]
        if merged_p50:
            roi_top_hat_p50 = float(np.mean(merged_p50))
        if merged_p95:
            roi_top_hat_p95 = float(np.mean(merged_p95))
        if merged_p98:
            roi_top_hat_p98 = float(np.mean(merged_p98))
        if roi_top_hat_p50 and roi_top_hat_p50 > 0:
            roi_top_hat_p98_over_p50 = float(roi_top_hat_p98 / roi_top_hat_p50)

    preview_gray_is_true = None
    if isinstance(roi_stats, dict):
        preview_gray_is_true = roi_stats.get("preview_gray_is_true")

    if bev_stats:
        bev_stats["tiles"] = {
            "res_m": bev_res_m_tiles,
            "tile_size_px": bev_tile_size_px,
            "overlap_px": bev_tile_overlap_px,
            "tiles_count": bev_stats.get("tiles_count"),
            "empty_tile_ratio": bev_stats.get("empty_tile_ratio"),
        }

    metrics = {
        "epsg": 32632,
        "bbox": {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy},
        "bbox_check": bbox_ok,
        "fusion_source": fusion["source"],
        "fusion_run_dir": str(fusion.get("fusion_run_dir") or ""),
        "points_total_read": stats["points_total_read"],
        "points_road_surface": stats["points_ground"],
        "ratio_road_surface": ratio_ground,
        "intensity_max": stats["intensity_max"],
        "intensity_nonzero_ratio": intensity_nonzero_ratio,
        "dem_res_m": float(preset["grid_res_m"]),
        "dem_valid_ratio": dem_valid_ratio,
        "dem_density_p50": float(np.percentile(density[density > 0], 50)) if (density > 0).any() else 0.0,
        "dem_density_p95": float(np.percentile(density[density > 0], 95)) if (density > 0).any() else 0.0,
        "dem_zstd_p95": float(np.percentile(z_std[z_std > 0], 95)) if (z_std > 0).any() else 0.0,
        "polygon_area_m2": float(road_geom.area),
        "polygon_valid": True,
        "holes_cnt": int(holes_cnt),
        "bev": bev_stats,
        "bev_rois": roi_stats,
        "bev_input_source": "road_surface_points_utm32.laz" if bev_stats else "",
        "bev_points_used": int(bev_stats.get("points_in_mask", 0) or bev_stats.get("points_used_in_extent", 0))
    if bev_stats
        else 0,
        "bev_mask_area_m2": float(bev_stats.get("mask_area_m2", 0.0)) if bev_stats else 0.0,
        "top_hat_p95": bev_stats.get("top_hat_p95") if bev_stats else None,
        "top_hat_p98": bev_stats.get("top_hat_p98") if bev_stats else None,
        "roi_top_hat_p50": roi_top_hat_p50,
        "roi_top_hat_p95": roi_top_hat_p95,
        "roi_top_hat_p98": roi_top_hat_p98,
        "roi_top_hat_p98_over_p50": roi_top_hat_p98_over_p50,
        "preview_style": "legacy_gray",
        "preview_gray_is_true": preview_gray_is_true,
        "ref_surface": ref_info,
        "layer_filter": layer_info,
        "mask_seed": seed_info,
    }
    write_json(report_dir / "metrics.json", metrics)

    gates = {
        "epsg_ok": True,
        "bbox_ok": bool(bbox_ok.get("ok")),
        "points_ok": stats["points_ground"] > 0 and ratio_ground >= 0.02,
        "dem_ok": dem_valid_ratio >= 0.30,
        "bev_ok": bool(bev_stats.get("tiles_count", 0) > 0 and bev_stats.get("empty_tile_ratio", 1.0) < 0.9)
        if bev_stats
        else True,
        "roi_ok": (roi_stats.get("roi_count", 0) > 0) if (job.get("crosswalk_gpkg")) else True,
    }
    write_json(report_dir / "gates.json", gates)

    # large files manifest
    large_paths = [
        points_path,
        dem_path,
        quality_path,
        polygon_path,
    ]
    tiles_dir = bev_tiles_dir(outputs, bev_res_m_tiles)
    if tiles_dir.exists():
        large_paths.extend(list(tiles_dir.glob("*.tif")))
    manifest = build_manifest(large_paths)
    write_json(outputs / "large_files_manifest.json", manifest)

    write_text(run_dir / "logs" / "run_tail.log", (run_dir / "logs" / "run.log").read_text(encoding="utf-8")[-10000:])
    LOG.info("run_done")
    return {"run_dir": run_dir, "metrics": metrics}


def compute_mask_from_stats(
    z_base: np.ndarray,
    cand: Dict[str, np.ndarray],
    slope: np.ndarray,
    grid_bounds: Tuple[float, float, float, float],
    grid_res: float,
    min_pts: int,
    rough_max: float,
    slope_max: float,
    island_min_m2: float,
    hole_max_m2: float,
) -> Tuple[np.ndarray, object, Dict[str, object]]:
    from pipeline.road_surface_evidence import mask_from_stats

    mask, geom, stats = mask_from_stats(
        z_min=z_base,
        cand_count=cand["cand_count"],
        cand_sum_z=cand["cand_sum_z"],
        cand_sum_z2=cand["cand_sum_z2"],
        slope=slope,
        grid_bounds=grid_bounds,
        grid_res=grid_res,
        min_pts=min_pts,
        rough_max=rough_max,
        slope_max=slope_max,
        island_min_m2=island_min_m2,
        hole_max_m2=hole_max_m2,
        keep_mode="largest",
    )
    return mask, geom, stats


def _normalize_job(job: Dict[str, object], defaults: Dict[str, object], job_dir: Path) -> Dict[str, object]:
    out = dict(job)
    if "output" not in out and "output_toggles" in out:
        out["output"] = out.get("output_toggles")
    required = ["drive_id", "frame_start", "frame_end", "fusion_source"]
    missing = []
    for k in required:
        if k not in out:
            missing.append(k)
            continue
        val = out.get(k)
        if val is None:
            missing.append(k)
            continue
        if isinstance(val, str) and not val.strip():
            missing.append(k)
    if missing:
        raise ValueError(f"job_missing_required:{missing}")

    ctx = {
        "REPO_ROOT": str(REPO_ROOT),
        "JOB_DIR": str(job_dir),
        "KITTI_ROOT": str(os.environ.get("KITTI_ROOT", "")),
    }
    out["kitti_root"] = _expand_placeholders(str(out.get("kitti_root") or ""), ctx)
    return out


def _run_job(job_path: Path) -> Dict[str, object]:
    defaults = _load_yaml(Path(CONFIG_FILE))
    job = _load_yaml(job_path)
    job = _normalize_job(job, defaults, job_path.parent)
    if DRY_RUN:
        print(f"dry_run plan: drive_id={job['drive_id']} frames={job['frame_start']}-{job['frame_end']} source={job['fusion_source']}")
        return {"run_dir": "", "metrics": {}}
    return _build_surface(job, defaults)


def main() -> int:
    if MODE == "single":
        _run_job(Path(JOB_FILE))
        return 0
    if MODE == "batch":
        batch = _load_yaml(Path(BATCH_FILE))
        jobs = batch.get("jobs") or []
        defaults = _load_yaml(Path(CONFIG_FILE))
        batch_dir = Path(BATCH_FILE).parent
        for item in jobs:
            if isinstance(item, dict) and item.get("drive_id"):
                if JOB_ID_FILTER and str(item.get("job_id") or "") != JOB_ID_FILTER:
                    continue
                job = _normalize_job(item, defaults, batch_dir)
                if DRY_RUN:
                    print(f"dry_run plan: drive_id={job['drive_id']} frames={job['frame_start']}-{job['frame_end']} source={job['fusion_source']}")
                    continue
                _build_surface(job, defaults)
                continue
            path = item.get("path") if isinstance(item, dict) else item
            if not path:
                continue
            _run_job(Path(path))
        return 0
    raise SystemExit("invalid MODE")


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

from pipeline.lidar_semantic.export_pointcloud import write_las
from scripts.pipeline_common import (
    ensure_dir,
    ensure_overwrite,
    now_ts,
    relpath,
    setup_logging,
    write_csv,
    write_gpkg_layer,
    write_json,
    write_text,
)

LOG = logging.getLogger("intensity_extract_0010_f280_300")

# Fixed parameters (no autotune)
ROI_PAD_M = 30.0
BG_INNER_M = 10.0
BG_OUTER_M = 35.0
GROUND_BAND_DZ_M = 0.12
GRID_RES_M = 0.20
HIGH_INT_PCTL_GLOBAL = 99.7
HIGH_INT_PCTL_INSIDE = 99.0
MAX_EXPORT_POINTS = 1_000_000
ENABLE_SHIFT_SEARCH = True
SHIFT_MAX_M = 6.0
SHIFT_STEP_M = 0.5
TARGET_EPSG = 32632

TRUTH_GPKG = Path(r"E:\Work\nav-road-pipeline\crosswalk_truth_utm32.gpkg")


@dataclass
class LasMeta:
    point_count: int
    point_record_length: int
    offset_to_points: int
    scales: Tuple[float, float, float]
    offsets: Tuple[float, float, float]
    bounds: Tuple[float, float, float, float, float, float]


def _read_las_header(path: Path) -> LasMeta:
    import struct

    header_size = 227
    with path.open("rb") as f:
        header = f.read(header_size)
    fmt = "<4sHH16sBB32s32sHHH I I B H I 5I ddd ddd dddddd"
    values = struct.unpack(fmt, header[: struct.calcsize(fmt)])
    (
        sig,
        _file_src,
        _global_enc,
        _proj_id,
        _ver_major,
        _ver_minor,
        _sys_id,
        _gen_soft,
        _doy,
        _year,
        _header_size,
        offset_to_points,
        _num_vlrs,
        _point_format,
        point_record_length,
        point_count,
        *_num_by_return,
        sx,
        sy,
        sz,
        ox,
        oy,
        oz,
        maxx,
        minx,
        maxy,
        miny,
        maxz,
        minz,
    ) = values
    if sig != b"LASF":
        raise RuntimeError("invalid_las_signature")
    return LasMeta(
        point_count=int(point_count),
        point_record_length=int(point_record_length),
        offset_to_points=int(offset_to_points),
        scales=(float(sx), float(sy), float(sz)),
        offsets=(float(ox), float(oy), float(oz)),
        bounds=(float(minx), float(miny), float(minz), float(maxx), float(maxy), float(maxz)),
    )


def _read_las_points(path: Path) -> Tuple[np.ndarray, np.ndarray, LasMeta]:
    meta = _read_las_header(path)
    if meta.point_record_length < 20:
        raise RuntimeError("unsupported_point_record_length")
    with path.open("rb") as f:
        f.seek(meta.offset_to_points)
        raw = f.read(meta.point_count * meta.point_record_length)
    pad = meta.point_record_length - 20
    dtype = np.dtype(
        [
            ("x", "<i4"),
            ("y", "<i4"),
            ("z", "<i4"),
            ("intensity", "<u2"),
            ("bitfield", "u1"),
            ("classification", "u1"),
            ("scan_angle", "i1"),
            ("user_data", "u1"),
            ("point_src", "<u2"),
            ("pad", f"V{pad}") if pad > 0 else ("pad", "V0"),
        ]
    )
    arr = np.frombuffer(raw, dtype=dtype, count=meta.point_count)
    sx, sy, sz = meta.scales
    ox, oy, oz = meta.offsets
    x = arr["x"].astype(np.float64) * sx + ox
    y = arr["y"].astype(np.float64) * sy + oy
    z = arr["z"].astype(np.float64) * sz + oz
    intensity = arr["intensity"].astype(np.uint16)
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    return points, intensity, meta


def _list_layers(path: Path) -> List[Dict[str, str]]:
    import pyogrio

    layers_raw = pyogrio.list_layers(str(path))
    rows = []
    for item in layers_raw:
        if isinstance(item, dict):
            name = str(item.get("name", ""))
            gtype = str(item.get("geometry_type", ""))
        else:
            name = str(item[0]) if len(item) > 0 else ""
            gtype = str(item[1]) if len(item) > 1 else ""
        rows.append({"name": name, "geometry_type": gtype})
    return rows


def _pick_truth_layer(layers: List[Dict[str, str]]) -> Tuple[Optional[str], str]:
    def is_poly(row: Dict[str, str]) -> bool:
        gt = str(row.get("geometry_type", "")).lower()
        return "polygon" in gt

    for row in layers:
        name = row["name"].lower()
        if ("crosswalk" in name or "truth" in name) and is_poly(row):
            return row["name"], "name_contains_crosswalk_or_truth"
    for row in layers:
        if is_poly(row):
            return row["name"], "first_polygon_layer"
    return None, "no_polygon_layer"


def _truth_geom(gdf: gpd.GeoDataFrame) -> Polygon:
    geom = unary_union([g for g in gdf.geometry if g is not None and not g.is_empty])
    if geom.is_empty:
        raise RuntimeError("truth_geometry_empty")
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type in {"MultiPolygon", "GeometryCollection"}:
        polys = [g for g in geom.geoms if g.geom_type == "Polygon"]
        if not polys:
            raise RuntimeError("truth_geometry_no_polygon")
        return unary_union(polys)
    raise RuntimeError(f"truth_geometry_invalid:{geom.geom_type}")


def _mask_points_in_polygon(points_xy: np.ndarray, poly: object) -> np.ndarray:
    if points_xy.size == 0:
        return np.zeros((0,), dtype=bool)
    try:
        from shapely import vectorized as shp_vec

        return np.asarray(shp_vec.contains(poly, points_xy[:, 0], points_xy[:, 1]), dtype=bool)
    except Exception:
        pass
    prep_poly = prep(poly)
    if hasattr(prep_poly, "contains_xy"):
        try:
            mask = prep_poly.contains_xy(points_xy[:, 0], points_xy[:, 1])
            return np.asarray(mask, dtype=bool)
        except Exception:
            return np.array([bool(prep_poly.contains_xy(x, y)) for x, y in points_xy], dtype=bool)
    return np.array([bool(prep_poly.contains(Polygon([(x, y), (x + 0.01, y), (x, y + 0.01)]))) for x, y in points_xy], dtype=bool)


def _intensity_stats(inten: np.ndarray) -> Dict[str, float]:
    if inten.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "nonzero_ratio": 0.0,
            "dynamic_range": 0.0,
        }
    vals = inten.astype(np.float64)
    p50 = float(np.percentile(vals, 50))
    p99 = float(np.percentile(vals, 99))
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p50": p50,
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "p99": p99,
        "nonzero_ratio": float(np.mean(vals > 0.0)),
        "dynamic_range": float(p99 - p50),
    }


def _balanced_sample(a: np.ndarray, b: np.ndarray, max_n: int = 200000) -> Tuple[np.ndarray, np.ndarray]:
    if a.size == 0 or b.size == 0:
        return a, b
    n = int(min(a.size, b.size, max_n))
    if n <= 0:
        return a, b
    rng = np.random.default_rng(0)
    if a.size > n:
        a = rng.choice(a, size=n, replace=False)
    if b.size > n:
        b = rng.choice(b, size=n, replace=False)
    return a, b


def _auc_score(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size == 0 or neg.size == 0:
        return 0.5
    scores = np.concatenate([pos, neg]).astype(np.float64)
    labels = np.concatenate([np.ones(pos.size, dtype=np.int8), np.zeros(neg.size, dtype=np.int8)])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            avg = float(np.mean(ranks[order[start:end]]))
            ranks[order[start:end]] = avg
        start = end
    pos_ranks = ranks[labels == 1]
    n_pos = pos.size
    n_neg = neg.size
    auc = (float(np.sum(pos_ranks)) - n_pos * (n_pos + 1) * 0.5) / max(float(n_pos * n_neg), 1.0)
    return float(auc)


def _shift_grid(shift_max: float, step: float) -> List[Tuple[float, float]]:
    vals = np.arange(-float(shift_max), float(shift_max) + 1e-6, float(step))
    return [(float(dx), float(dy)) for dx in vals for dy in vals]


def _grid_ground_mask(points_xyz: np.ndarray, res_m: float, dz: float) -> np.ndarray:
    if points_xyz.size == 0:
        return np.zeros((0,), dtype=bool)
    xy = points_xyz[:, :2].astype(np.float64)
    z = points_xyz[:, 2].astype(np.float64)
    mins = xy.min(axis=0)
    idx = np.floor((xy - mins) / float(res_m)).astype(np.int64)
    lin = idx[:, 0] * 2000003 + idx[:, 1]
    order = np.argsort(lin, kind="mergesort")
    lin_sorted = lin[order]
    z_sorted = z[order]
    unique_mask = np.empty_like(lin_sorted, dtype=bool)
    unique_mask[0] = True
    unique_mask[1:] = lin_sorted[1:] != lin_sorted[:-1]
    group_starts = np.where(unique_mask)[0]
    group_ends = np.append(group_starts[1:], lin_sorted.size)
    ground_z = np.empty((lin_sorted.size,), dtype=np.float64)
    for start, end in zip(group_starts, group_ends):
        p10 = float(np.percentile(z_sorted[start:end], 10))
        ground_z[start:end] = p10
    ground_z_full = np.empty_like(z, dtype=np.float64)
    ground_z_full[order] = ground_z
    return z <= (ground_z_full + float(dz))


def _sample_for_export(points_xyz: np.ndarray, intensity: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray, str]:
    if points_xyz.size == 0:
        return points_xyz, intensity, "empty"
    if points_xyz.shape[0] <= int(max_points):
        return points_xyz, intensity, "none"
    rng = np.random.default_rng(0)
    idx = rng.choice(points_xyz.shape[0], size=int(max_points), replace=False)
    return points_xyz[idx], intensity[idx], f"random_{max_points}"


def _plot_hist_compare(inside: np.ndarray, bg: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    bins = 256
    if inside.size:
        ax.hist(inside.astype(np.float64), bins=bins, range=(0, 65535), alpha=0.6, label="inside", color="orange", density=True)
    if bg.size:
        ax.hist(bg.astype(np.float64), bins=bins, range=(0, 65535), alpha=0.5, label="bg", color="steelblue", density=True)
    ax.set_title("Intensity Histogram: inside vs bg")
    ax.set_xlabel("intensity (uint16)")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_bev_density(points_xy: np.ndarray, out_path: Path, max_points: int = 300000) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if points_xy.size == 0:
        return
    rng = np.random.default_rng(0)
    idx = np.arange(points_xy.shape[0])
    if idx.size > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    pts = points_xy[idx]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c="black", alpha=0.15)
    ax.set_title("BEV Density (ground)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_bev_high(points_xy: np.ndarray, out_path: Path, max_points: int = 200000) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if points_xy.size == 0:
        return
    rng = np.random.default_rng(0)
    idx = np.arange(points_xy.shape[0])
    if idx.size > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    pts = points_xy[idx]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.scatter(pts[:, 0], pts[:, 1], s=0.6, c="red", alpha=0.7)
    ax.set_title("BEV High Intensity (ground)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _resolve_fused_path() -> Path:
    cand = Path("/mnt/data/fused_points_utm32.laz")
    if cand.exists():
        return cand
    if Path(r"runs\lidar_fuse_0010_f280_300_20260129_223419\fused\fused_points_utm32.laz").exists():
        return Path(r"runs\lidar_fuse_0010_f280_300_20260129_223419\fused\fused_points_utm32.laz")
    runs = sorted(Path("runs").glob("lidar_fuse_0010_f280_300_*"), reverse=True)
    for run_dir in runs:
        p = run_dir / "fused" / "fused_points_utm32.laz"
        if p.exists():
            return p
    raise RuntimeError("fused_points_utm32_laz_not_found")


def main() -> int:
    run_id = now_ts()
    run_dir = Path("runs") / f"intensity_extract_0010_f280_300_{run_id}"
    ensure_overwrite(run_dir)
    ensure_dir(run_dir)
    setup_logging(run_dir / "run.log")
    LOG.info("run_start")

    fused_path = _resolve_fused_path()
    input_path = fused_path
    if input_path.suffix.lower() == ".laz":
        las_path = input_path.with_suffix(".las")
        if las_path.exists():
            input_path = las_path
        else:
            write_text(run_dir / "report.md", f"input_laz_missing_las:{input_path}")
            write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "input_laz_missing_las"})
            return 2

    if not TRUTH_GPKG.exists():
        write_text(run_dir / "report.md", f"truth_missing:{TRUTH_GPKG}")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "truth_missing"})
        return 2

    layers = _list_layers(TRUTH_GPKG)
    truth_layer, layer_reason = _pick_truth_layer(layers)
    if truth_layer is None:
        write_text(run_dir / "report.md", "truth_polygon_layer_missing")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "truth_polygon_layer_missing"})
        return 2

    gdf = gpd.read_file(TRUTH_GPKG, layer=truth_layer)
    if gdf.empty:
        write_text(run_dir / "report.md", "truth_empty")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "truth_empty"})
        return 2
    if gdf.crs is None or str(gdf.crs.to_epsg()) != str(TARGET_EPSG):
        gdf = gdf.to_crs(epsg=TARGET_EPSG)
    truth_poly = _truth_geom(gdf)

    gis_dir = ensure_dir(run_dir / "gis")
    write_gpkg_layer(gis_dir / "truth_selected_utm32.gpkg", truth_layer, gdf, TARGET_EPSG, [], overwrite=True)

    roi_poly = truth_poly.buffer(float(ROI_PAD_M))
    bg_inner = truth_poly.buffer(float(BG_INNER_M))
    bg_outer = truth_poly.buffer(float(BG_OUTER_M))
    bg_ring = bg_outer.difference(bg_inner)
    roi_gdf = gpd.GeoDataFrame(
        [{"kind": "roi", "geometry": roi_poly}, {"kind": "bg_ring", "geometry": bg_ring}],
        geometry="geometry",
        crs=f"EPSG:{TARGET_EPSG}",
    )
    write_gpkg_layer(gis_dir / "roi_utm32.gpkg", "roi", roi_gdf, TARGET_EPSG, [], overwrite=True)

    LOG.info("load_points: %s", input_path)
    points_xyz, intensity, meta = _read_las_points(input_path)
    crs_note = "crs_header_not_parsed; assumed EPSG:32632"

    roi_bounds = roi_poly.bounds
    bbox_mask = (
        (points_xyz[:, 0] >= roi_bounds[0])
        & (points_xyz[:, 0] <= roi_bounds[2])
        & (points_xyz[:, 1] >= roi_bounds[1])
        & (points_xyz[:, 1] <= roi_bounds[3])
    )
    points_bbox = points_xyz[bbox_mask]
    intensity_bbox = intensity[bbox_mask]
    roi_mask = _mask_points_in_polygon(points_bbox[:, :2], roi_poly)
    roi_points = points_bbox[roi_mask]
    roi_intensity = intensity_bbox[roi_mask]

    if roi_points.size == 0:
        write_text(run_dir / "report.md", "roi_points_empty")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "roi_points_empty"})
        return 2

    ground_mask = _grid_ground_mask(roi_points, GRID_RES_M, GROUND_BAND_DZ_M)
    ground_points = roi_points[ground_mask]
    ground_intensity = roi_intensity[ground_mask]

    roi_pts_out, roi_int_out, roi_ds = _sample_for_export(roi_points, roi_intensity, MAX_EXPORT_POINTS)
    ground_pts_out, ground_int_out, ground_ds = _sample_for_export(ground_points, ground_intensity, MAX_EXPORT_POINTS)

    pc_dir = ensure_dir(run_dir / "pointcloud")
    write_las(pc_dir / "roi_points_utm32.laz", roi_pts_out, roi_int_out, np.ones((roi_pts_out.shape[0],), dtype=np.uint8), TARGET_EPSG)
    write_las(
        pc_dir / "roi_ground_points_utm32.laz",
        ground_pts_out,
        ground_int_out,
        np.full((ground_pts_out.shape[0],), 2, dtype=np.uint8),
        TARGET_EPSG,
    )

    th_roi = float(np.percentile(ground_intensity.astype(np.float64), HIGH_INT_PCTL_GLOBAL)) if ground_intensity.size else 0.0
    high_roi_mask = ground_intensity >= th_roi
    high_roi_points = ground_points[high_roi_mask]
    high_roi_intensity = ground_intensity[high_roi_mask]
    high_roi_pts_out, high_roi_int_out, high_roi_ds = _sample_for_export(high_roi_points, high_roi_intensity, MAX_EXPORT_POINTS)
    write_las(
        pc_dir / "high_intensity_roi_utm32.laz",
        high_roi_pts_out,
        high_roi_int_out,
        np.full((high_roi_pts_out.shape[0],), 2, dtype=np.uint8),
        TARGET_EPSG,
    )

    inside_mask = _mask_points_in_polygon(ground_points[:, :2], truth_poly)
    inside_points = ground_points[inside_mask]
    inside_intensity = ground_intensity[inside_mask]
    th_inside = float(np.percentile(inside_intensity.astype(np.float64), HIGH_INT_PCTL_INSIDE)) if inside_intensity.size else 0.0
    high_inside_mask = inside_intensity >= th_inside
    high_inside_points = inside_points[high_inside_mask]
    high_inside_intensity = inside_intensity[high_inside_mask]
    hi_in_pts_out, hi_in_int_out, hi_in_ds = _sample_for_export(high_inside_points, high_inside_intensity, MAX_EXPORT_POINTS)
    write_las(
        pc_dir / "high_intensity_truth_utm32.laz",
        hi_in_pts_out,
        hi_in_int_out,
        np.full((hi_in_pts_out.shape[0],), 2, dtype=np.uint8),
        TARGET_EPSG,
    )

    shift_rows: List[Dict[str, object]] = []
    best_dx = 0.0
    best_dy = 0.0
    best_delta = -1e9
    best_auc = 0.5
    if ENABLE_SHIFT_SEARCH:
        LOG.info("shift_search_start")
        shifts = _shift_grid(SHIFT_MAX_M, SHIFT_STEP_M)
        for dx, dy in shifts:
            truth_s = translate(truth_poly, xoff=dx, yoff=dy)
            bg_s = translate(bg_ring, xoff=dx, yoff=dy)
            inside_m = _mask_points_in_polygon(ground_points[:, :2], truth_s)
            bg_m = _mask_points_in_polygon(ground_points[:, :2], bg_s)
            inside = ground_intensity[inside_m]
            bg = ground_intensity[bg_m]
            inside_p95 = float(np.percentile(inside, 95)) if inside.size else 0.0
            bg_p95 = float(np.percentile(bg, 95)) if bg.size else 0.0
            delta_p95 = float(inside_p95 - bg_p95)
            inside_s, bg_samp = _balanced_sample(inside, bg)
            auc = _auc_score(inside_s, bg_samp)
            shift_rows.append(
                {
                    "dx": float(dx),
                    "dy": float(dy),
                    "inside_n": int(inside.size),
                    "bg_n": int(bg.size),
                    "inside_p95": inside_p95,
                    "bg_p95": bg_p95,
                    "delta_p95": delta_p95,
                    "auc": float(auc),
                }
            )
            if delta_p95 > best_delta:
                best_delta = delta_p95
                best_auc = float(auc)
                best_dx = float(dx)
                best_dy = float(dy)
        LOG.info("shift_search_done: best_dx=%.2f best_dy=%.2f", best_dx, best_dy)

    truth_aligned = translate(truth_poly, xoff=best_dx, yoff=best_dy)
    gdf_aligned = gpd.GeoDataFrame([{"geometry": truth_aligned}], geometry="geometry", crs=f"EPSG:{TARGET_EPSG}")
    write_gpkg_layer(gis_dir / "truth_aligned_utm32.gpkg", truth_layer, gdf_aligned, TARGET_EPSG, [], overwrite=True)

    inside_mask_best = _mask_points_in_polygon(ground_points[:, :2], truth_aligned)
    bg_mask_best = _mask_points_in_polygon(ground_points[:, :2], translate(bg_ring, xoff=best_dx, yoff=best_dy))
    inside_best = ground_intensity[inside_mask_best]
    bg_best = ground_intensity[bg_mask_best]
    inside_stats = _intensity_stats(inside_best)
    bg_stats = _intensity_stats(bg_best)
    inside_s, bg_samp = _balanced_sample(inside_best, bg_best)
    auc_final = _auc_score(inside_s, bg_samp)
    delta_p95 = float((inside_stats["p95"] or 0.0) - (bg_stats["p95"] or 0.0))

    tables_dir = ensure_dir(run_dir / "tables")
    write_json(
        tables_dir / "intensity_stats.json",
        {
            "roi_ground": _intensity_stats(ground_intensity),
            "inside_best": inside_stats,
            "bg_best": bg_stats,
            "th_roi": th_roi,
            "th_inside": th_inside,
        },
    )
    write_csv(
        tables_dir / "intensity_inside_vs_bg.csv",
        [
            {
                "region": "inside",
                "count": int(inside_best.size),
                "min": inside_stats["min"],
                "p50": inside_stats["p50"],
                "p90": inside_stats["p90"],
                "p95": inside_stats["p95"],
                "p99": inside_stats["p99"],
                "max": inside_stats["max"],
                "nonzero_ratio": inside_stats["nonzero_ratio"],
            },
            {
                "region": "bg",
                "count": int(bg_best.size),
                "min": bg_stats["min"],
                "p50": bg_stats["p50"],
                "p90": bg_stats["p90"],
                "p95": bg_stats["p95"],
                "p99": bg_stats["p99"],
                "max": bg_stats["max"],
                "nonzero_ratio": bg_stats["nonzero_ratio"],
            },
        ],
        ["region", "count", "min", "p50", "p90", "p95", "p99", "max", "nonzero_ratio"],
    )
    if shift_rows:
        write_csv(
            tables_dir / "shift_search.csv",
            shift_rows,
            ["dx", "dy", "inside_n", "bg_n", "inside_p95", "bg_p95", "delta_p95", "auc"],
        )

    images_dir = ensure_dir(run_dir / "images")
    _plot_hist_compare(inside_best, bg_best, images_dir / "hist_inside_bg.png")
    _plot_bev_density(ground_points[:, :2], images_dir / "bev_density.png")
    _plot_bev_high(high_roi_points[:, :2], images_dir / "bev_high_intensity.png")

    intensity_separable = bool(auc_final >= 0.65 and delta_p95 >= 800.0)
    if intensity_separable:
        conclusion = "usable"
    elif auc_final >= 0.55 and delta_p95 >= 400.0:
        conclusion = "uncertain"
    else:
        conclusion = "unusable"

    decision = {
        "status": "PASS",
        "intensity_separable": intensity_separable,
        "zebra_hint": "needs_visual_check",
        "auc": float(auc_final),
        "delta_p95": float(delta_p95),
        "best_shift": {"dx": best_dx, "dy": best_dy},
        "conclusion": conclusion,
    }
    write_json(run_dir / "decision.json", decision)

    report = [
        "# Intensity Extract 0010 f280-300",
        "",
        f"- input: {input_path}",
        f"- truth_gpkg: {TRUTH_GPKG}",
        f"- truth_layer_selected: {truth_layer}",
        f"- truth_layer_reason: {layer_reason}",
        f"- crs: EPSG:{TARGET_EPSG} ({crs_note})",
        f"- roi_pad_m: {ROI_PAD_M}",
        f"- ground_band_dz_m: {GROUND_BAND_DZ_M}",
        f"- grid_res_m: {GRID_RES_M}",
        f"- high_int_pctl_global: {HIGH_INT_PCTL_GLOBAL}",
        f"- high_int_pctl_inside: {HIGH_INT_PCTL_INSIDE}",
        f"- th_roi: {th_roi:.1f}",
        f"- th_inside: {th_inside:.1f}",
        f"- best_shift: dx={best_dx:.2f}, dy={best_dy:.2f}",
        f"- auc: {auc_final:.4f}",
        f"- delta_p95: {delta_p95:.1f}",
        f"- conclusion: {conclusion}",
        "",
        "## Outputs",
        f"- {relpath(run_dir, gis_dir / 'truth_selected_utm32.gpkg')}",
        f"- {relpath(run_dir, gis_dir / 'truth_aligned_utm32.gpkg')}",
        f"- {relpath(run_dir, gis_dir / 'roi_utm32.gpkg')}",
        f"- {relpath(run_dir, pc_dir / 'roi_points_utm32.laz')} (sample={roi_ds})",
        f"- {relpath(run_dir, pc_dir / 'roi_ground_points_utm32.laz')} (sample={ground_ds})",
        f"- {relpath(run_dir, pc_dir / 'high_intensity_roi_utm32.laz')} (sample={high_roi_ds})",
        f"- {relpath(run_dir, pc_dir / 'high_intensity_truth_utm32.laz')} (sample={hi_in_ds})",
        f"- {relpath(run_dir, tables_dir / 'intensity_stats.json')}",
        f"- {relpath(run_dir, tables_dir / 'intensity_inside_vs_bg.csv')}",
        f"- {relpath(run_dir, tables_dir / 'shift_search.csv')}",
        f"- {relpath(run_dir, images_dir / 'bev_density.png')}",
        f"- {relpath(run_dir, images_dir / 'bev_high_intensity.png')}",
        f"- {relpath(run_dir, images_dir / 'hist_inside_bg.png')}",
    ]
    write_text(run_dir / "report.md", "\n".join(report))
    LOG.info("run_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

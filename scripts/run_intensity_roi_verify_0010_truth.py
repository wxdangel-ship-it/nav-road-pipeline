from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

from pipeline._io import load_yaml
from pipeline.lidar_semantic.export_pointcloud import write_las
from scripts.pipeline_common import (
    ensure_overwrite,
    now_ts,
    relpath,
    setup_logging,
    write_csv,
    write_gpkg_layer,
    write_json,
    write_text,
)


LOG = logging.getLogger("intensity_roi_verify_0010_truth")

TRUTH_GPKG = Path(r"E:\Work\nav-road-pipeline\crosswalk_truth_utm32.gpkg")

REQUIRED_KEYS = [
    "ROI_PAD_M",
    "BG_INNER_M",
    "BG_OUTER_M",
    "SHIFT_MAX_M",
    "SHIFT_STEP_M",
    "GRID_RES_M",
    "INT_STAT",
    "BG_WIN_RADIUS_M",
    "SCORE_NORM_PCTL",
    "TOP_INT_PCTL_ROI",
    "TOP_SCORE_PCTL",
    "MIN_CELL_PTS",
    "MAX_EXPORT_POINTS",
    "TARGET_EPSG",
    "OVERWRITE",
]


@dataclass
class LasMeta:
    point_count: int
    record_length: int
    offset_to_points: int
    scales: Tuple[float, float, float]
    offsets: Tuple[float, float, float]


def _load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return dict(load_yaml(path) or {})


def _normalize_cfg(cfg: Dict[str, object]) -> Dict[str, object]:
    def _norm(v):
        if isinstance(v, dict):
            return {k: _norm(v[k]) for k in sorted(v.keys())}
        if isinstance(v, list):
            return [_norm(x) for x in v]
        return v

    return _norm(cfg)


def _hash_cfg(cfg: Dict[str, object]) -> str:
    raw = json.dumps(_normalize_cfg(cfg), sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _write_resolved(run_dir: Path, cfg: Dict[str, object]) -> str:
    import yaml

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8"
    )
    params_hash = _hash_cfg(cfg)
    (run_dir / "params_hash.txt").write_text(params_hash + "\n", encoding="utf-8")
    return params_hash


def _read_las_header(path: Path) -> LasMeta:
    import struct

    header_size = 227
    with path.open("rb") as f:
        header = f.read(header_size)
    if len(header) < header_size:
        raise RuntimeError("las_header_short")
    fmt = "<4sHH16sBB32s32sHHH I I B H I 5I ddd ddd dddddd"
    values = struct.unpack(fmt, header[: struct.calcsize(fmt)])
    sig = values[0]
    if sig != b"LASF":
        raise RuntimeError("invalid_las_signature")
    offset_to_points = int(values[11])
    record_len = int(values[14])
    point_count = int(values[15])
    scales = (float(values[21]), float(values[22]), float(values[23]))
    offsets = (float(values[24]), float(values[25]), float(values[26]))
    return LasMeta(
        point_count=point_count,
        record_length=record_len,
        offset_to_points=offset_to_points,
        scales=scales,
        offsets=offsets,
    )


def _read_las_points(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    meta = _read_las_header(path)
    if meta.record_length < 20:
        raise RuntimeError("unsupported_point_record_length")
    with path.open("rb") as f:
        f.seek(meta.offset_to_points)
        raw = f.read(meta.point_count * meta.record_length)
    pad = meta.record_length - 20
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
    return points, intensity


def _latest_ground_clean() -> Path:
    runs = sorted(Path("runs").glob("lidar_ground_0010_f250_500_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        cand = run / "pointcloud" / "ground_points_clean_utm32.laz"
        if cand.exists():
            return cand
    raise RuntimeError("ground_points_clean_not_found")


def _list_layers(path: Path) -> List[Dict[str, str]]:
    try:
        import pyogrio

        layers_raw = pyogrio.list_layers(str(path))
    except Exception as exc:
        raise RuntimeError(f"list_layers_failed:{exc}") from exc
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


def _pick_truth_layer(layers: List[Dict[str, str]]) -> Optional[str]:
    def is_poly(row: Dict[str, str]) -> bool:
        gt = str(row.get("geometry_type", "")).lower()
        return "polygon" in gt

    for row in layers:
        name = row["name"].lower()
        if ("crosswalk" in name or "truth" in name) and is_poly(row):
            return row["name"]
    for row in layers:
        if is_poly(row):
            return row["name"]
    return None


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
    p90 = float(np.percentile(vals, 90))
    p95 = float(np.percentile(vals, 95))
    p99 = float(np.percentile(vals, 99))
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "nonzero_ratio": float(np.mean(vals > 0)),
        "dynamic_range": float(np.max(vals) - np.min(vals)),
    }


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    if scores.size == 0 or labels.size == 0:
        return 0.0
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(scores.size) + 1
    pos = labels == 1
    n_pos = int(np.sum(pos))
    n_neg = int(scores.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    sum_ranks = float(np.sum(ranks[pos]))
    auc = (sum_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _sample_for_export(
    points_xyz: np.ndarray, intensity: np.ndarray, max_points: int
) -> Tuple[np.ndarray, np.ndarray, str]:
    if points_xyz.size == 0:
        return points_xyz, intensity, "empty"
    if points_xyz.shape[0] <= max_points:
        return points_xyz, intensity, "none"
    rng = np.random.default_rng(0)
    idx = rng.choice(points_xyz.shape[0], size=int(max_points), replace=False)
    return points_xyz[idx], intensity[idx], f"random_{max_points}"


def _grid_spec(points_xy: np.ndarray, res_m: float) -> Tuple[float, float, float, int, int, rasterio.Affine]:
    minx = float(np.min(points_xy[:, 0]))
    miny = float(np.min(points_xy[:, 1]))
    maxx = float(np.max(points_xy[:, 0]))
    maxy = float(np.max(points_xy[:, 1]))
    width = int(np.ceil((maxx - minx) / res_m)) + 1
    height = int(np.ceil((maxy - miny) / res_m)) + 1
    maxy_aligned = miny + height * res_m
    transform = from_origin(minx, maxy_aligned, res_m, res_m)
    return minx, miny, maxy_aligned, width, height, transform


def _group_slices(lin_idx: np.ndarray):
    order = np.argsort(lin_idx, kind="mergesort")
    lin_sorted = lin_idx[order]
    uniq, start, counts = np.unique(lin_sorted, return_index=True, return_counts=True)
    return order, uniq, start, counts


def _box_mean(values: np.ndarray, valid_mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return np.where(valid_mask, values, np.nan)
    vals = np.where(valid_mask, values, 0.0).astype(np.float64)
    counts = valid_mask.astype(np.int64)
    pad = radius_px
    vals_p = np.pad(vals, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
    cnts_p = np.pad(counts, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    sat = np.cumsum(np.cumsum(vals_p, axis=0), axis=1)
    sat_c = np.cumsum(np.cumsum(cnts_p, axis=0), axis=1)
    sat = np.pad(sat, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    sat_c = np.pad(sat_c, ((1, 0), (1, 0)), mode="constant", constant_values=0)
    h, w = values.shape
    y0 = np.arange(0, h)
    x0 = np.arange(0, w)
    y1 = y0 + 2 * pad
    x1 = x0 + 2 * pad
    sum_block = (
        sat[y1[:, None] + 1, x1[None, :] + 1]
        - sat[y0[:, None], x1[None, :] + 1]
        - sat[y1[:, None] + 1, x0[None, :]]
        + sat[y0[:, None], x0[None, :]]
    )
    cnt_block = (
        sat_c[y1[:, None] + 1, x1[None, :] + 1]
        - sat_c[y0[:, None], x1[None, :] + 1]
        - sat_c[y1[:, None] + 1, x0[None, :]]
        + sat_c[y0[:, None], x0[None, :]]
    )
    out = np.full((h, w), np.nan, dtype=np.float32)
    valid = cnt_block > 0
    out[valid] = (sum_block[valid] / cnt_block[valid]).astype(np.float32)
    return out


def _write_raster(path: Path, arr: np.ndarray, transform: rasterio.Affine, epsg: int, nodata: Optional[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = arr.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=str(arr.dtype),
        crs=f"EPSG:{epsg}",
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as ds:
        ds.write(arr, 1)


def _plot_gray(arr: np.ndarray, out_path: Path, title: str, vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(arr, origin="upper", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_mask_overlay(score: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(score, origin="upper", cmap="gray", vmin=0.0, vmax=1.0)
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = mask.astype(np.float32) * 0.5
    ax.imshow(overlay, origin="upper")
    ax.set_title("score mask overlay")
    ax.set_axis_off()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_histogram(inside: np.ndarray, bg: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(inside, bins=64, range=(0, 65535), alpha=0.6, label="inside", color="tab:red")
    ax.hist(bg, bins=64, range=(0, 65535), alpha=0.6, label="bg", color="tab:blue")
    ax.set_xlabel("intensity (uint16)")
    ax.set_ylabel("count")
    ax.set_title("inside vs bg intensity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _shift_grid(max_m: float, step_m: float) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.arange(-max_m, max_m + 1e-6, step_m)
    dx, dy = np.meshgrid(vals, vals)
    return dx, dy


def main() -> None:
    cfg_path = Path("configs/intensity_roi_verify_0010_truth.yaml")
    cfg = _load_yaml(cfg_path)
    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise KeyError(f"Missing required key {key} in {cfg_path}")

    run_id = now_ts()
    run_dir = Path("runs") / f"intensity_roi_verify_0010_truth_{run_id}"
    if bool(cfg.get("OVERWRITE", True)):
        ensure_overwrite(run_dir)
    setup_logging(run_dir / "run.log")
    _write_resolved(run_dir, cfg)

    warnings: List[str] = []
    gis_dir = run_dir / "gis"
    pc_dir = run_dir / "pointcloud"
    ras_dir = run_dir / "rasters"
    tbl_dir = run_dir / "tables"
    img_dir = run_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    layers = _list_layers(TRUTH_GPKG)
    write_csv(tbl_dir / "truth_layers.csv", layers, ["name", "geometry_type"])
    layer_name = _pick_truth_layer(layers)
    if layer_name is None:
        report = [
            "# ROI intensity verify 0010 truth",
            "",
            "- status: FAIL",
            "- reason: no_polygon_layer_in_truth_gpkg",
            f"- truth: {TRUTH_GPKG}",
            f"- layers_csv: {relpath(run_dir, tbl_dir / 'truth_layers.csv')}",
        ]
        write_text(run_dir / "report.md", "\n".join(report) + "\n")
        return

    truth_gdf = gpd.read_file(TRUTH_GPKG, layer=layer_name)
    if truth_gdf.crs is None:
        truth_gdf = truth_gdf.set_crs(int(cfg["TARGET_EPSG"]))
    truth_geom = _truth_geom(truth_gdf)
    truth_gdf = gpd.GeoDataFrame(geometry=[truth_geom], crs=truth_gdf.crs)
    write_gpkg_layer(gis_dir / "truth_selected_utm32.gpkg", "truth_selected", truth_gdf, int(cfg["TARGET_EPSG"]), warnings)

    input_las = _latest_ground_clean()
    points_xyz, intensity = _read_las_points(input_las)
    finite_mask = np.isfinite(points_xyz).all(axis=1)
    points_xyz = points_xyz[finite_mask]
    intensity = intensity[finite_mask]
    if points_xyz.size == 0:
        report = [
            "# ROI intensity verify 0010 truth",
            "",
            "- status: FAIL",
            "- reason: no_valid_points",
            f"- input: {input_las}",
        ]
        write_text(run_dir / "report.md", "\n".join(report) + "\n")
        return

    roi = truth_geom.buffer(float(cfg["ROI_PAD_M"]))
    bg_ring = truth_geom.buffer(float(cfg["BG_OUTER_M"])).difference(truth_geom.buffer(float(cfg["BG_INNER_M"])))
    roi_gdf = gpd.GeoDataFrame(geometry=[roi], crs=truth_gdf.crs)
    bg_gdf = gpd.GeoDataFrame(geometry=[bg_ring], crs=truth_gdf.crs)
    write_gpkg_layer(gis_dir / "roi_utm32.gpkg", "roi", roi_gdf, int(cfg["TARGET_EPSG"]), warnings)
    write_gpkg_layer(gis_dir / "bg_ring_utm32.gpkg", "bg_ring", bg_gdf, int(cfg["TARGET_EPSG"]), warnings)

    points_xy = points_xyz[:, :2]
    roi_mask = _mask_points_in_polygon(points_xy, roi)
    roi_points = points_xyz[roi_mask]
    roi_intensity = intensity[roi_mask]

    max_export = int(cfg["MAX_EXPORT_POINTS"])
    roi_pts_out, roi_int_out, roi_ds = _sample_for_export(roi_points, roi_intensity, max_export)
    write_las(
        pc_dir / "roi_ground_clean_utm32.laz",
        roi_pts_out,
        roi_int_out,
        np.ones((roi_pts_out.shape[0],), dtype=np.uint8),
        int(cfg["TARGET_EPSG"]),
    )

    if roi_points.size == 0:
        report = [
            "# ROI intensity verify 0010 truth",
            "",
            "- status: FAIL",
            "- reason: roi_has_no_points",
            f"- input: {input_las}",
        ]
        write_text(run_dir / "report.md", "\n".join(report) + "\n")
        return

    dx_grid, dy_grid = _shift_grid(float(cfg["SHIFT_MAX_M"]), float(cfg["SHIFT_STEP_M"]))
    shift_rows: List[Dict[str, object]] = []
    best = None
    for i in range(dx_grid.shape[0]):
        for j in range(dx_grid.shape[1]):
            dx = float(dx_grid[i, j])
            dy = float(dy_grid[i, j])
            truth_shift = translate(truth_geom, xoff=dx, yoff=dy)
            bg_shift = translate(bg_ring, xoff=dx, yoff=dy)
            inside_mask = _mask_points_in_polygon(roi_points[:, :2], truth_shift)
            bg_mask = _mask_points_in_polygon(roi_points[:, :2], bg_shift)
            inside_int = roi_intensity[inside_mask]
            bg_int = roi_intensity[bg_mask]
            if inside_int.size == 0 or bg_int.size == 0:
                auc = 0.0
                delta_p95 = 0.0
            else:
                labels = np.concatenate([np.ones_like(inside_int), np.zeros_like(bg_int)])
                scores = np.concatenate([inside_int, bg_int]).astype(np.float64)
                auc = _roc_auc(scores, labels)
                delta_p95 = float(np.percentile(inside_int, 95) - np.percentile(bg_int, 95))
            row = {
                "dx": dx,
                "dy": dy,
                "auc": float(auc),
                "delta_p95": float(delta_p95),
                "inside_n": int(inside_int.size),
                "bg_n": int(bg_int.size),
            }
            shift_rows.append(row)
            key = (float(auc), float(delta_p95))
            if best is None or key > best[0]:
                best = (key, row)

    write_csv(tbl_dir / "shift_search.csv", shift_rows, ["dx", "dy", "auc", "delta_p95", "inside_n", "bg_n"])
    best_row = best[1] if best else {"dx": 0.0, "dy": 0.0, "auc": 0.0, "delta_p95": 0.0}
    best_dx = float(best_row["dx"])
    best_dy = float(best_row["dy"])

    truth_aligned = translate(truth_geom, xoff=best_dx, yoff=best_dy)
    truth_aligned_gdf = gpd.GeoDataFrame(geometry=[truth_aligned], crs=truth_gdf.crs)
    write_gpkg_layer(gis_dir / "truth_aligned_utm32.gpkg", "truth_aligned", truth_aligned_gdf, int(cfg["TARGET_EPSG"]), warnings)

    inside_mask = _mask_points_in_polygon(roi_points[:, :2], truth_aligned)
    bg_mask = _mask_points_in_polygon(roi_points[:, :2], translate(bg_ring, xoff=best_dx, yoff=best_dy))
    inside_points = roi_points[inside_mask]
    inside_int = roi_intensity[inside_mask]
    bg_points = roi_points[bg_mask]
    bg_int = roi_intensity[bg_mask]

    write_las(
        pc_dir / "inside_ground_utm32.laz",
        inside_points,
        inside_int,
        np.ones((inside_points.shape[0],), dtype=np.uint8),
        int(cfg["TARGET_EPSG"]),
    )
    write_las(
        pc_dir / "bg_ground_utm32.laz",
        bg_points,
        bg_int,
        np.ones((bg_points.shape[0],), dtype=np.uint8),
        int(cfg["TARGET_EPSG"]),
    )

    inside_stats = _intensity_stats(inside_int)
    bg_stats = _intensity_stats(bg_int)
    inside_bg_payload = {
        "best_shift": {"dx": best_dx, "dy": best_dy},
        "auc": float(best_row.get("auc", 0.0)),
        "delta_p95": float(best_row.get("delta_p95", 0.0)),
        "inside_stats": inside_stats,
        "bg_stats": bg_stats,
    }
    write_json(tbl_dir / "inside_bg_stats.json", inside_bg_payload)
    if inside_int.size and bg_int.size:
        _plot_histogram(inside_int, bg_int, img_dir / "hist_inside_bg.png")

    res_m = float(cfg["GRID_RES_M"])
    minx, miny, maxy_aligned, width, height, transform = _grid_spec(roi_points[:, :2], res_m)
    x = roi_points[:, 0]
    y = roi_points[:, 1]
    col_idx = np.floor((x - minx) / res_m).astype(np.int64)
    row_idx = np.floor((maxy_aligned - y) / res_m).astype(np.int64)
    valid = (col_idx >= 0) & (row_idx >= 0) & (col_idx < width) & (row_idx < height)
    col_v = col_idx[valid]
    row_v = row_idx[valid]
    int_v = roi_intensity[valid].astype(np.float64)
    lin = row_v * int(width) + col_v
    order, uniq, start, counts = _group_slices(lin)
    col_s = col_v[order]
    row_s = row_v[order]
    int_s = int_v[order]
    min_cell = int(cfg["MIN_CELL_PTS"])
    pctl = 95 if str(cfg["INT_STAT"]).lower() == "p95" else 95
    p95 = np.full((height, width), np.nan, dtype=np.float32)
    count = np.zeros((height, width), dtype=np.int32)
    for u, s, c in zip(uniq, start, counts):
        if c < min_cell:
            continue
        cx = int(col_s[s])
        cy = int(row_s[s])
        sl = slice(int(s), int(s + c))
        p95[cy, cx] = float(np.percentile(int_s[sl], pctl))
        count[cy, cx] = int(c)

    valid_mask = (count >= min_cell) & np.isfinite(p95)
    radius_px = int(np.ceil(float(cfg["BG_WIN_RADIUS_M"]) / res_m))
    bg = _box_mean(p95, valid_mask, radius_px)
    score = np.where(valid_mask & np.isfinite(bg), np.maximum(0.0, p95 - bg), np.nan).astype(np.float32)
    score_valid = score[np.isfinite(score)]
    if score_valid.size == 0:
        score_norm = np.zeros_like(score, dtype=np.float32)
    else:
        denom = float(np.percentile(score_valid, float(cfg["SCORE_NORM_PCTL"])))
        denom = max(denom, 1e-6)
        score_norm = np.clip(score / denom, 0.0, 1.0).astype(np.float32)
    score_valid_norm = score_norm[np.isfinite(score_norm)]
    if score_valid_norm.size == 0:
        top_thr = 1.1
    else:
        top_thr = float(np.percentile(score_valid_norm, float(cfg["TOP_SCORE_PCTL"])))
    score_mask = (score_norm >= top_thr) & np.isfinite(score_norm)

    _write_raster(ras_dir / "roi_intensity_p95_utm32.tif", p95.astype(np.float32), transform, int(cfg["TARGET_EPSG"]), np.nan)
    _write_raster(ras_dir / "roi_intensity_bg_utm32.tif", bg.astype(np.float32), transform, int(cfg["TARGET_EPSG"]), np.nan)
    _write_raster(ras_dir / "roi_intensity_score_utm32.tif", score_norm.astype(np.float32), transform, int(cfg["TARGET_EPSG"]), np.nan)
    _write_raster(ras_dir / "roi_score_mask_top_utm32.tif", score_mask.astype(np.uint8), transform, int(cfg["TARGET_EPSG"]), 0)

    img_dir.mkdir(parents=True, exist_ok=True)
    p95_valid = p95[np.isfinite(p95)]
    if p95_valid.size:
        vmin, vmax = float(np.min(p95_valid)), float(np.max(p95_valid))
    else:
        vmin, vmax = None, None
    _plot_gray(p95, img_dir / "roi_intensity_p95.png", "ROI intensity p95", vmin=vmin, vmax=vmax)
    _plot_gray(score_norm, img_dir / "roi_score.png", "ROI score (normalized)", vmin=0.0, vmax=1.0)
    _plot_mask_overlay(score_norm, score_mask, img_dir / "roi_score_mask_overlay.png")

    dx_vals = sorted(set([row["dx"] for row in shift_rows]))
    dy_vals = sorted(set([row["dy"] for row in shift_rows]))
    dx_index = {v: i for i, v in enumerate(dx_vals)}
    dy_index = {v: i for i, v in enumerate(dy_vals)}
    heat = np.full((len(dy_vals), len(dx_vals)), np.nan, dtype=np.float32)
    for row in shift_rows:
        heat[dy_index[row["dy"]], dx_index[row["dx"]]] = float(row["auc"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heat, origin="lower", cmap="viridis")
    ax.set_xticks(range(len(dx_vals)))
    ax.set_yticks(range(len(dy_vals)))
    ax.set_xticklabels([f"{v:.1f}" for v in dx_vals], rotation=45, ha="right")
    ax.set_yticklabels([f"{v:.1f}" for v in dy_vals])
    ax.set_xlabel("dx (m)")
    ax.set_ylabel("dy (m)")
    ax.set_title("shift AUC heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(img_dir / "shift_heatmap.png")
    plt.close(fig)

    top_int_thr = float(np.percentile(roi_intensity.astype(np.float64), float(cfg["TOP_INT_PCTL_ROI"])))
    high_int_mask = roi_intensity >= top_int_thr
    high_int_pts = roi_points[high_int_mask]
    high_intensity = roi_intensity[high_int_mask]
    high_int_pts_out, high_int_out, hi_ds = _sample_for_export(high_int_pts, high_intensity, max_export)
    write_las(
        pc_dir / "roi_high_intensity_utm32.laz",
        high_int_pts_out,
        high_int_out,
        np.ones((high_int_pts_out.shape[0],), dtype=np.uint8),
        int(cfg["TARGET_EPSG"]),
    )

    rows_all = row_idx
    cols_all = col_idx
    valid_all = (cols_all >= 0) & (rows_all >= 0) & (cols_all < width) & (rows_all < height)
    mask_sel = np.zeros((roi_points.shape[0],), dtype=bool)
    idx = np.where(valid_all)[0]
    mask_sel[idx] = score_mask[rows_all[valid_all], cols_all[valid_all]]
    high_score_points = roi_points[mask_sel]
    high_score_intensity = roi_intensity[mask_sel]
    high_score_pts_out, high_score_int_out, hs_ds = _sample_for_export(high_score_points, high_score_intensity, max_export)
    write_las(
        pc_dir / "roi_high_score_utm32.laz",
        high_score_pts_out,
        high_score_int_out,
        np.ones((high_score_pts_out.shape[0],), dtype=np.uint8),
        int(cfg["TARGET_EPSG"]),
    )

    top_mask_area_ratio = float(np.mean(score_mask[valid_mask])) if np.any(valid_mask) else 0.0
    intensity_separable = bool(best_row.get("auc", 0.0) >= 0.65 and best_row.get("delta_p95", 0.0) >= 800.0)
    score_visible = bool(0.002 <= top_mask_area_ratio <= 0.03)
    decision = {
        "status": "PASS" if intensity_separable else "WARN",
        "best_shift": {"dx": best_dx, "dy": best_dy},
        "auc": float(best_row.get("auc", 0.0)),
        "delta_p95": float(best_row.get("delta_p95", 0.0)),
        "intensity_separable": intensity_separable,
        "score_visible": score_visible,
        "top_mask_area_ratio": top_mask_area_ratio,
        "roi_points": int(roi_points.shape[0]),
        "inside_points": int(inside_points.shape[0]),
        "bg_points": int(bg_points.shape[0]),
        "export_sampling": {
            "roi_ground": roi_ds,
            "roi_high_intensity": hi_ds,
            "roi_high_score": hs_ds,
        },
    }
    write_json(run_dir / "decision.json", decision)

    report = [
        "# ROI intensity verify 0010 truth (clean ground)",
        "",
        f"- input_ground: {input_las}",
        f"- truth_layer: {layer_name}",
        f"- best_shift_dx: {best_dx:.2f}",
        f"- best_shift_dy: {best_dy:.2f}",
        f"- auc: {decision['auc']:.4f}",
        f"- delta_p95: {decision['delta_p95']:.1f}",
        f"- intensity_separable: {intensity_separable}",
        f"- score_visible: {score_visible}",
        f"- top_mask_area_ratio: {top_mask_area_ratio:.4f}",
        f"- roi_export_sampling: {roi_ds}",
        f"- high_int_export_sampling: {hi_ds}",
        f"- high_score_export_sampling: {hs_ds}",
        "",
        "## outputs",
        f"- {relpath(run_dir, gis_dir / 'truth_selected_utm32.gpkg')}",
        f"- {relpath(run_dir, gis_dir / 'truth_aligned_utm32.gpkg')}",
        f"- {relpath(run_dir, gis_dir / 'roi_utm32.gpkg')}",
        f"- {relpath(run_dir, gis_dir / 'bg_ring_utm32.gpkg')}",
        f"- {relpath(run_dir, pc_dir / 'roi_ground_clean_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'roi_high_intensity_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'roi_high_score_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'inside_ground_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'bg_ground_utm32.laz')}",
        f"- {relpath(run_dir, ras_dir / 'roi_intensity_p95_utm32.tif')}",
        f"- {relpath(run_dir, ras_dir / 'roi_intensity_bg_utm32.tif')}",
        f"- {relpath(run_dir, ras_dir / 'roi_intensity_score_utm32.tif')}",
        f"- {relpath(run_dir, ras_dir / 'roi_score_mask_top_utm32.tif')}",
        f"- {relpath(run_dir, tbl_dir / 'shift_search.csv')}",
        f"- {relpath(run_dir, tbl_dir / 'inside_bg_stats.json')}",
        f"- {relpath(run_dir, img_dir / 'roi_intensity_p95.png')}",
        f"- {relpath(run_dir, img_dir / 'roi_score.png')}",
        f"- {relpath(run_dir, img_dir / 'roi_score_mask_overlay.png')}",
        f"- {relpath(run_dir, img_dir / 'hist_inside_bg.png')}",
        f"- {relpath(run_dir, img_dir / 'shift_heatmap.png')}",
        "",
        "## conclusion",
        "- 建议优先查看 roi_high_score_utm32.laz（正射）与 roi_high_intensity_utm32.laz（倾斜）以判断标线条纹结构。",
    ]
    if warnings:
        report.append("")
        report.append("## warnings")
        report.extend([f"- {w}" for w in warnings])
    write_text(run_dir / "report.md", "\n".join(report) + "\n")


if __name__ == "__main__":
    main()

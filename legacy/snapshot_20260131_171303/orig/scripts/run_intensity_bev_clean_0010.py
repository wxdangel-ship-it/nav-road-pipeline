from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_origin

from pipeline._io import load_yaml
from pipeline.lidar_semantic.export_pointcloud import write_las
from scripts.pipeline_common import ensure_overwrite, now_ts, relpath, setup_logging, write_json, write_text


LOG = logging.getLogger("intensity_bev_clean_0010")

REQUIRED_KEYS = [
    "GRID_RES_M",
    "STAT",
    "MIN_CELL_PTS",
    "BG_WIN_RADIUS_M",
    "SCORE_NORM_PCTL",
    "TOP_SCORE_PCTL",
    "OUTPUT_EPSG",
    "OVERWRITE",
]


def _resolve_input_laz(cfg: Dict[str, object]) -> Path:
    cfg_path = str(cfg.get("INPUT_LAZ") or "").strip()
    if cfg_path:
        return Path(cfg_path)
    runs = sorted(Path("runs").glob("lidar_ground_0010_f250_500_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    best = None
    best_score = None
    for run in runs:
        decision = run / "decision.json"
        if not decision.exists():
            continue
        try:
            payload = json.loads(decision.read_text(encoding="utf-8"))
        except Exception:
            continue
        score = payload.get("dtm_std_p90_clean")
        if score is None:
            continue
        cand = run / "pointcloud" / "ground_points_clean_utm32.laz"
        if not cand.exists():
            cand = run / "pointcloud" / "ground_points_utm32.laz"
        if not cand.exists():
            continue
        if best_score is None or float(score) < float(best_score):
            best_score = float(score)
            best = cand
    if best is not None:
        return best
    for run in runs:
        cand = run / "pointcloud" / "ground_points_clean_utm32.laz"
        if cand.exists():
            return cand
        cand = run / "pointcloud" / "ground_points_utm32.laz"
        if cand.exists():
            return cand
    raise RuntimeError("no_ground_points_clean_found")


def _resolve_existing_las(path: Path) -> Path:
    if path.exists():
        return path
    if path.suffix.lower() == ".laz":
        alt = path.with_suffix(".las")
        if alt.exists():
            return alt
    raise RuntimeError(f"input_las_missing:{path}")


def _read_las_points(path: Path) -> Tuple[np.ndarray, np.ndarray]:
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
        point_record_length = int(values[14])
        point_count = int(values[15])
        sx, sy, sz = float(values[21]), float(values[22]), float(values[23])
        ox, oy, oz = float(values[24]), float(values[25]), float(values[26])
        if point_record_length < 20:
            raise RuntimeError("unsupported_point_record_length")
        f.seek(offset_to_points)
        raw = f.read(point_count * point_record_length)
    pad = point_record_length - 20
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
    arr = np.frombuffer(raw, dtype=dtype, count=point_count)
    x = arr["x"].astype(np.float64) * sx + ox
    y = arr["y"].astype(np.float64) * sy + oy
    z = arr["z"].astype(np.float64) * sz + oz
    intensity = arr["intensity"].astype(np.uint16)
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    return points, intensity


def _grid_spec(points_xyz: np.ndarray, res_m: float) -> Tuple[float, float, float, int, int, rasterio.Affine]:
    minx = float(np.min(points_xyz[:, 0]))
    miny = float(np.min(points_xyz[:, 1]))
    maxx = float(np.max(points_xyz[:, 0]))
    maxy = float(np.max(points_xyz[:, 1]))
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


def _score_stats(values: np.ndarray) -> Dict[str, float]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "nonzero_ratio": 0.0,
            "valid_ratio": 0.0,
        }
    return {
        "p50": float(np.percentile(valid, 50)),
        "p90": float(np.percentile(valid, 90)),
        "p95": float(np.percentile(valid, 95)),
        "p99": float(np.percentile(valid, 99)),
        "nonzero_ratio": float(np.mean(valid > 0)),
        "valid_ratio": float(np.mean(np.isfinite(values))),
    }


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


def main() -> None:
    cfg_path = Path("configs/lidar_intensity_bev_clean_0010.yaml")
    cfg = load_yaml(cfg_path)
    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise KeyError(f"Missing required key {key} in {cfg_path}")

    run_id = now_ts()
    run_dir = Path("runs") / f"intensity_bev_clean_0010_{run_id}"
    if bool(cfg.get("OVERWRITE", True)):
        ensure_overwrite(run_dir)

    setup_logging(run_dir / "run.log")
    LOG.info("run_dir=%s", run_dir)

    input_path = _resolve_input_laz(cfg)
    input_path = _resolve_existing_las(input_path)
    LOG.info("input_las=%s", input_path)

    points_xyz, intensity = _read_las_points(input_path)
    finite_mask = np.isfinite(points_xyz).all(axis=1)
    dropped = int(np.sum(~finite_mask))
    if dropped:
        points_xyz = points_xyz[finite_mask]
        intensity = intensity[finite_mask]
    if points_xyz.size == 0:
        report = [
            "# Clean ground intensity BEV (0010 f250-500)",
            "",
            "- status: FAIL",
            "- reason: no_valid_points",
            f"- input: {input_path}",
            f"- dropped_nonfinite: {dropped}",
        ]
        write_text(run_dir / "report.md", "\n".join(report) + "\n")
        return
    intensity_stats = _intensity_stats(intensity)
    tables_dir = run_dir / "tables"
    write_json(tables_dir / "intensity_stats.json", intensity_stats)

    if intensity_stats["nonzero_ratio"] <= 0.0:
        report = [
            "# Clean ground intensity BEV (0010 f250-500)",
            "",
            "- status: FAIL",
            "- reason: intensity_all_zero",
            f"- input: {input_path}",
            f"- intensity_nonzero_ratio: {intensity_stats['nonzero_ratio']:.6f}",
        ]
        write_text(run_dir / "report.md", "\n".join(report) + "\n")
        return

    res_m = float(cfg["GRID_RES_M"])
    minx, miny, maxy, width, height, transform = _grid_spec(points_xyz, res_m)
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    col = np.floor((x - minx) / res_m).astype(np.int64)
    row = np.floor((maxy - y) / res_m).astype(np.int64)
    valid = (col >= 0) & (row >= 0) & (col < width) & (row < height)
    col_v = col[valid]
    row_v = row[valid]
    int_v = intensity[valid].astype(np.float64)
    lin = row_v * int(width) + col_v
    order, uniq, start, counts = _group_slices(lin)
    col_s = col_v[order]
    row_s = row_v[order]
    int_s = int_v[order]

    pctl = 95 if str(cfg.get("STAT", "p95")).lower() == "p95" else 95
    min_cell = int(cfg["MIN_CELL_PTS"])
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
        top_thr = 1.1
    else:
        norm_pctl = float(cfg["SCORE_NORM_PCTL"])
        denom = float(np.percentile(score_valid, norm_pctl))
        denom = max(denom, 1e-6)
        score_norm = np.clip(score / denom, 0.0, 1.0).astype(np.float32)
        score_norm_valid = score_norm[np.isfinite(score_norm)]
        top_thr = float(np.percentile(score_norm_valid, float(cfg["TOP_SCORE_PCTL"])))

    score_mask = (score_norm >= top_thr) & np.isfinite(score_norm)

    rasters_dir = run_dir / "rasters"
    _write_raster(rasters_dir / "intensity_p95_utm32.tif", p95.astype(np.float32), transform, int(cfg["OUTPUT_EPSG"]), np.nan)
    _write_raster(rasters_dir / "intensity_bg_utm32.tif", bg.astype(np.float32), transform, int(cfg["OUTPUT_EPSG"]), np.nan)
    _write_raster(
        rasters_dir / "intensity_score_tophat_utm32.tif",
        score_norm.astype(np.float32),
        transform,
        int(cfg["OUTPUT_EPSG"]),
        np.nan,
    )
    _write_raster(
        rasters_dir / "score_mask_top_utm32.tif",
        score_mask.astype(np.uint8),
        transform,
        int(cfg["OUTPUT_EPSG"]),
        0,
    )

    rows_all = row
    cols_all = col
    valid_all = (cols_all >= 0) & (rows_all >= 0) & (cols_all < width) & (rows_all < height)
    mask_sel = np.zeros((points_xyz.shape[0],), dtype=bool)
    idx = np.where(valid_all)[0]
    mask_sel[idx] = score_mask[rows_all[valid_all], cols_all[valid_all]]
    high_points = points_xyz[mask_sel]
    high_intensity = intensity[mask_sel]
    if high_points.size:
        hi_finite = np.isfinite(high_points).all(axis=1)
        if not np.all(hi_finite):
            high_points = high_points[hi_finite]
            high_intensity = high_intensity[hi_finite]
    dropped_overflow = 0
    if high_points.size:
        mins = np.min(high_points, axis=0)
        max_allowed = mins + 0.001 * 2147483647.0
        in_range = (high_points[:, 0] <= max_allowed[0]) & (high_points[:, 1] <= max_allowed[1]) & (high_points[:, 2] <= max_allowed[2])
        if not np.all(in_range):
            dropped_overflow = int(np.sum(~in_range))
            high_points = high_points[in_range]
            high_intensity = high_intensity[in_range]
    pc_dir = run_dir / "pointcloud"
    write_las(
        pc_dir / "high_score_points_utm32.laz",
        high_points,
        high_intensity,
        np.ones((high_points.shape[0],), dtype=np.uint8),
        int(cfg["OUTPUT_EPSG"]),
    )

    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    p95_valid = p95[np.isfinite(p95)]
    if p95_valid.size:
        vmin, vmax = float(np.min(p95_valid)), float(np.max(p95_valid))
    else:
        vmin, vmax = None, None
    _plot_gray(p95, images_dir / "intensity_p95_preview.png", "intensity p95", vmin=vmin, vmax=vmax)
    _plot_gray(score_norm, images_dir / "score_preview.png", "score (normalized)", vmin=0.0, vmax=1.0)
    _plot_mask_overlay(score_norm, score_mask, images_dir / "score_mask_overlay.png")

    score_stats = _score_stats(score_norm)
    score_stats["top_score_threshold"] = float(top_thr if np.isfinite(top_thr) else 0.0)
    score_stats["score_norm_pctl"] = float(cfg["SCORE_NORM_PCTL"])
    score_stats["top_score_pctl"] = float(cfg["TOP_SCORE_PCTL"])
    write_json(tables_dir / "score_stats.json", score_stats)

    report = [
        "# Clean ground intensity BEV (0010 f250-500)",
        "",
        f"- input: {input_path}",
        f"- dropped_nonfinite: {dropped}",
        f"- grid_res_m: {res_m}",
        f"- min_cell_pts: {min_cell}",
        f"- bg_win_radius_m: {float(cfg['BG_WIN_RADIUS_M'])}",
        f"- score_norm_pctl: {float(cfg['SCORE_NORM_PCTL'])}",
        f"- top_score_pctl: {float(cfg['TOP_SCORE_PCTL'])}",
        f"- intensity_nonzero_ratio: {intensity_stats['nonzero_ratio']:.6f}",
        f"- score_top_threshold: {score_stats['top_score_threshold']:.6f}",
        f"- dropped_high_score_overflow: {dropped_overflow}",
        "",
        "## outputs",
        f"- {relpath(run_dir, rasters_dir / 'intensity_p95_utm32.tif')}",
        f"- {relpath(run_dir, rasters_dir / 'intensity_bg_utm32.tif')}",
        f"- {relpath(run_dir, rasters_dir / 'intensity_score_tophat_utm32.tif')}",
        f"- {relpath(run_dir, rasters_dir / 'score_mask_top_utm32.tif')}",
        f"- {relpath(run_dir, pc_dir / 'high_score_points_utm32.laz')}",
        f"- {relpath(run_dir, images_dir / 'intensity_p95_preview.png')}",
        f"- {relpath(run_dir, images_dir / 'score_preview.png')}",
        f"- {relpath(run_dir, images_dir / 'score_mask_overlay.png')}",
        f"- {relpath(run_dir, tables_dir / 'intensity_stats.json')}",
        f"- {relpath(run_dir, tables_dir / 'score_stats.json')}",
        "",
        "## conclusion",
        "- 请查看 score_preview 与 score_mask_overlay 是否呈条纹块状结构以判断强度可分性。",
    ]
    write_text(run_dir / "report.md", "\n".join(report) + "\n")


if __name__ == "__main__":
    main()

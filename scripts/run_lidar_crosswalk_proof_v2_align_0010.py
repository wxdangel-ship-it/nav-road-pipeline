from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import features
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
from shapely.ops import unary_union

from scripts.pipeline_common import ensure_dir, ensure_overwrite, relpath, setup_logging, write_csv, write_json, write_text, write_gpkg_layer


@dataclass
class BevBundle:
    score: np.ndarray
    intensity: np.ndarray
    count: np.ndarray
    transform: rasterio.Affine
    res_m: float


def _latest_debug_run() -> Optional[Path]:
    runs = sorted(Path("runs").glob("lidar_intensity_debug_0010_f250_300_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _read_las_points(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        header = f.read(227)
        if len(header) < 227:
            raise RuntimeError("las_header_short")
        offset_to_points = struct.unpack_from("<I", header, 96)[0]
        point_format = struct.unpack_from("<B", header, 104)[0]
        record_len = struct.unpack_from("<H", header, 105)[0]
        point_count = struct.unpack_from("<I", header, 107)[0]
        sx, sy, sz = struct.unpack_from("<ddd", header, 131)
        ox, oy, oz = struct.unpack_from("<ddd", header, 155)
        if point_format != 0 or record_len < 20:
            raise RuntimeError("las_format_unsupported")
        f.seek(offset_to_points)
        raw = f.read(point_count * record_len)
        if len(raw) < point_count * record_len:
            raise RuntimeError("las_points_short")
        xi = np.ndarray(shape=(point_count,), dtype="<i4", buffer=raw, offset=0, strides=(record_len,))
        yi = np.ndarray(shape=(point_count,), dtype="<i4", buffer=raw, offset=4, strides=(record_len,))
        zi = np.ndarray(shape=(point_count,), dtype="<i4", buffer=raw, offset=8, strides=(record_len,))
        intens = np.ndarray(shape=(point_count,), dtype="<u2", buffer=raw, offset=12, strides=(record_len,))
        x = xi.astype(np.float64) * float(sx) + float(ox)
        y = yi.astype(np.float64) * float(sy) + float(oy)
        z = zi.astype(np.float64) * float(sz) + float(oz)
        pts = np.stack([x, y, z], axis=1).astype(np.float32)
        return pts, intens.astype(np.float32)


def _grid_spec(bounds: Tuple[float, float, float, float], res_m: float) -> Tuple[float, float, int, int]:
    minx, miny, maxx, maxy = bounds
    width = int(np.ceil((maxx - minx) / res_m)) + 1
    height = int(np.ceil((maxy - miny) / res_m)) + 1
    return float(minx), float(miny), width, height


def _group_slices(lin_idx: np.ndarray):
    order = np.argsort(lin_idx, kind="mergesort")
    lin_sorted = lin_idx[order]
    uniq, start, counts = np.unique(lin_sorted, return_index=True, return_counts=True)
    return order, uniq, start, counts


def _build_bev(points: np.ndarray, intensity: np.ndarray, roi_geom: object, res_m: float) -> BevBundle:
    minx, miny, maxx, maxy = roi_geom.bounds
    minx, miny, width, height = _grid_spec((minx, miny, maxx, maxy), res_m)
    transform = rasterio.transform.from_origin(minx, miny + height * res_m, res_m, res_m)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    ix = np.floor((x - minx) / res_m).astype(np.int32)
    iy = np.floor((y - miny) / res_m).astype(np.int32)
    valid = (ix >= 0) & (iy >= 0) & (ix < width) & (iy < height)
    ixv = ix[valid]
    iyv = iy[valid]
    zv = z[valid]
    iv = intensity[valid]
    lin = iyv.astype(np.int64) * int(width) + ixv.astype(np.int64)
    order, uniq, start, counts = _group_slices(lin)
    ix_sorted = ixv[order]
    iy_sorted = iyv[order]
    z_sorted = zv[order]
    i_sorted = iv[order]

    height_p10 = np.full((height, width), np.nan, dtype=np.float32)
    count = np.zeros((height, width), dtype=np.int32)
    for u, s, c in zip(uniq, start, counts):
        sl = slice(s, s + c)
        cx = int(ix_sorted[s])
        cy = int(iy_sorted[s])
        height_p10[cy, cx] = np.percentile(z_sorted[sl], 10).astype(np.float32)
        count[cy, cx] = int(c)

    ground_band = np.isfinite(height_p10[iyv, ixv]) & (zv <= (height_p10[iyv, ixv] + 0.06))
    ixg = ixv[ground_band]
    iyg = iyv[ground_band]
    ig = iv[ground_band]
    lin_g = iyg.astype(np.int64) * int(width) + ixg.astype(np.int64)
    order_g, uniq_g, start_g, counts_g = _group_slices(lin_g)
    ixg_sorted = ixg[order_g]
    iyg_sorted = iyg[order_g]
    ig_sorted = ig[order_g]
    p95 = np.full((height, width), np.nan, dtype=np.float32)
    for u, s, c in zip(uniq_g, start_g, counts_g):
        sl = slice(s, s + c)
        cx = int(ixg_sorted[s])
        cy = int(iyg_sorted[s])
        p95[cy, cx] = float(np.percentile(ig_sorted[sl], 95))

    return BevBundle(score=p95, intensity=p95, count=count, transform=transform, res_m=res_m)


def _box_stat(values: np.ndarray, valid_mask: np.ndarray, radius_px: int, stat: str) -> np.ndarray:
    if radius_px <= 0:
        return np.where(valid_mask, values, np.nan)
    pad = int(radius_px)
    out = np.zeros_like(values, dtype=np.float32)
    h, w = values.shape
    for y in range(h):
        y0 = max(0, y - pad)
        y1 = min(h, y + pad + 1)
        for x in range(w):
            x0 = max(0, x - pad)
            x1 = min(w, x + pad + 1)
            win = values[y0:y1, x0:x1]
            mask = valid_mask[y0:y1, x0:x1]
            vals = win[mask]
            if vals.size == 0:
                out[y, x] = 0.0
            else:
                out[y, x] = float(np.median(vals) if stat == "median" else np.mean(vals))
    return out


def _robust_mad(values: np.ndarray, valid_mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return np.where(valid_mask, values, np.nan)
    pad = int(radius_px)
    out = np.zeros_like(values, dtype=np.float32)
    h, w = values.shape
    for y in range(h):
        y0 = max(0, y - pad)
        y1 = min(h, y + pad + 1)
        for x in range(w):
            x0 = max(0, x - pad)
            x1 = min(w, x + pad + 1)
            win = values[y0:y1, x0:x1]
            mask = valid_mask[y0:y1, x0:x1]
            vals = win[mask]
            if vals.size == 0:
                out[y, x] = 0.0
            else:
                med = float(np.median(vals))
                out[y, x] = float(np.median(np.abs(vals - med)))
    return out


def _normalize(score: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    vals = score[valid_mask]
    if vals.size == 0:
        return np.zeros_like(score, dtype=np.float32)
    p99 = float(np.percentile(vals, 99))
    p99 = max(p99, 1e-6)
    return np.clip(score / p99, 0.0, 1.0).astype(np.float32)


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


def _cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    mx = float(np.mean(x))
    my = float(np.mean(y))
    sx = float(np.var(x, ddof=1)) if x.size > 1 else 0.0
    sy = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
    pooled = math.sqrt((sx + sy) / 2.0) if (sx + sy) > 0 else 1e-6
    return (mx - my) / pooled


def _profile_fft_band_ratio(score: np.ndarray, mask: np.ndarray, res_m: float) -> float:
    if np.sum(mask) == 0:
        return 0.0
    valid = mask & np.isfinite(score)
    if np.sum(valid) == 0:
        return 0.0
    ys, xs = np.where(valid)
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    coords -= coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    n = np.array([-v[1], v[0]], dtype=np.float32)
    proj = coords @ n
    bins = np.arange(float(np.min(proj)), float(np.max(proj)) + 1.0, 1.0)
    if len(bins) < 8:
        return 0.0
    digit = np.digitize(proj, bins) - 1
    profile = np.zeros((len(bins),), dtype=np.float32)
    counts = np.zeros((len(bins),), dtype=np.int32)
    for idx, b in enumerate(digit):
        if 0 <= b < len(bins):
            profile[b] += float(score[int(ys[idx]), int(xs[idx])])
            counts[b] += 1
    valid = counts > 0
    profile[valid] = profile[valid] / counts[valid]
    profile = profile - float(np.mean(profile[valid])) if np.any(valid) else profile
    fft = np.fft.rfft(profile)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(profile), d=res_m)
    band = (freqs >= 1.25) & (freqs <= 2.857)
    band_energy = float(np.sum(power[band]))
    total_energy = float(np.sum(power)) + 1e-6
    return band_energy / total_energy


def _obj(auc: float, d_score: float, band_ratio: float) -> float:
    return 2.0 * auc + 1.5 * max(0.0, min(d_score / 2.0, 1.0)) + 1.0 * max(0.0, min(math.log(max(band_ratio, 1e-6)) / math.log(5.0), 1.0))


def _metrics(score: np.ndarray, truth: Polygon, transform: rasterio.Affine) -> Dict[str, float]:
    truth_mask = features.rasterize([(truth, 1)], out_shape=score.shape, transform=transform, fill=0, dtype="uint8").astype(bool)
    ring = truth.buffer(15.0).difference(truth.buffer(5.0))
    bg_mask = features.rasterize([(ring, 1)], out_shape=score.shape, transform=transform, fill=0, dtype="uint8").astype(bool)
    inside = score[truth_mask]
    bg = score[bg_mask]
    inside = inside[np.isfinite(inside)]
    bg = bg[np.isfinite(bg)]
    if inside.size == 0 or bg.size == 0:
        return {"AUC": 0.0, "d": 0.0, "band_ratio": 0.0, "inside_empty": float(inside.size == 0), "bg_empty": float(bg.size == 0)}
    labels = np.concatenate([np.ones_like(inside), np.zeros_like(bg)])
    vals = np.concatenate([inside, bg])
    auc = _roc_auc(vals, labels)
    d_score = _cohen_d(inside, bg)
    band_ratio = _profile_fft_band_ratio(score, truth_mask, transform.a)
    if not np.isfinite(d_score):
        d_score = 0.0
    if not np.isfinite(band_ratio):
        band_ratio = 0.0
    return {"AUC": float(auc), "d": float(d_score), "band_ratio": float(band_ratio), "inside_empty": 0.0, "bg_empty": 0.0}


def _save_overlay(score: np.ndarray, transform: rasterio.Affine, truth: Polygon, out_path: Path, title: str) -> None:
    minx, miny, maxx, maxy = rasterio.transform.array_bounds(score.shape[0], score.shape[1], transform)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    vmin = float(np.nanpercentile(score, 2)) if np.isfinite(score).any() else 0.0
    vmax = float(np.nanpercentile(score, 98)) if np.isfinite(score).any() else 1.0
    ax.imshow(score, extent=(minx, maxx, miny, maxy), origin="upper", cmap="gray", vmin=vmin, vmax=vmax)
    if truth is not None and not truth.is_empty:
        for geom in getattr(truth, "geoms", [truth]):
            if isinstance(geom, Polygon):
                tx, ty = geom.exterior.xy
                ax.plot(tx, ty, color="lime", linewidth=1.0)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"lidar_crosswalk_proof_v2_align_0010_f250_300_{run_id}"
    ensure_overwrite(run_dir)
    ensure_dir(run_dir)
    setup_logging(run_dir / "run.log")

    debug_run = _latest_debug_run()
    if debug_run is None:
        write_text(run_dir / "report.md", "missing debug run")
        return 2

    raw_las = debug_run / "clip" / "raw_clip_with_intensity_utm32.las"
    roi_path = debug_run / "roi_bbox_utm32.gpkg"
    truth_path = Path("crosswalk_truth_utm32.gpkg")
    if not truth_path.exists():
        write_text(run_dir / "report.md", "missing crosswalk_truth_utm32.gpkg")
        return 2

    truth = gpd.read_file(truth_path)
    if truth.empty:
        write_text(run_dir / "report.md", "truth empty")
        return 2
    truth = truth.to_crs("EPSG:32632")
    truth_poly = unary_union(truth.geometry)

    roi_gdf = gpd.read_file(roi_path)
    roi_geom = roi_gdf.geometry.iloc[0]
    points, intensity = _read_las_points(raw_las)

    res_m = 0.05
    bev = _build_bev(points, intensity, roi_geom, res_m)
    valid = (bev.count >= 3) & np.isfinite(bev.intensity)
    bev_intensity = np.where(valid, bev.intensity, np.nan)
    radius_px = int(round(3.0 / res_m))
    mean = _box_stat(bev_intensity, valid, radius_px, "mean")
    median = _box_stat(bev_intensity, valid, radius_px, "median")
    mad = _robust_mad(bev_intensity, valid, radius_px)
    score_m1 = np.maximum(0.0, bev_intensity - mean)
    score_m2 = (bev_intensity - median) / (mad + 1e-6)
    score_m1 = _normalize(score_m1, valid)
    score_m2 = _normalize(score_m2, valid)
    score = score_m1 if np.nanmean(score_m1) >= np.nanmean(score_m2) else score_m2

    xs = np.arange(-8.0, 8.01, 0.5)
    ys = np.arange(-8.0, 8.01, 0.5)
    grid = []
    best = {"obj": -1.0}
    for dx in xs:
        for dy in ys:
            moved = translate(truth_poly, xoff=dx, yoff=dy)
            if not moved.intersects(roi_geom):
                continue
            m = _metrics(score, moved, bev.transform)
            obj = _obj(m["AUC"], m["d"], m["band_ratio"])
            grid.append({"dx": dx, "dy": dy, "theta": 0.0, "AUC": m["AUC"], "d": m["d"], "band_ratio": m["band_ratio"], "OBJ": obj})
            if obj > best["obj"]:
                best = {"dx": dx, "dy": dy, "theta": 0.0, **m, "obj": obj}

    if not grid:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "truth_out_of_extent"})
        write_text(run_dir / "report.md", "truth_out_of_extent")
        return 2

    top = sorted(grid, key=lambda r: float(r["OBJ"]), reverse=True)[:5]
    if best["AUC"] < 0.65 and best["d"] < 0.6:
        thetas = [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
        for item in top:
            for theta in thetas:
                moved = rotate(truth_poly, theta, origin="centroid", use_radians=False)
                moved = translate(moved, xoff=item["dx"], yoff=item["dy"])
                if not moved.intersects(roi_geom):
                    continue
                m = _metrics(score, moved, bev.transform)
                obj = _obj(m["AUC"], m["d"], m["band_ratio"])
                grid.append({"dx": item["dx"], "dy": item["dy"], "theta": theta, "AUC": m["AUC"], "d": m["d"], "band_ratio": m["band_ratio"], "OBJ": obj})
                if obj > best["obj"]:
                    best = {"dx": item["dx"], "dy": item["dy"], "theta": theta, **m, "obj": obj}

    grid_csv = ensure_dir(run_dir / "tables")
    write_csv(grid_csv / "grid_search.csv", grid, ["dx", "dy", "theta", "AUC", "d", "band_ratio", "OBJ"])

    heat = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    for row in grid:
        ix = int(round((row["dx"] - xs[0]) / 0.5))
        iy = int(round((row["dy"] - ys[0]) / 0.5))
        heat[iy, ix] = float(row["OBJ"])
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    ax.imshow(heat, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap="viridis")
    ax.set_xlabel("dx (m)")
    ax.set_ylabel("dy (m)")
    fig.tight_layout()
    aligned_dir = ensure_dir(run_dir / "aligned")
    fig.savefig(aligned_dir / "grid_search_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    aligned_truth = rotate(truth_poly, best.get("theta", 0.0), origin="centroid", use_radians=False)
    aligned_truth = translate(aligned_truth, xoff=best.get("dx", 0.0), yoff=best.get("dy", 0.0))
    truth_aligned = gpd.GeoDataFrame([{"geometry": aligned_truth}], geometry="geometry", crs="EPSG:32632")
    write_gpkg_layer(aligned_dir / "crosswalk_truth_aligned_utm32.gpkg", "crosswalk_truth", truth_aligned, 32632, [], overwrite=True)
    write_json(aligned_dir / "best_transform.json", best)

    images_dir = ensure_dir(run_dir / "images")
    _save_overlay(score, bev.transform, truth_poly, images_dir / "bev_score_overlay_truth_before.png", "truth before")
    _save_overlay(score, bev.transform, aligned_truth, images_dir / "bev_score_overlay_truth_aligned.png", "truth aligned")

    metrics_best = best
    passed = metrics_best["AUC"] >= 0.70 and metrics_best["d"] >= 0.8 and metrics_best["band_ratio"] >= 2.0
    reason = "ok" if passed else "no_contrast_even_after_align"
    decision = {
        "status": "PASS" if passed else "FAIL",
        "dx": metrics_best.get("dx", 0.0),
        "dy": metrics_best.get("dy", 0.0),
        "theta": metrics_best.get("theta", 0.0),
        "AUC_best": metrics_best["AUC"],
        "d_best": metrics_best["d"],
        "band_ratio_best": metrics_best["band_ratio"],
        "OBJ_best": metrics_best["obj"],
        "reason": reason,
    }
    write_json(run_dir / "decision.json", decision)

    report = [
        "# Crosswalk proof v2 align report",
        "",
        f"- conclusion: {decision['status']}",
        f"- best_dx: {decision['dx']}",
        f"- best_dy: {decision['dy']}",
        f"- best_AUC: {decision['AUC_best']:.3f}",
        f"- best_d: {decision['d_best']:.3f}",
        f"- best_band_ratio: {decision['band_ratio_best']:.3f}",
        "",
        "## Outputs",
        f"- {relpath(run_dir, aligned_dir / 'crosswalk_truth_aligned_utm32.gpkg')}",
        f"- {relpath(run_dir, aligned_dir / 'best_transform.json')}",
        f"- {relpath(run_dir, aligned_dir / 'grid_search_heatmap.png')}",
        f"- {relpath(run_dir, images_dir / 'bev_score_overlay_truth_before.png')}",
        f"- {relpath(run_dir, images_dir / 'bev_score_overlay_truth_aligned.png')}",
    ]
    write_text(run_dir / "report.md", "\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

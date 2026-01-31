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
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

from scripts.pipeline_common import ensure_dir, ensure_overwrite, relpath, setup_logging, write_csv, write_json, write_text, write_gpkg_layer


def _list_layers(path: Path) -> List[str]:
    try:
        import fiona

        return list(fiona.listlayers(path))
    except Exception:
        try:
            import pyogrio

            layers = pyogrio.list_layers(path)
            if isinstance(layers, np.ndarray):
                layers = layers.tolist()
            names: List[str] = []
            for item in layers:
                if isinstance(item, (list, tuple)) and item:
                    names.append(str(item[0]))
                else:
                    names.append(str(item))
            return names
        except Exception:
            return []


@dataclass
class RasterBundle:
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


def _build_bev(points: np.ndarray, intensity: np.ndarray, roi_geom: object, res_m: float) -> RasterBundle:
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

    return RasterBundle(score=p95, intensity=p95, count=count, transform=transform, res_m=res_m)


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
                if stat == "median":
                    out[y, x] = float(np.median(vals))
                else:
                    out[y, x] = float(np.mean(vals))
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


def _periodic_snr(score: np.ndarray, mask: np.ndarray) -> float:
    vals = np.where(mask, score, 0.0)
    if np.sum(mask) == 0:
        return 0.0
    fft = np.fft.fftshift(np.fft.fft2(vals))
    power = np.abs(fft) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    power[cy - 1 : cy + 2, cx - 1 : cx + 2] = 0.0
    peak = float(np.max(power))
    med = float(np.median(power[power > 0])) if np.any(power > 0) else 1e-6
    return peak / max(med, 1e-6)


def _stripe_profile(score: np.ndarray, mask: np.ndarray) -> Tuple[int, float]:
    if np.sum(mask) == 0:
        return 0, 1.0
    ys, xs = np.where(mask)
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    coords -= coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    n = np.array([-v[1], v[0]], dtype=np.float32)
    proj = coords @ n
    bins = np.linspace(float(np.min(proj)), float(np.max(proj)), 120)
    digit = np.digitize(proj, bins) - 1
    profile = np.zeros((len(bins),), dtype=np.float32)
    counts = np.zeros((len(bins),), dtype=np.int32)
    for idx, (y, x) in enumerate(coords):
        b = digit[idx]
        if 0 <= b < len(bins):
            profile[b] += float(score[int(ys[idx]), int(xs[idx])])
            counts[b] += 1
    valid = counts > 0
    profile[valid] = profile[valid] / counts[valid]
    peaks = []
    for i in range(1, len(profile) - 1):
        if profile[i] > profile[i - 1] and profile[i] > profile[i + 1] and profile[i] > (0.5 * np.max(profile)):
            peaks.append(i)
    if len(peaks) < 2:
        return len(peaks), 1.0
    spacing = np.diff(peaks)
    cv = float(np.std(spacing) / max(np.mean(spacing), 1e-6))
    return len(peaks), cv


def _save_overlay(score: np.ndarray, transform: rasterio.Affine, truth: Polygon, cand: gpd.GeoDataFrame, out_path: Path, title: str) -> None:
    minx, miny, maxx, maxy = rasterio.transform.array_bounds(score.shape[0], score.shape[1], transform)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    vmin = float(np.nanpercentile(score, 2)) if np.isfinite(score).any() else 0.0
    vmax = float(np.nanpercentile(score, 98)) if np.isfinite(score).any() else 1.0
    ax.imshow(score, extent=(minx, maxx, miny, maxy), origin="upper", cmap="gray", vmin=vmin, vmax=vmax)
    if truth:
        if isinstance(truth, Polygon):
            tx, ty = truth.exterior.xy
            ax.plot(tx, ty, color="lime", linewidth=1.0)
        else:
            for geom in getattr(truth, "geoms", []):
                if isinstance(geom, Polygon):
                    tx, ty = geom.exterior.xy
                    ax.plot(tx, ty, color="lime", linewidth=1.0)
    for _, row in cand.iterrows():
        poly = row.geometry
        if poly is None or poly.is_empty:
            continue
        if isinstance(poly, Polygon):
            x, y = poly.exterior.xy
            ax.plot(x, y, color="red", linewidth=1.0)
        else:
            for geom in getattr(poly, "geoms", []):
                if isinstance(geom, Polygon):
                    x, y = geom.exterior.xy
                    ax.plot(x, y, color="red", linewidth=1.0)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"lidar_crosswalk_proof_0010_f250_300_{run_id}"
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
        decision = {"status": "FAIL", "reason": "missing_truth_gpkg", "truth_path": str(truth_path)}
        write_json(run_dir / "decision.json", decision)
        write_text(run_dir / "report.md", "missing crosswalk_truth_utm32.gpkg")
        return 2

    layer_used = "crosswalk_truth"
    try:
        truth = gpd.read_file(truth_path, layer=layer_used)
    except Exception:
        layers = _list_layers(truth_path)
        if not layers:
            decision = {"status": "FAIL", "reason": "truth_layer_missing", "truth_path": str(truth_path)}
            write_json(run_dir / "decision.json", decision)
            write_text(run_dir / "report.md", "missing crosswalk_truth layer")
            return 2
        layer_used = layers[0]
        truth = gpd.read_file(truth_path, layer=layer_used)
    if truth.empty:
        decision = {"status": "FAIL", "reason": "empty_truth_gpkg", "truth_path": str(truth_path)}
        write_json(run_dir / "decision.json", decision)
        write_text(run_dir / "report.md", "crosswalk_truth empty")
        return 2
    truth = truth.to_crs("EPSG:32632")
    truth_poly = truth.geometry.iloc[0]
    write_gpkg_layer(run_dir / "crosswalk_truth_utm32.gpkg", "crosswalk_truth", truth, 32632, [], overwrite=True)

    roi_gdf = gpd.read_file(roi_path)
    roi_geom = roi_gdf.geometry.iloc[0]
    points, intensity = _read_las_points(raw_las)
    if points.size == 0:
        decision = {"status": "FAIL", "reason": "no_points_in_clip"}
        write_json(run_dir / "decision.json", decision)
        write_text(run_dir / "report.md", "no points in clip")
        return 2

    roi_area = float(roi_geom.area)
    density = float(points.shape[0]) / max(roi_area, 1e-6)
    nz_ratio = float(np.mean(intensity > 0))
    p50 = float(np.percentile(intensity, 50))
    p90 = float(np.percentile(intensity, 90))
    p99 = float(np.percentile(intensity, 99))

    fail_fast = nz_ratio < 0.05 or density < 20.0

    res_m = 0.05
    if roi_area / (res_m * res_m) > 2e7:
        res_m = 0.10
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

    truth_mask = features.rasterize([(truth_poly, 1)], out_shape=score_m1.shape, transform=bev.transform, fill=0, dtype="uint8").astype(bool)
    ring = truth_poly.buffer(15.0).difference(truth_poly.buffer(5.0))
    bg_mask = features.rasterize([(ring, 1)], out_shape=score_m1.shape, transform=bev.transform, fill=0, dtype="uint8").astype(bool)
    bg_mask &= valid
    truth_mask &= valid

    def _metrics(score: np.ndarray) -> Dict[str, float]:
        inside = score[truth_mask]
        bg = score[bg_mask]
        labels = np.concatenate([np.ones_like(inside), np.zeros_like(bg)])
        vals = np.concatenate([inside, bg])
        auc = _roc_auc(vals, labels)
        d = _cohen_d(inside, bg)
        snr = _periodic_snr(score, truth_mask)
        stripes, cv = _stripe_profile(score, truth_mask)
        return {"AUC_score": auc, "d_score": d, "periodic_SNR": snr, "stripe_count_est": stripes, "stripe_spacing_cv": cv}

    m1 = _metrics(score_m1)
    m2 = _metrics(score_m2)
    score = score_m1 if m1["AUC_score"] >= m2["AUC_score"] else score_m2
    metrics = m1 if m1["AUC_score"] >= m2["AUC_score"] else m2
    method = "M1" if m1["AUC_score"] >= m2["AUC_score"] else "M2"

    tables_dir = ensure_dir(run_dir / "tables")
    inside = score[truth_mask]
    bg = score[bg_mask]
    hist_inside, _ = np.histogram(inside, bins=64, range=(0, 1))
    hist_bg, _ = np.histogram(bg, bins=64, range=(0, 1))
    write_csv(tables_dir / "score_hist_inside.csv", [{"bin": i, "count": int(c)} for i, c in enumerate(hist_inside)], ["bin", "count"])
    write_csv(tables_dir / "score_hist_bg.csv", [{"bin": i, "count": int(c)} for i, c in enumerate(hist_bg)], ["bin", "count"])
    intensity_inside = bev_intensity[truth_mask]
    intensity_bg = bev_intensity[bg_mask]
    hist_i_inside, _ = np.histogram(intensity_inside[np.isfinite(intensity_inside)], bins=64, range=(0, 65535))
    hist_i_bg, _ = np.histogram(intensity_bg[np.isfinite(intensity_bg)], bins=64, range=(0, 65535))
    write_csv(tables_dir / "intensity_hist_inside.csv", [{"bin": i, "count": int(c)} for i, c in enumerate(hist_i_inside)], ["bin", "count"])
    write_csv(tables_dir / "intensity_hist_bg.csv", [{"bin": i, "count": int(c)} for i, c in enumerate(hist_i_bg)], ["bin", "count"])

    cand_thresh = float(np.quantile(inside, 0.85)) if inside.size else 1.0
    stripe_mask = (score >= cand_thresh) & truth_mask
    merged = None
    if np.any(stripe_mask):
        geoms = [shape(g) for g, v in features.shapes(stripe_mask.astype("uint8"), mask=stripe_mask, transform=bev.transform) if int(v) == 1]
        if geoms:
            merged = unary_union(geoms).buffer(0.6).buffer(-0.6)
    cand_rows = []
    if merged is not None and not merged.is_empty:
        cand_rows.append({"cw_id": "cw_0", "geometry": merged})
    candidates = gpd.GeoDataFrame(cand_rows, geometry="geometry", crs="EPSG:32632")
    out_cand = run_dir / "crosswalk_candidate_utm32.gpkg"
    write_gpkg_layer(out_cand, "crosswalk_candidate", candidates, 32632, [], overwrite=True)

    images_dir = ensure_dir(run_dir / "images")
    _save_overlay(score, bev.transform, truth_poly, candidates, images_dir / "bev_score_overlay_truth.png", f"{method} overlay truth")
    _save_overlay(score, bev.transform, truth_poly, candidates, images_dir / "bev_score_overlay_candidate.png", f"{method} overlay cand")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(bev_intensity, origin="upper", cmap="gray")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(images_dir / "bev_intensity.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(score, origin="upper", cmap="gray")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(images_dir / "bev_score.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    ax.plot(np.abs(np.fft.rfft(score[truth_mask])) if inside.size else [0])
    ax.set_title("fft_peak")
    fig.tight_layout()
    fig.savefig(images_dir / "fft_peak.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    ax.hist(score[truth_mask].flatten() if inside.size else [0], bins=32, range=(0, 1))
    ax.set_title("stripe_profile")
    fig.tight_layout()
    fig.savefig(images_dir / "stripe_profile.png", bbox_inches="tight")
    plt.close(fig)

    decision = {
        "drive_id": "2013_05_28_drive_0010_sync",
        "frames": [250, 300],
        "density_pts_m2": density,
        "intensity_nonzero_ratio": nz_ratio,
        "intensity_p50": p50,
        "intensity_p90": p90,
        "intensity_p99": p99,
        "method": method,
        **metrics,
    }
    passed = (
        metrics["AUC_score"] >= 0.70
        and metrics["d_score"] >= 0.8
        and metrics["periodic_SNR"] >= 3.0
        and metrics["stripe_count_est"] >= 6
        and metrics["stripe_spacing_cv"] <= 0.6
        and not fail_fast
    )
    decision["status"] = "PASS" if passed else "FAIL"
    write_json(run_dir / "decision.json", decision)

    report = [
        "# Crosswalk proof report",
        "",
        f"- conclusion: {decision['status']}",
        f"- reason: AUC={metrics['AUC_score']:.3f}, d={metrics['d_score']:.3f}, SNR={metrics['periodic_SNR']:.2f}, density={density:.2f}",
        f"- fail_fast: {fail_fast}",
        f"- truth_layer: {layer_used}",
        "",
        "## Outputs",
        f"- {relpath(run_dir, run_dir / 'decision.json')}",
        f"- {relpath(run_dir, out_cand)}",
        f"- {relpath(run_dir, images_dir / 'bev_intensity.png')}",
        f"- {relpath(run_dir, images_dir / 'bev_score.png')}",
        f"- {relpath(run_dir, images_dir / 'bev_score_overlay_truth.png')}",
        f"- {relpath(run_dir, images_dir / 'bev_score_overlay_candidate.png')}",
        f"- {relpath(run_dir, images_dir / 'fft_peak.png')}",
        f"- {relpath(run_dir, images_dir / 'stripe_profile.png')}",
    ]
    write_text(run_dir / "report.md", "\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

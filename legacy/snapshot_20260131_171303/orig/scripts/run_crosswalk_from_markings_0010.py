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
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union

from scripts.pipeline_common import (
    ensure_dir,
    ensure_overwrite,
    load_yaml,
    relpath,
    setup_logging,
    validate_output_crs,
    write_csv,
    write_text,
    write_gpkg_layer,
)


@dataclass
class Stripe:
    poly: Polygon
    length_m: float
    width_m: float
    area_m2: float
    ori_deg: float
    centroid: Tuple[float, float]


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


def _build_intensity_rasters(
    points: np.ndarray,
    intensity: np.ndarray,
    roi_geom: object,
    res_m: float,
) -> Tuple[np.ndarray, np.ndarray, rasterio.Affine]:
    minx, miny, maxx, maxy = roi_geom.bounds
    minx, miny, width, height = _grid_spec((minx, miny, maxx, maxy), res_m)
    transform = rasterio.transform.from_origin(minx, miny + height * res_m, res_m, res_m)
    x = points[:, 0]
    y = points[:, 1]
    ix = np.floor((x - minx) / res_m).astype(np.int32)
    iy = np.floor((y - miny) / res_m).astype(np.int32)
    valid = (ix >= 0) & (iy >= 0) & (ix < width) & (iy < height)
    ixv = ix[valid]
    iyv = iy[valid]
    iv = intensity[valid]

    score = np.full((height, width), np.nan, dtype=np.float32)
    if iv.size:
        lin = iyv.astype(np.int64) * int(width) + ixv.astype(np.int64)
        order = np.argsort(lin, kind="mergesort")
        lin_sorted = lin[order]
        uniq, start, counts = np.unique(lin_sorted, return_index=True, return_counts=True)
        iv_sorted = iv[order]
        ix_sorted = ixv[order]
        iy_sorted = iyv[order]
        for u, s, c in zip(uniq, start, counts):
            if c <= 0:
                continue
            sl = slice(s, s + c)
            cx = int(ix_sorted[s])
            cy = int(iy_sorted[s])
            score[cy, cx] = float(np.percentile(iv_sorted[sl], 95))
    valid_mask = np.isfinite(score)
    return score, valid_mask, transform


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


def _select_marking_threshold(score: np.ndarray, ratio_min: float, ratio_max: float) -> Tuple[float, float]:
    vals = score[np.isfinite(score)]
    if vals.size == 0:
        return 1.0, 0.0
    target = float((ratio_min + ratio_max) * 0.5)
    target = max(1e-6, min(target, 0.999))
    thr = float(np.quantile(vals, 1.0 - target))
    ratio = float(np.mean(vals >= thr))
    if ratio < ratio_min:
        thr = float(np.quantile(vals, 1.0 - ratio_min))
    if ratio > ratio_max:
        thr = float(np.quantile(vals, 1.0 - ratio_max))
    ratio = float(np.mean(vals >= thr))
    return thr, ratio


def _stripe_orientation(rect: Polygon) -> Tuple[float, float, float]:
    coords = list(rect.exterior.coords)
    if len(coords) < 5:
        return 0.0, 0.0, 0.0
    edges = []
    for i in range(4):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        edges.append((length, dx, dy))
    edges = sorted(edges, key=lambda e: e[0])
    w = edges[0][0]
    l = edges[-1][0]
    dx, dy = edges[-1][1], edges[-1][2]
    angle = math.degrees(math.atan2(dy, dx)) % 180.0
    return l, w, angle


def _stripe_candidates(
    mask: np.ndarray,
    score: np.ndarray,
    transform: rasterio.Affine,
    params: Dict[str, object],
) -> Tuple[List[Stripe], Dict[str, int]]:
    w_min, w_max = params["STRIPE_W_RANGE_M"]
    l_min, l_max = params["STRIPE_L_RANGE_M"]
    a_min, a_max = params["STRIPE_AREA_RANGE_M2"]
    stripes: List[Stripe] = []
    counts = {"raw": 0, "filtered": 0}
    for geom, val in features.shapes(mask.astype("uint8"), mask=mask.astype(bool), transform=transform):
        if int(val) != 1:
            continue
        counts["raw"] += 1
        poly = shape(geom)
        if poly.is_empty:
            continue
        area = float(poly.area)
        if area < a_min or area > a_max:
            continue
        rect = poly.minimum_rotated_rectangle
        l, w, ori = _stripe_orientation(rect)
        if l < l_min or l > l_max or w < w_min or w > w_max:
            continue
        counts["filtered"] += 1
        stripes.append(Stripe(poly=poly, length_m=l, width_m=w, area_m2=area, ori_deg=ori, centroid=(poly.centroid.x, poly.centroid.y)))
    return stripes, counts


def _ori_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _cluster_stripes(stripes: List[Stripe], params: Dict[str, object]) -> List[List[int]]:
    n = len(stripes)
    if n == 0:
        return []
    ori_tol = float(params["CROSSWALK_ORI_TOL_DEG"])
    gap_max = float(params["CROSSWALK_GAP_MAX_M"])
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _ori_diff(stripes[i].ori_deg, stripes[j].ori_deg) > ori_tol:
                continue
            if stripes[i].poly.distance(stripes[j].poly) > gap_max:
                continue
            li = stripes[i].length_m
            lj = stripes[j].length_m
            if abs(li - lj) / max(li, lj, 1e-6) > 0.6:
                continue
            adj[i].append(j)
            adj[j].append(i)
    clusters: List[List[int]] = []
    seen = [False] * n
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        clusters.append(comp)
    return clusters


def _mean_score_in_poly(score: np.ndarray, transform: rasterio.Affine, poly: Polygon) -> float:
    mask = features.rasterize([(poly, 1)], out_shape=score.shape, transform=transform, fill=0, dtype="uint8").astype(bool)
    vals = score[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals))


def _overlay_plot(
    score: np.ndarray,
    transform: rasterio.Affine,
    candidates: gpd.GeoDataFrame,
    out_path: Path,
) -> None:
    minx, miny, maxx, maxy = rasterio.transform.array_bounds(score.shape[0], score.shape[1], transform)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    vmin = float(np.nanpercentile(score, 2)) if np.isfinite(score).any() else 0.0
    vmax = float(np.nanpercentile(score, 98)) if np.isfinite(score).any() else 1.0
    ax.imshow(score, extent=(minx, maxx, miny, maxy), origin="upper", cmap="gray", vmin=vmin, vmax=vmax)
    for _, row in candidates.iterrows():
        poly = row.geometry
        if poly is None or poly.is_empty:
            continue
        x, y = poly.exterior.xy
        ax.plot(x, y, color="white", linewidth=1.0)
        rect = poly.minimum_rotated_rectangle
        rx, ry = rect.exterior.xy
        ax.plot(rx, ry, color="red", linewidth=1.0)
        label = f"{int(row['stripe_count'])} | {row['mrr_w_m']:.1f}/{row['mrr_l_m']:.1f}"
        ax.text(poly.centroid.x, poly.centroid.y, label, color="yellow", fontsize=6)
    if candidates.empty:
        ax.text((minx + maxx) / 2, (miny + maxy) / 2, "0 candidates", color="red", fontsize=12, ha="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _hist_plot(angles: List[float], debug_counts: Dict[str, int], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    if angles:
        ax.hist(angles, bins=36, range=(0, 180), color="steelblue", alpha=0.8)
    ax.set_xlabel("Orientation (deg)")
    ax.set_ylabel("Count")
    text = f"raw={debug_counts.get('raw',0)}, stripe={debug_counts.get('filtered',0)}, clusters={debug_counts.get('clusters',0)}, candidates={debug_counts.get('candidates',0)}"
    ax.set_title(text)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    cfg = load_yaml(Path("configs/lidar_semantic_v2_0010.yaml"))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"lidar_crosswalk_stripe_0010_f250_300_{run_id}"
    if bool(cfg.get("OVERWRITE", True)):
        ensure_overwrite(run_dir)
    ensure_dir(run_dir)
    setup_logging(run_dir / "run.log")

    debug_run = _latest_debug_run()
    if debug_run is None:
        write_text(run_dir / "report.md", "missing intensity debug run")
        return 2

    raw_las = debug_run / "clip" / "raw_clip_with_intensity_utm32.las"
    roi_path = debug_run / "roi_bbox_utm32.gpkg"
    if not raw_las.exists() or not roi_path.exists():
        write_text(run_dir / "report.md", "missing raw clip or roi bbox")
        return 2

    roi_gdf = gpd.read_file(roi_path)
    if roi_gdf.empty:
        write_text(run_dir / "report.md", "roi empty")
        return 2
    roi_geom = roi_gdf.geometry.iloc[0]

    points, intensity = _read_las_points(raw_las)
    if points.size == 0:
        write_text(run_dir / "report.md", "no points in raw clip")
        return 2

    res_m = float(cfg.get("RASTER_RES_M", 0.2))
    score_p95, valid_mask, transform = _build_intensity_rasters(points, intensity, roi_geom, res_m)
    radius_px = int(round(float(cfg.get("BG_WIN_RADIUS_M", 3.0)) / res_m))
    bg_mean = _box_mean(np.nan_to_num(score_p95, nan=0.0), valid_mask, radius_px)
    score_raw = np.maximum(0.0, score_p95 - bg_mean)
    if np.isfinite(score_raw).any():
        p99 = float(np.nanpercentile(score_raw, 99))
    else:
        p99 = 1.0
    p99 = max(p99, 1e-6)
    score = np.clip(score_raw / p99, 0.0, 1.0).astype(np.float32)

    ratio_min = float(cfg.get("MARKING_AREA_RATIO_MIN", 0.005))
    ratio_max = float(cfg.get("MARKING_AREA_RATIO_MAX", 0.04))
    thr, ratio = _select_marking_threshold(score, ratio_min, ratio_max)
    marking_mask = score >= float(thr)

    road_mask_used = False
    params = {
        "CROSSWALK_ORI_TOL_DEG": float(cfg.get("CROSSWALK_ORI_TOL_DEG", 12.0)),
        "CROSSWALK_GAP_MAX_M": float(cfg.get("CROSSWALK_GAP_MAX_M", 2.2)),
        "CROSSWALK_MERGE_BUF_M": float(cfg.get("CROSSWALK_MERGE_BUF_M", 0.6)),
        "STRIPE_W_RANGE_M": cfg.get("STRIPE_W_RANGE_M", [0.12, 0.80]),
        "STRIPE_L_RANGE_M": cfg.get("STRIPE_L_RANGE_M", [0.80, 12.0]),
        "STRIPE_AREA_RANGE_M2": cfg.get("STRIPE_AREA_RANGE_M2", [0.15, 20.0]),
        "CLUSTER_MIN_STRIPES": int(cfg.get("CLUSTER_MIN_STRIPES", 6)),
        "CLUSTER_W_RANGE_M": cfg.get("CLUSTER_W_RANGE_M", [2.5, 10.0]),
        "CLUSTER_L_RANGE_M": cfg.get("CLUSTER_L_RANGE_M", [3.0, 28.0]),
        "CLUSTER_AREA_RANGE_M2": cfg.get("CLUSTER_AREA_RANGE_M2", [10.0, 350.0]),
        "USE_ROAD_MASK": bool(cfg.get("USE_ROAD_MASK", True)),
        "marking_threshold": float(thr),
        "marking_ratio": float(ratio),
    }

    stripes, stripe_counts = _stripe_candidates(marking_mask, score, transform, params)
    clusters = _cluster_stripes(stripes, params)

    candidates_rows = []
    cluster_count = 0
    for comp in clusters:
        if len(comp) < int(params["CLUSTER_MIN_STRIPES"]):
            continue
        cluster_count += 1
        polys = [stripes[i].poly for i in comp]
        merged = unary_union(polys)
        merged = merged.buffer(float(params["CROSSWALK_MERGE_BUF_M"])).buffer(-float(params["CROSSWALK_MERGE_BUF_M"]))
        rect = merged.minimum_rotated_rectangle
        l, w, ori = _stripe_orientation(rect)
        area = float(merged.area)
        w_min, w_max = params["CLUSTER_W_RANGE_M"]
        l_min, l_max = params["CLUSTER_L_RANGE_M"]
        a_min, a_max = params["CLUSTER_AREA_RANGE_M2"]
        if not (w_min <= w <= w_max and l_min <= l <= l_max and a_min <= area <= a_max):
            continue
        mean_score = _mean_score_in_poly(score, transform, merged)
        stripe_density = float(len(comp)) / max(w, 1e-6)
        if stripe_density < 0.8:
            continue
        candidates_rows.append(
            {
                "cw_id": f"cw_{cluster_count}",
                "stripe_count": int(len(comp)),
                "ori_deg": float(ori),
                "mrr_w_m": float(w),
                "mrr_l_m": float(l),
                "area_m2": float(area),
                "mean_score": float(mean_score),
                "params_json": json.dumps(params, ensure_ascii=True),
                "geometry": merged,
            }
        )

    if candidates_rows:
        candidates = gpd.GeoDataFrame(candidates_rows, geometry="geometry", crs="EPSG:32632")
    else:
        candidates = gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32632")
    drive_id = "2013_05_28_drive_0010_sync"
    drive_dir = ensure_dir(run_dir / "drives" / drive_id)
    vec_dir = ensure_dir(drive_dir / "vectors")
    qa_dir = ensure_dir(drive_dir / "qa")

    out_gpkg = vec_dir / "crosswalk_candidates_utm32.gpkg"
    write_gpkg_layer(out_gpkg, "crosswalk_candidates", candidates, 32632, [], overwrite=True)

    debug_counts = {
        "raw": stripe_counts.get("raw", 0),
        "filtered": stripe_counts.get("filtered", 0),
        "clusters": len(clusters),
        "candidates": len(candidates),
    }
    debug_rows = [{"stage": k, "count": v} for k, v in debug_counts.items()]
    debug_rows.append({"stage": "marking_threshold", "count": thr})
    write_csv(qa_dir / "crosswalk_debug.csv", debug_rows, ["stage", "count"])

    overlay_path = qa_dir / "crosswalk_overlay.png"
    _overlay_plot(score, transform, candidates, overlay_path)

    hist_path = qa_dir / "stripe_orientation_hist.png"
    angles = [s.ori_deg for s in stripes]
    _hist_plot(angles, debug_counts, hist_path)

    report_lines = [
        "# Crosswalk by stripe clustering",
        "",
        f"- drive_id: {drive_id}",
        f"- debug_run: {debug_run}",
        f"- road_mask_used: {road_mask_used}",
        "",
        "## Params",
        "```json",
        json.dumps(params, indent=2),
        "```",
        "",
        "## Counts",
        f"- stripes_raw: {debug_counts['raw']}",
        f"- stripes_filtered: {debug_counts['filtered']}",
        f"- clusters: {debug_counts['clusters']}",
        f"- candidates: {debug_counts['candidates']}",
        "",
        "## Outputs",
        f"- {relpath(run_dir, out_gpkg)}",
        f"- {relpath(run_dir, qa_dir / 'crosswalk_debug.csv')}",
        f"- {relpath(run_dir, overlay_path)}",
        f"- {relpath(run_dir, hist_path)}",
    ]
    write_text(qa_dir / "report.md", "\n".join(report_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

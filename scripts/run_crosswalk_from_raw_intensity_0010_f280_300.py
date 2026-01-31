
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

from pipeline._io import load_yaml
from pipeline.calib.kitti360_world import transform_points_V_to_W
from pipeline.datasets.kitti360_io import load_kitti360_lidar_points
from pipeline.lidar_semantic.export_pointcloud import write_las
from scripts.pipeline_common import ensure_overwrite, now_ts, relpath, setup_logging, write_csv, write_gpkg_layer, write_json, write_text

LOG = logging.getLogger("crosswalk_from_raw_intensity_0010_f280_300")

TRUTH_GPKG = Path(r"E:\Work\nav-road-pipeline\crosswalk_truth_utm32.gpkg")
RANSAC_SAMPLE_MAX = 20000

REQUIRED_KEYS = [
    "BUFFER_M",
    "GRID_RES_M",
    "INT_STAT",
    "MIN_CELL_PTS",
    "BG_WIN_RADIUS_M",
    "SCORE_NORM_PCTL",
    "SCORE_TH_PCTL",
    "MORPH_CLOSE_M",
    "MORPH_OPEN_M",
    "MIN_COMPONENT_AREA_M2",
    "STRIPE_W_RANGE_M",
    "STRIPE_L_RANGE_M",
    "STRIPE_AREA_RANGE_M2",
    "ORI_TOL_DEG",
    "GAP_MAX_M",
    "CLUSTER_MIN_STRIPES",
    "CAND_W_RANGE_M",
    "CAND_L_RANGE_M",
    "CAND_AREA_RANGE_M2",
    "OVERWRITE",
]


@dataclass
class BevBundle:
    p95: np.ndarray
    bg: np.ndarray
    score: np.ndarray
    score_mask: np.ndarray
    transform: rasterio.Affine
    valid_mask: np.ndarray
    res_m: float


def _load_cfg(path: Path) -> Dict[str, object]:
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


def _auto_find_kitti_root(scans: List[str]) -> Optional[Path]:
    env_root = os.environ.get("POC_DATA_ROOT", "").strip()
    if env_root:
        scans.append(env_root)
        p = Path(env_root)
        if p.exists():
            return p
    for cand in [r"E:\KITTI360\KITTI-360", r"D:\KITTI360\KITTI-360", r"C:\KITTI360\KITTI-360"]:
        scans.append(cand)
        p = Path(cand)
        if p.exists():
            return p
    repo = Path(".").resolve()
    for base in [repo / "data", repo / "datasets"]:
        if not base.exists():
            continue
        for child in base.iterdir():
            scans.append(str(child))
            if child.is_dir() and ("KITTI-360" in child.name or "KITTI360" in child.name):
                return child
    return None


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


def _frame_ids(start: int, end: int) -> List[str]:
    return [f"{i:010d}" for i in range(int(start), int(end) + 1)]


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
        record_len = int(values[14])
        count = int(values[15])
        sx, sy, sz = float(values[21]), float(values[22]), float(values[23])
        ox, oy, oz = float(values[24]), float(values[25]), float(values[26])
        if record_len < 20:
            raise RuntimeError("unsupported_point_record_length")
        f.seek(offset_to_points)
        raw = f.read(count * record_len)
    pad = record_len - 20
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
    arr = np.frombuffer(raw, dtype=dtype, count=count)
    x = arr["x"].astype(np.float64) * sx + ox
    y = arr["y"].astype(np.float64) * sy + oy
    z = arr["z"].astype(np.float64) * sz + oz
    intensity = arr["intensity"].astype(np.uint16)
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    return points, intensity


def _fit_plane_with_sampling(
    points: np.ndarray, iters: int, dist_thresh: float, normal_min_z: float, sample_max: int
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if points.size == 0:
        return None, None
    if points.shape[0] > int(sample_max):
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=int(sample_max), replace=False)
        sample = points[idx]
    else:
        sample = points
    n, d, _ = _ransac_plane(sample, iters, dist_thresh, normal_min_z)
    return n, d


def _ransac_plane(
    points: np.ndarray, iters: int, dist_thresh: float, normal_min_z: float
) -> Tuple[Optional[np.ndarray], Optional[float], np.ndarray]:
    if points.shape[0] < 3:
        return None, None, np.zeros((0,), dtype=bool)
    rng = np.random.default_rng(0)
    best_inliers = np.zeros((points.shape[0],), dtype=bool)
    best_count = 0
    for _ in range(int(iters)):
        idx = rng.choice(points.shape[0], size=3, replace=False)
        p1, p2, p3 = points[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm = float(np.linalg.norm(n))
        if norm < 1e-6:
            continue
        n = n / norm
        if abs(float(n[2])) < float(normal_min_z):
            continue
        d = -float(np.dot(n, p1))
        dist = np.abs(points @ n + d)
        inliers = dist < float(dist_thresh)
        count = int(np.sum(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_count == 0:
        return None, None, np.zeros((points.shape[0],), dtype=bool)
    inlier_pts = points[best_inliers]
    centroid = np.mean(inlier_pts, axis=0)
    _, _, vv = np.linalg.svd(inlier_pts - centroid)
    n = vv[-1, :]
    n = n / max(1e-6, float(np.linalg.norm(n)))
    if n[2] < 0:
        n = -n
    d = -float(np.dot(n, centroid))
    dist = np.abs(points @ n + d)
    inliers = dist < float(dist_thresh)
    return n, d, inliers


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


def _build_bev(points_xyz: np.ndarray, intensity: np.ndarray, cfg: Dict[str, object]) -> Optional[BevBundle]:
    if points_xyz.size == 0:
        return None
    res_m = float(cfg["GRID_RES_M"])
    minx, miny, maxy_aligned, width, height, transform = _grid_spec(points_xyz[:, :2], res_m)
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    col = np.floor((x - minx) / res_m).astype(np.int64)
    row = np.floor((maxy_aligned - y) / res_m).astype(np.int64)
    valid = (col >= 0) & (row >= 0) & (col < width) & (row < height)
    col_v = col[valid]
    row_v = row[valid]
    int_v = intensity[valid].astype(np.float64)
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
        top_thr = float(np.percentile(score_valid_norm, float(cfg["SCORE_TH_PCTL"])))
    score_mask = (score_norm >= top_thr) & np.isfinite(score_norm)
    return BevBundle(p95=p95, bg=bg, score=score_norm, score_mask=score_mask, transform=transform, valid_mask=valid_mask, res_m=res_m)


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


def _plot_score(arr: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr, origin="upper", cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_mask(arr: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr, origin="upper", cmap="gray")
    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _overlay_polys(score: np.ndarray, polys: List[Polygon], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(score, origin="upper", cmap="gray", vmin=0.0, vmax=1.0)
    for poly in polys:
        if poly.is_empty:
            continue
        x, y = poly.exterior.xy
        ax.plot(x, y, color="red", linewidth=1)
    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _make_disk(radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    r = int(radius_px)
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    mask = (x * x + y * y) <= r * r
    return mask.astype(np.uint8)


def _binary_dilate(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if kernel.size == 1:
        return mask.copy()
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant", constant_values=0)
    out = np.zeros_like(mask, dtype=np.uint8)
    ys, xs = np.where(kernel > 0)
    for dy, dx in zip(ys - pad_y, xs - pad_x):
        out |= padded[pad_y + dy : pad_y + dy + mask.shape[0], pad_x + dx : pad_x + dx + mask.shape[1]]
    return out.astype(np.uint8)


def _binary_erode(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if kernel.size == 1:
        return mask.copy()
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant", constant_values=1)
    out = np.ones_like(mask, dtype=np.uint8)
    ys, xs = np.where(kernel > 0)
    for dy, dx in zip(ys - pad_y, xs - pad_x):
        out &= padded[pad_y + dy : pad_y + dy + mask.shape[0], pad_x + dx : pad_x + dx + mask.shape[1]]
    return out.astype(np.uint8)


def _binary_open(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return _binary_dilate(_binary_erode(mask, kernel), kernel)


def _binary_close(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return _binary_erode(_binary_dilate(mask, kernel), kernel)


def _mrr_params(poly: Polygon) -> Tuple[float, float, float]:
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    if len(coords) < 4:
        return 0.0, 0.0, 0.0
    edges = []
    for i in range(4):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        edges.append(((x1 - x0), (y1 - y0)))
    lengths = [np.hypot(e[0], e[1]) for e in edges]
    order = np.argsort(lengths)[::-1]
    l = float(lengths[order[0]])
    w = float(lengths[order[2]])
    ex, ey = edges[order[0]]
    ori = float(np.degrees(np.arctan2(ey, ex))) % 180.0
    return w, l, ori


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _score_stats_inside_bg(points_xy: np.ndarray, intensity: np.ndarray, truth: Polygon, bg_ring: Polygon) -> Dict[str, float]:
    inside_mask = _mask_points_in_polygon(points_xy, truth)
    bg_mask = _mask_points_in_polygon(points_xy, bg_ring)
    inside = intensity[inside_mask]
    bg = intensity[bg_mask]
    if inside.size == 0 or bg.size == 0:
        return {"auc": 0.0, "delta_p95": 0.0, "inside_n": int(inside.size), "bg_n": int(bg.size)}
    labels = np.concatenate([np.ones_like(inside), np.zeros_like(bg)])
    scores = np.concatenate([inside, bg]).astype(np.float64)
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(scores.size) + 1
    pos = labels == 1
    n_pos = int(np.sum(pos))
    n_neg = int(scores.size - n_pos)
    sum_ranks = float(np.sum(ranks[pos]))
    auc = (sum_ranks - n_pos * (n_pos + 1) / 2.0) / max(1, n_pos * n_neg)
    delta_p95 = float(np.percentile(inside, 95) - np.percentile(bg, 95))
    return {"auc": float(auc), "delta_p95": delta_p95, "inside_n": int(inside.size), "bg_n": int(bg.size)}


def main() -> None:
    cfg_path = Path("configs/crosswalk_from_raw_intensity_0010_f280_300.yaml")
    cfg = _load_cfg(cfg_path)
    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise KeyError(f"Missing required key {key} in {cfg_path}")

    run_id = now_ts()
    run_dir = Path("runs") / f"crosswalk_from_raw_intensity_0010_f280_300_{run_id}"
    if bool(cfg.get("OVERWRITE", True)):
        ensure_overwrite(run_dir)
    setup_logging(run_dir / "run.log")
    _write_resolved(run_dir, cfg)

    gis_dir = run_dir / "gis"
    ras_dir = run_dir / "rasters"
    img_dir = run_dir / "images"
    tbl_dir = run_dir / "tables"

    layers = _list_layers(TRUTH_GPKG)
    write_csv(tbl_dir / "layers.csv", layers, ["name", "geometry_type"])
    layer_name = _pick_truth_layer(layers)
    if layer_name is None:
        report = [
            "# Crosswalk from raw intensity 0010 f280-300",
            "",
            "- status: FAIL",
            "- reason: no_polygon_layer_in_truth_gpkg",
            f"- truth: {TRUTH_GPKG}",
            f"- layers_csv: {relpath(run_dir, tbl_dir / 'layers.csv')}",
        ]
        write_text(run_dir / "report.md", "\n".join(report) + "\n")
        return

    truth_gdf = gpd.read_file(TRUTH_GPKG, layer=layer_name)
    if truth_gdf.crs is None:
        truth_gdf = truth_gdf.set_crs(32632)
    if int(truth_gdf.crs.to_epsg() or 0) != 32632:
        truth_gdf = truth_gdf.to_crs(32632)
    truth_geom = _truth_geom(truth_gdf)
    truth_gdf = gpd.GeoDataFrame(geometry=[truth_geom], crs=truth_gdf.crs)
    write_gpkg_layer(gis_dir / "truth_selected_utm32.gpkg", "truth_selected", truth_gdf, 32632, [])
    truth_buf = truth_geom.buffer(float(cfg["BUFFER_M"]))
    buf_gdf = gpd.GeoDataFrame(geometry=[truth_buf], crs=truth_gdf.crs)
    write_gpkg_layer(gis_dir / "truth_buffer10_utm32.gpkg", "truth_buffer10", buf_gdf, 32632, [])
    bg_ring = truth_buf.difference(truth_geom)

    raw_roi = None
    for run in sorted(Path("runs").glob("crosswalk_intensity_ablation_0010_f280_300_*"), key=lambda p: p.stat().st_mtime, reverse=True):
        cand = run / "pointcloud" / "roi_ground_raw_utm32.laz"
        if cand.exists():
            raw_roi = cand
            break

    points_world = None
    intensity = None
    if raw_roi is not None:
        points_world, intensity = _read_las_points(raw_roi)
    else:
        clip = None
        for run in sorted(Path("runs").glob("lidar_clip_truthbuf10_0010_f280_300_*"), key=lambda p: p.stat().st_mtime, reverse=True):
            cand = run / "pointcloud" / "clip_truthbuf10_utm32.laz"
            if cand.exists():
                clip = cand
                break
        if clip is None:
            report = [
                "# Crosswalk from raw intensity 0010 f280-300",
                "",
                "- status: FAIL",
                "- reason: missing_input_pointcloud",
            ]
            write_text(run_dir / "report.md", "\n".join(report) + "\n")
            return
        points_world, intensity = _read_las_points(clip)
        n_raw, d_raw = _fit_plane_with_sampling(points_world, 200, 0.12, 0.90, RANSAC_SAMPLE_MAX)
        if n_raw is None:
            report = [
                "# Crosswalk from raw intensity 0010 f280-300",
                "",
                "- status: FAIL",
                "- reason: ransac_failed",
            ]
            write_text(run_dir / "report.md", "\n".join(report) + "\n")
            return
        dist = np.abs(points_world @ n_raw + float(d_raw))
        inliers = dist <= 0.12
        points_world = points_world[inliers]
        intensity = intensity[inliers]

    finite_mask = np.isfinite(points_world).all(axis=1)
    points_world = points_world[finite_mask]
    intensity = intensity[finite_mask]
    if points_world.size == 0:
        report = [
            "# Crosswalk from raw intensity 0010 f280-300",
            "",
            "- status: FAIL",
            "- reason: no_points",
        ]
        write_text(run_dir / "report.md", "\n".join(report) + "\n")
        return

    roi_mask = _mask_points_in_polygon(points_world[:, :2], truth_buf)
    points_roi = points_world[roi_mask]
    intensity_roi = intensity[roi_mask]

    if points_roi.size == 0:
        report = [
            "# Crosswalk from raw intensity 0010 f280-300",
            "",
            "- status: FAIL",
            "- reason: roi_empty",
        ]
        write_text(run_dir / "report.md", "\n".join(report) + "\n")
        return

    bev = _build_bev(points_roi, intensity_roi, cfg)
    if bev is None:
        report = [
            "# Crosswalk from raw intensity 0010 f280-300",
            "",
            "- status: FAIL",
            "- reason: bev_empty",
        ]
        write_text(run_dir / "report.md", "\n".join(report) + "\n")
        return

    _write_raster(ras_dir / "intensity_p95_utm32.tif", bev.p95.astype(np.float32), bev.transform, 32632, np.nan)
    _write_raster(ras_dir / "intensity_bg_utm32.tif", bev.bg.astype(np.float32), bev.transform, 32632, np.nan)
    _write_raster(ras_dir / "score_utm32.tif", bev.score.astype(np.float32), bev.transform, 32632, np.nan)
    _write_raster(ras_dir / "score_mask_utm32.tif", bev.score_mask.astype(np.uint8), bev.transform, 32632, 0)

    img_dir.mkdir(parents=True, exist_ok=True)
    _plot_score(bev.score, img_dir / "score_preview.png", "score")
    _plot_mask(bev.score_mask.astype(np.uint8), img_dir / "score_mask_preview.png", "score mask")

    close_px = int(np.ceil(float(cfg["MORPH_CLOSE_M"]) / bev.res_m))
    open_px = int(np.ceil(float(cfg["MORPH_OPEN_M"]) / bev.res_m))
    mask = bev.score_mask.astype(np.uint8)
    mask = _binary_close(mask, _make_disk(close_px))
    mask = _binary_open(mask, _make_disk(open_px))

    shapes = list(features.shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=bev.transform))
    components = []
    component_polys: List[Polygon] = []
    for geom, value in shapes:
        if value == 0:
            continue
        if geom["type"] != "Polygon":
            continue
        poly = Polygon(geom["coordinates"][0])
        if poly.is_empty:
            continue
        area = float(poly.area)
        if area < float(cfg["MIN_COMPONENT_AREA_M2"]):
            continue
        components.append({"area_m2": area, "perimeter_m": float(poly.length)})
        component_polys.append(poly)

    write_csv(tbl_dir / "components.csv", components, ["area_m2", "perimeter_m"])

    stripes: List[Dict[str, object]] = []
    stripe_polys: List[Polygon] = []
    w_min, w_max = cfg["STRIPE_W_RANGE_M"]
    l_min, l_max = cfg["STRIPE_L_RANGE_M"]
    a_min, a_max = cfg["STRIPE_AREA_RANGE_M2"]
    for poly in component_polys:
        w, l, ori = _mrr_params(poly)
        area = float(poly.area)
        if w < float(w_min) or w > float(w_max):
            continue
        if l < float(l_min) or l > float(l_max):
            continue
        if area < float(a_min) or area > float(a_max):
            continue
        stripes.append({"w": w, "l": l, "ori": ori, "area_m2": area})
        stripe_polys.append(poly)

    if stripe_polys:
        gdf_stripes = gpd.GeoDataFrame(stripes, geometry=stripe_polys, crs="EPSG:32632")
        write_gpkg_layer(gis_dir / "stripes_utm32.gpkg", "stripes", gdf_stripes, 32632, [])
        write_csv(tbl_dir / "stripes.csv", stripes, ["w", "l", "ori", "area_m2"])
    else:
        write_csv(tbl_dir / "stripes.csv", [], ["w", "l", "ori", "area_m2"])

    stripe_count = len(stripe_polys)
    clusters: List[Dict[str, object]] = []
    candidates: List[Dict[str, object]] = []
    candidate_polys: List[Polygon] = []
    if stripe_polys:
        n = len(stripe_polys)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        ori_vals = [s["ori"] for s in stripes]
        for i in range(n):
            for j in range(i + 1, n):
                if _angle_diff(ori_vals[i], ori_vals[j]) > float(cfg["ORI_TOL_DEG"]):
                    continue
                if stripe_polys[i].distance(stripe_polys[j]) > float(cfg["GAP_MAX_M"]):
                    continue
                union(i, j)

        clusters_map: Dict[int, List[int]] = {}
        for i in range(n):
            clusters_map.setdefault(find(i), []).append(i)

        for idxs in clusters_map.values():
            if len(idxs) < int(cfg["CLUSTER_MIN_STRIPES"]):
                continue
            polys = [stripe_polys[i] for i in idxs]
            union_poly = unary_union(polys)
            union_poly = union_poly.buffer(float(cfg["MORPH_CLOSE_M"])).buffer(-float(cfg["MORPH_CLOSE_M"]))
            if union_poly.is_empty:
                continue
            w, l, ori = _mrr_params(union_poly)
            area = float(union_poly.area)
            w_min_c, w_max_c = cfg["CAND_W_RANGE_M"]
            l_min_c, l_max_c = cfg["CAND_L_RANGE_M"]
            a_min_c, a_max_c = cfg["CAND_AREA_RANGE_M2"]
            if w < float(w_min_c) or w > float(w_max_c):
                continue
            if l < float(l_min_c) or l > float(l_max_c):
                continue
            if area < float(a_min_c) or area > float(a_max_c):
                continue
            mask_r = features.rasterize([(union_poly, 1)], out_shape=bev.score.shape, transform=bev.transform, fill=0).astype(bool)
            score_mean = float(np.nanmean(bev.score[mask_r])) if np.any(mask_r) else 0.0
            candidates.append(
                {
                    "stripe_count": len(idxs),
                    "w": w,
                    "l": l,
                    "ori": ori,
                    "area_m2": area,
                    "score_mean": score_mean,
                }
            )
            candidate_polys.append(union_poly)
            clusters.append({"stripe_count": len(idxs), "w": w, "l": l, "ori": ori, "area_m2": area})

    write_csv(tbl_dir / "clusters.csv", clusters, ["stripe_count", "w", "l", "ori", "area_m2"])

    if candidates:
        gdf_cand = gpd.GeoDataFrame(candidates, geometry=candidate_polys, crs="EPSG:32632")
        write_gpkg_layer(gis_dir / "crosswalk_candidate_utm32.gpkg", "crosswalk_candidates", gdf_cand, 32632, [])
    else:
        gdf_cand = gpd.GeoDataFrame(geometry=[], crs="EPSG:32632")
        write_gpkg_layer(gis_dir / "crosswalk_candidate_utm32.gpkg", "crosswalk_candidates", gdf_cand, 32632, [])

    _overlay_polys(bev.score, stripe_polys, img_dir / "stripes_overlay.png", "stripes")
    _overlay_polys(bev.score, candidate_polys, img_dir / "candidate_overlay.png", "candidates")

    stats_inside = _score_stats_inside_bg(points_roi[:, :2], intensity_roi, truth_geom, bg_ring)
    write_json(tbl_dir / "stats.json", stats_inside)

    status = "PASS" if len(candidate_polys) >= 1 else "WARN"
    decision = {
        "status": status,
        "candidate_count": len(candidate_polys),
        "stripe_count": stripe_count,
        "stats": stats_inside,
    }
    write_json(run_dir / "decision.json", decision)

    report = [
        "# Crosswalk from raw intensity 0010 f280-300",
        "",
        f"- status: {status}",
        f"- candidate_count: {len(candidate_polys)}",
        f"- stripe_count: {stripe_count}",
        f"- auc: {stats_inside['auc']:.4f}",
        f"- delta_p95: {stats_inside['delta_p95']:.1f}",
        "",
        "## outputs",
        f"- {relpath(run_dir, gis_dir / 'crosswalk_candidate_utm32.gpkg')}",
        f"- {relpath(run_dir, gis_dir / 'stripes_utm32.gpkg')}",
        f"- {relpath(run_dir, ras_dir / 'score_utm32.tif')}",
        f"- {relpath(run_dir, img_dir / 'candidate_overlay.png')}",
        f"- {relpath(run_dir, img_dir / 'stripes_overlay.png')}",
    ]
    write_text(run_dir / "report.md", "\n".join(report) + "\n")


if __name__ == "__main__":
    main()

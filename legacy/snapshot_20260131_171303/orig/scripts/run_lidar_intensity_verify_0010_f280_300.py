from __future__ import annotations

import hashlib
import json
import math
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

from pipeline.datasets.kitti360_io import _find_velodyne_dir, load_kitti360_lidar_points, load_kitti360_lidar_points_world_full
from pipeline.lidar_semantic.accum_points_world import _voxel_downsample
from pipeline.lidar_semantic.export_pointcloud import write_las
from scripts.pipeline_common import (
    ensure_dir,
    ensure_overwrite,
    relpath,
    setup_logging,
    write_csv,
    write_gpkg_layer,
    write_json,
    write_text,
)


LOG = logging.getLogger("lidar_intensity_verify")

REQUIRED_KEYS = [
    "FRAME_START",
    "FRAME_END",
    "STRIDE",
    "TARGET_EPSG",
    "TRUTH_GPKG",
    "TRUTH_LAYER",
    "ROI_PAD_M",
    "BG_INNER_M",
    "BG_OUTER_M",
    "SHIFT_MAX_M",
    "SHIFT_STEP_M_COARSE",
    "SHIFT_STEP_M_FINE",
    "GROUND_ENABLE",
    "GROUND_RANSAC_DZ",
    "RANGE_MIN_M",
    "RANGE_MAX_M",
    "INTENSITY_AUTO_SCALE",
    "INTENSITY_USABLE_NONZERO_MIN",
    "INTENSITY_MIN_DYNAMIC_RANGE",
    "EXPORT_MAX_POINTS",
    "OVERWRITE",
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
    raw = json.dumps(_normalize(cfg), sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _dump_resolved(run_dir: Path, cfg: Dict[str, object]) -> None:
    import yaml

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8"
    )

def _write_resolved(run_dir: Path, cfg: Dict[str, object]) -> str:
    _dump_resolved(run_dir, cfg)
    params_hash = _hash_cfg(cfg)
    (run_dir / "params_hash.txt").write_text(params_hash + "\n", encoding="utf-8")
    return params_hash


def _resolve_config(base: Dict[str, object], run_dir: Path) -> Tuple[Dict[str, object], str]:
    cfg = dict(base)
    defaults = {
        "FRAME_START": 280,
        "FRAME_END": 300,
        "STRIDE": 1,
        "TARGET_EPSG": 32632,
        "TRUTH_GPKG": r"E:\Work\nav-road-pipeline\crosswalk_truth_utm32.gpkg",
        "TRUTH_LAYER": "crosswalk_truth",
        "ROI_PAD_M": 25.0,
        "BG_INNER_M": 10.0,
        "BG_OUTER_M": 35.0,
        "SHIFT_MAX_M": 8.0,
        "SHIFT_STEP_M_COARSE": 0.5,
        "SHIFT_STEP_M_FINE": 0.1,
        "GROUND_ENABLE": True,
        "GROUND_RANSAC_DZ": 0.12,
        "RANGE_MIN_M": 3.0,
        "RANGE_MAX_M": 45.0,
        "INTENSITY_AUTO_SCALE": True,
        "INTENSITY_USABLE_NONZERO_MIN": 0.05,
        "INTENSITY_MIN_DYNAMIC_RANGE": 2000,
        "EXPORT_MAX_POINTS": 3000000,
        "OVERWRITE": True,
    }
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys: {missing}")
    params_hash = _write_resolved(run_dir, cfg)
    return cfg, params_hash


def _auto_find_kitti_root(cfg: Dict[str, object], scans: List[str]) -> Optional[Path]:
    env_root = str(cfg.get("KITTI_ROOT") or "").strip()
    if env_root:
        scans.append(env_root)
        p = Path(env_root)
        if p.exists():
            return p
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


def _frame_ids(start: int, end: int, stride: int) -> List[str]:
    return [f"{i:010d}" for i in range(int(start), int(end) + 1, max(1, int(stride)))]


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
        if "crosswalk" in row["name"].lower() and is_poly(row):
            return row["name"], "name_contains_crosswalk_polygon"
    for row in layers:
        if "truth" in row["name"].lower() and is_poly(row):
            return row["name"], "name_contains_truth_polygon"
    for row in layers:
        if is_poly(row):
            return row["name"], "first_polygon_layer"
    return None, "no_polygon_layer"


def _intensity_map(raw: np.ndarray, auto_scale: bool) -> Tuple[np.ndarray, str]:
    if raw.size == 0:
        return raw.astype(np.uint16), "empty"
    if not auto_scale:
        return np.clip(raw, 0, 65535).astype(np.uint16), "no_scale"
    if raw.dtype.kind == "f":
        max_val = float(np.nanmax(raw))
        if max_val <= 1.5:
            return np.round(np.clip(raw, 0.0, 1.0) * 65535.0).astype(np.uint16), "float01_to_uint16"
        if max_val <= 255.0:
            return np.round(np.clip(raw, 0.0, 255.0) * 256.0).astype(np.uint16), "float255_to_uint16"
    if raw.dtype.kind in {"i", "u"}:
        max_val = float(np.max(raw))
        if max_val <= 255.0:
            return (raw.astype(np.uint16) * 256).astype(np.uint16), "uint8_to_uint16"
    return np.clip(raw, 0, 65535).astype(np.uint16), "uint16_direct"


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
    nz = float(np.mean(vals > 0.0))
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p50": p50,
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "p99": p99,
        "nonzero_ratio": nz,
        "dynamic_range": float(p99 - p50),
    }


def _hist(inten: np.ndarray, bins: int = 256) -> np.ndarray:
    if inten.size == 0:
        return np.zeros((bins,), dtype=np.int64)
    hist, _ = np.histogram(inten.astype(np.float64), bins=bins, range=(0, 65535))
    return hist.astype(np.int64)


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


def _ransac_plane(
    points: np.ndarray,
    iters: int,
    dist_thresh: float,
    normal_min_z: float = 0.9,
) -> Tuple[Optional[np.ndarray], Optional[float], np.ndarray]:
    if points.shape[0] < 3:
        return None, None, np.zeros((points.shape[0],), dtype=bool)
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


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    var_a = float(np.var(a))
    var_b = float(np.var(b))
    n_a = float(a.size)
    n_b = float(b.size)
    pooled = ((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / max(n_a + n_b - 2.0, 1.0)
    if pooled <= 1e-12:
        return 0.0
    return (mean_a - mean_b) / math.sqrt(pooled)


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


def _shift_grid(shift_max: float, step: float) -> List[Tuple[float, float]]:
    vals = np.arange(-float(shift_max), float(shift_max) + 1e-6, float(step))
    shifts = [(float(dx), float(dy)) for dx in vals for dy in vals]
    return shifts


def _eval_shift(
    points_xy: np.ndarray,
    intensity: np.ndarray,
    truth_poly: object,
    bg_ring: object,
    dx: float,
    dy: float,
) -> Dict[str, object]:
    truth_s = translate(truth_poly, xoff=dx, yoff=dy)
    bg_s = translate(bg_ring, xoff=dx, yoff=dy)
    inside_mask = _mask_points_in_polygon(points_xy, truth_s)
    bg_mask = _mask_points_in_polygon(points_xy, bg_s)
    inside = intensity[inside_mask]
    bg = intensity[bg_mask]
    inside_s, bg_samp = _balanced_sample(inside, bg)
    auc = _auc_score(inside_s, bg_samp)
    d_score = _cohen_d(inside, bg)
    inside_p95 = float(np.percentile(inside, 95)) if inside.size else 0.0
    bg_p95 = float(np.percentile(bg, 95)) if bg.size else 0.0
    return {
        "dx": float(dx),
        "dy": float(dy),
        "inside_n": int(inside.size),
        "bg_n": int(bg.size),
        "inside_p95": inside_p95,
        "bg_p95": bg_p95,
        "delta_p95": float(inside_p95 - bg_p95),
        "auc": float(auc),
        "d_score": float(d_score),
    }


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


def _plot_bev(points_xy: np.ndarray, intensity: np.ndarray, mask_inside: np.ndarray, mask_bg: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if points_xy.size == 0:
        return
    rng = np.random.default_rng(0)
    idx = np.arange(points_xy.shape[0])
    if idx.size > 250000:
        idx = rng.choice(idx, size=250000, replace=False)
    pts = points_xy[idx]
    inten = intensity[idx]
    inside = mask_inside[idx]
    bg = mask_bg[idx]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    if np.any(bg):
        ax.scatter(pts[bg, 0], pts[bg, 1], s=0.5, c=inten[bg], cmap="viridis", alpha=0.25)
    if np.any(inside):
        ax.scatter(pts[inside, 0], pts[inside, 1], s=0.8, c=inten[inside], cmap="inferno", alpha=0.65)
    ax.set_title("BEV intensity: inside vs bg")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_shift_heatmap(rows: List[Dict[str, object]], out_path: Path, use_key: str, best_dx: float, best_dy: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dxs = sorted({float(r["dx"]) for r in rows})
    dys = sorted({float(r["dy"]) for r in rows})
    if not dxs or not dys:
        return
    grid = np.full((len(dys), len(dxs)), np.nan, dtype=np.float64)
    dx_idx = {v: i for i, v in enumerate(dxs)}
    dy_idx = {v: i for i, v in enumerate(dys)}
    for r in rows:
        grid[dy_idx[float(r["dy"])], dx_idx[float(r["dx"])]] = float(r.get(use_key, 0.0))
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    im = ax.imshow(grid, origin="lower", cmap="viridis", extent=(min(dxs), max(dxs), min(dys), max(dys)))
    ax.scatter([best_dx], [best_dy], c="red", s=20)
    ax.set_title(f"shift heatmap ({use_key})")
    ax.set_xlabel("dx (m)")
    ax.set_ylabel("dy (m)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _downsample_for_export(points: np.ndarray, intensity: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray, str]:
    if points.size == 0:
        return points, intensity, "empty"
    if points.shape[0] <= int(max_points):
        return points, intensity, "none"
    voxel = 0.08
    pts_v, inten_v = _voxel_downsample(points, intensity, voxel)
    if pts_v.shape[0] <= int(max_points):
        return pts_v, inten_v, f"voxel_{voxel:.2f}"
    rng = np.random.default_rng(0)
    idx = rng.choice(pts_v.shape[0], size=int(max_points), replace=False)
    return pts_v[idx], inten_v[idx], "random"


def _intensity_field(raw: np.ndarray) -> np.ndarray:
    if raw.ndim == 2 and raw.shape[1] >= 4:
        return raw[:, 3]
    if raw.dtype.fields:
        for key in ("intensity", "reflectance"):
            if key in raw.dtype.fields:
                return raw[key]
    raise RuntimeError("intensity_field_missing")

def main() -> int:
    base_cfg = _load_yaml(Path("configs/lidar_intensity_verify_0010_f280_300.yaml"))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"lidar_intensity_verify_0010_f280_300_{run_id}"
    if bool(base_cfg.get("OVERWRITE", True)):
        ensure_overwrite(run_dir)
    ensure_dir(run_dir)
    setup_logging(run_dir / "run.log")
    LOG.info("run_start")

    cfg, params_hash = _resolve_config(base_cfg, run_dir)
    notes: List[str] = []
    warnings: List[str] = []

    scans: List[str] = []
    data_root = _auto_find_kitti_root(cfg, scans)
    if data_root is None:
        write_text(run_dir / "report.md", "data_root_missing")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "data_root_missing", "scan_paths": scans})
        return 2

    drive_id = _select_drive_0010(data_root)
    if drive_id != "2013_05_28_drive_0010_sync":
        warnings.append(f"drive_id_not_exact:{drive_id}")

    velodyne_dir = _find_velodyne_dir(data_root, drive_id)
    frame_start = int(cfg["FRAME_START"])
    frame_end = int(cfg["FRAME_END"])
    stride = int(cfg["STRIDE"])
    expected_frames = list(range(frame_start, frame_end + 1, max(1, stride)))
    available = {int(p.stem) for p in velodyne_dir.glob("*.bin") if p.stem.isdigit()}
    missing = [f for f in expected_frames if f not in available]
    if missing:
        write_text(run_dir / "report.md", f"missing_frames:{missing}")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "missing_frames", "missing": missing})
        return 2
    frame_ids = _frame_ids(frame_start, frame_end, stride)

    truth_path = Path(str(cfg["TRUTH_GPKG"]))
    truth_layer = str(cfg["TRUTH_LAYER"])
    if not truth_path.exists():
        write_text(run_dir / "report.md", f"truth_missing:{truth_path}")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "truth_missing"})
        return 2

    layers = _list_layers(truth_path)
    layer_names = {row["name"] for row in layers}
    resolved_layer = truth_layer if truth_layer in layer_names else None
    resolve_reason = "config_layer_exists" if resolved_layer else "config_layer_missing"
    if resolved_layer is None:
        resolved_layer, resolve_reason = _pick_truth_layer(layers)
    cfg["truth_layer_resolved"] = resolved_layer or ""
    _dump_resolved(run_dir, cfg)
    if resolved_layer is None:
        report = [
            "# LiDAR intensity verify 0010 f280-300",
            "",
            "- status: FAIL",
            "- reason: truth_polygon_layer_missing",
            f"- truth_gpkg: {truth_path}",
            f"- truth_layer_config: {truth_layer}",
            "- layers:",
        ]
        report += [f"- {row['name']} ({row['geometry_type']})" for row in layers]
        write_text(run_dir / "report.md", "\n".join(report))
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "truth_polygon_layer_missing"})
        return 2

    gdf = gpd.read_file(truth_path, layer=resolved_layer)
    LOG.info("truth_loaded: layer=%s rows=%s", resolved_layer, int(gdf.shape[0]))
    if gdf.empty:
        write_text(run_dir / "report.md", "truth_empty")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "truth_empty"})
        return 2
    if gdf.crs is None or str(gdf.crs.to_epsg()) != "32632":
        gdf = gdf.to_crs(epsg=32632)
    truth_poly = _truth_geom(gdf)

    gis_dir = ensure_dir(run_dir / "gis")
    write_gpkg_layer(gis_dir / "truth_original_utm32.gpkg", resolved_layer, gdf, 32632, warnings, overwrite=True)

    roi_poly = truth_poly.buffer(float(cfg["ROI_PAD_M"]))
    bg_inner = truth_poly.buffer(float(cfg["BG_INNER_M"]))
    bg_outer = truth_poly.buffer(float(cfg["BG_OUTER_M"]))
    bg_ring = bg_outer.difference(bg_inner)
    roi_gdf = gpd.GeoDataFrame(
        [{"kind": "roi", "geometry": roi_poly}, {"kind": "bg_ring", "geometry": bg_ring}],
        geometry="geometry",
        crs="EPSG:32632",
    )
    write_gpkg_layer(gis_dir / "roi_utm32.gpkg", "roi", roi_gdf, 32632, warnings, overwrite=True)
    LOG.info("roi_ready")

    roi_bounds = roi_poly.bounds
    range_min = float(cfg["RANGE_MIN_M"])
    range_max = float(cfg["RANGE_MAX_M"])
    ground_enable = bool(cfg["GROUND_ENABLE"])
    ground_dz = float(cfg["GROUND_RANSAC_DZ"])

    roi_points: List[np.ndarray] = []
    roi_intensity_raw: List[np.ndarray] = []
    roi_ground_points: List[np.ndarray] = []
    roi_ground_intensity_raw: List[np.ndarray] = []
    frame_errors: List[str] = []
    ground_fallback = 0
    frame_sampled = 0

    for idx_f, fid in enumerate(frame_ids):
        try:
            raw = load_kitti360_lidar_points(data_root, drive_id, fid)
            world = load_kitti360_lidar_points_world_full(data_root, drive_id, fid, cam_id="image_00")
            if raw.size == 0 or world.size == 0:
                continue
            intensity_raw = _intensity_field(raw).astype(np.float32)
            xyz_velo = raw[:, :3].astype(np.float32)
            dist = np.linalg.norm(xyz_velo, axis=1)
            in_range = (dist >= range_min) & (dist <= range_max)
            if not np.any(in_range):
                continue
            world = world[in_range]
            xyz_velo = xyz_velo[in_range]
            intensity_raw = intensity_raw[in_range]

            bbox_mask = (
                (world[:, 0] >= roi_bounds[0])
                & (world[:, 0] <= roi_bounds[2])
                & (world[:, 1] >= roi_bounds[1])
                & (world[:, 1] <= roi_bounds[3])
            )
            if not np.any(bbox_mask):
                continue
            world = world[bbox_mask]
            xyz_velo = xyz_velo[bbox_mask]
            intensity_raw = intensity_raw[bbox_mask]

            roi_mask = _mask_points_in_polygon(world[:, :2], roi_poly)
            if not np.any(roi_mask):
                continue
            world = world[roi_mask]
            xyz_velo = xyz_velo[roi_mask]
            intensity_raw = intensity_raw[roi_mask]
            if world.shape[0] > 300000:
                rng = np.random.default_rng(0)
                sel = rng.choice(world.shape[0], size=300000, replace=False)
                world = world[sel]
                xyz_velo = xyz_velo[sel]
                intensity_raw = intensity_raw[sel]
                frame_sampled += 1

            roi_points.append(world.astype(np.float32))
            roi_intensity_raw.append(intensity_raw.astype(np.float32))

            if ground_enable and world.size > 0:
                n, d, inliers = _ransac_plane(xyz_velo, iters=200, dist_thresh=ground_dz, normal_min_z=0.9)
                if n is None or d is None or not np.any(inliers):
                    ground_fallback += 1
                    z = xyz_velo[:, 2]
                    z_ref = float(np.percentile(z, 10))
                    inliers = np.abs(z - z_ref) <= ground_dz
                if np.any(inliers):
                    roi_ground_points.append(world[inliers].astype(np.float32))
                    roi_ground_intensity_raw.append(intensity_raw[inliers].astype(np.float32))
            else:
                roi_ground_points.append(world.astype(np.float32))
                roi_ground_intensity_raw.append(intensity_raw.astype(np.float32))
        except Exception as exc:
            frame_errors.append(f"{fid}:{exc}")
        if (idx_f + 1) % 5 == 0:
            LOG.info("frame_progress: %s/%s", idx_f + 1, len(frame_ids))

    if not roi_points:
        write_text(run_dir / "report.md", "roi_points_empty")
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "roi_points_empty", "errors": frame_errors[:5]})
        return 2

    roi_points_all = np.vstack(roi_points)
    roi_intensity_raw_all = np.concatenate(roi_intensity_raw)
    roi_ground_points_all = np.vstack(roi_ground_points) if roi_ground_points else np.empty((0, 3), dtype=np.float32)
    roi_ground_intensity_raw_all = (
        np.concatenate(roi_ground_intensity_raw) if roi_ground_intensity_raw else np.empty((0,), dtype=np.float32)
    )

    LOG.info("points_accumulated: roi=%s ground=%s", int(roi_points_all.shape[0]), int(roi_ground_points_all.shape[0]))
    mapped_all, map_rule = _intensity_map(roi_intensity_raw_all, bool(cfg["INTENSITY_AUTO_SCALE"]))
    intensity_stats = _intensity_stats(mapped_all)
    intensity_usable = (
        intensity_stats["nonzero_ratio"] >= float(cfg["INTENSITY_USABLE_NONZERO_MIN"])
        and intensity_stats["dynamic_range"] >= float(cfg["INTENSITY_MIN_DYNAMIC_RANGE"])
    )
    if not intensity_usable:
        notes.append("intensity_not_usable")

    mapped_ground, _ = _intensity_map(roi_ground_intensity_raw_all, bool(cfg["INTENSITY_AUTO_SCALE"]))

    tables_dir = ensure_dir(run_dir / "tables")
    write_json(
        tables_dir / "intensity_stats.json",
        {
            "mapping_rule": map_rule,
            "stats_roi": intensity_stats,
            "raw_min": float(np.min(roi_intensity_raw_all)) if roi_intensity_raw_all.size else 0.0,
            "raw_max": float(np.max(roi_intensity_raw_all)) if roi_intensity_raw_all.size else 0.0,
            "raw_mean": float(np.mean(roi_intensity_raw_all)) if roi_intensity_raw_all.size else 0.0,
            "raw_dtype": str(roi_intensity_raw_all.dtype),
            "intensity_usable": bool(intensity_usable),
        },
    )

    points_xy_ground = roi_ground_points_all[:, :2] if roi_ground_points_all.size else np.empty((0, 2), dtype=np.float32)
    shift_points_xy = points_xy_ground
    shift_intensity = mapped_ground
    shift_sample_note = "none"
    if points_xy_ground.shape[0] > 200000:
        rng = np.random.default_rng(0)
        sel = rng.choice(points_xy_ground.shape[0], size=200000, replace=False)
        shift_points_xy = points_xy_ground[sel]
        shift_intensity = mapped_ground[sel]
        shift_sample_note = f"random_{shift_points_xy.shape[0]}"
    shift_rows: List[Dict[str, object]] = []
    coarse_shifts = _shift_grid(float(cfg["SHIFT_MAX_M"]), float(cfg["SHIFT_STEP_M_COARSE"]))
    LOG.info("shift_search_start: points=%s", int(shift_points_xy.shape[0]))
    for dx, dy in coarse_shifts:
        row = _eval_shift(shift_points_xy, shift_intensity, truth_poly, bg_ring, dx, dy)
        shift_rows.append(row)

    sort_key = lambda r: (-float(r.get("auc", 0.0)), -float(r.get("delta_p95", 0.0)))
    top5 = sorted(shift_rows, key=sort_key)[:5]
    fine_step = float(cfg["SHIFT_STEP_M_FINE"])
    fine_rows: List[Dict[str, object]] = []
    seen = set((float(r["dx"]), float(r["dy"])) for r in shift_rows)
    for r in top5:
        base_dx = float(r["dx"])
        base_dy = float(r["dy"])
        fine_vals = np.arange(-1.0, 1.0 + 1e-6, fine_step)
        for fx in fine_vals:
            for fy in fine_vals:
                dx = float(base_dx + fx)
                dy = float(base_dy + fy)
                key = (dx, dy)
                if key in seen:
                    continue
                seen.add(key)
                row = _eval_shift(shift_points_xy, shift_intensity, truth_poly, bg_ring, dx, dy)
                fine_rows.append(row)

    all_rows = shift_rows + fine_rows
    all_rows_sorted = sorted(all_rows, key=sort_key)
    best = all_rows_sorted[0] if all_rows_sorted else {"dx": 0.0, "dy": 0.0, "auc": 0.5, "delta_p95": 0.0, "d_score": 0.0}
    best_dx = float(best.get("dx", 0.0))
    best_dy = float(best.get("dy", 0.0))
    LOG.info("shift_search_done: best_dx=%.2f best_dy=%.2f", best_dx, best_dy)

    write_csv(
        tables_dir / "shift_search.csv",
        all_rows,
        ["dx", "dy", "inside_n", "bg_n", "inside_p95", "bg_p95", "delta_p95", "auc", "d_score"],
    )

    best_truth = translate(truth_poly, xoff=best_dx, yoff=best_dy)
    gdf_best = gpd.GeoDataFrame([{"geometry": best_truth}], geometry="geometry", crs="EPSG:32632")
    write_gpkg_layer(gis_dir / "truth_aligned_utm32.gpkg", resolved_layer, gdf_best, 32632, warnings, overwrite=True)

    inside_mask = _mask_points_in_polygon(points_xy_ground, best_truth)
    bg_mask = _mask_points_in_polygon(points_xy_ground, translate(bg_ring, xoff=best_dx, yoff=best_dy))
    inside_int = mapped_ground[inside_mask]
    bg_int = mapped_ground[bg_mask]

    inside_stats = _intensity_stats(inside_int)
    bg_stats = _intensity_stats(bg_int)
    auc_final = _auc_score(*_balanced_sample(inside_int, bg_int))
    d_final = _cohen_d(inside_int, bg_int)
    delta_p95 = float(inside_stats["p95"] - bg_stats["p95"])

    intensity_for_markings = (
        auc_final >= 0.65 and d_final >= 0.5 and delta_p95 >= 800.0
    )

    status = "PASS" if intensity_for_markings else "WARN"
    if not intensity_usable:
        status = "FAIL"

    decision = {
        "status": status,
        "intensity_usable": bool(intensity_usable),
        "intensity_usable_for_markings": bool(intensity_for_markings),
        "best_shift": {"dx": best_dx, "dy": best_dy},
        "metrics": {
            "auc": float(auc_final),
            "d_score": float(d_final),
            "delta_p95": float(delta_p95),
            "inside_stats": inside_stats,
            "bg_stats": bg_stats,
        },
        "notes": notes,
        "params_hash": params_hash,
    }
    write_json(run_dir / "decision.json", decision)

    pc_dir = ensure_dir(run_dir / "pointcloud")
    roi_points_out, roi_inten_out, roi_ds = _downsample_for_export(
        roi_points_all.astype(np.float32), mapped_all.astype(np.uint16), int(cfg["EXPORT_MAX_POINTS"])
    )
    ground_points_out, ground_inten_out, ground_ds = _downsample_for_export(
        roi_ground_points_all.astype(np.float32), mapped_ground.astype(np.uint16), int(cfg["EXPORT_MAX_POINTS"])
    )

    write_las(
        pc_dir / "roi_points_utm32.laz",
        roi_points_out,
        roi_inten_out,
        np.ones((roi_points_out.shape[0],), dtype=np.uint8),
        32632,
    )
    write_las(
        pc_dir / "roi_ground_points_utm32.laz",
        ground_points_out,
        ground_inten_out,
        np.full((ground_points_out.shape[0],), 2, dtype=np.uint8),
        32632,
    )

    inside_pts = roi_ground_points_all[inside_mask]
    bg_pts = roi_ground_points_all[bg_mask]
    write_las(
        pc_dir / "truth_inside_ground_utm32.laz",
        inside_pts.astype(np.float32),
        inside_int.astype(np.uint16),
        np.full((inside_pts.shape[0],), 2, dtype=np.uint8),
        32632,
    )
    write_las(
        pc_dir / "bg_ring_ground_utm32.laz",
        bg_pts.astype(np.float32),
        bg_int.astype(np.uint16),
        np.full((bg_pts.shape[0],), 2, dtype=np.uint8),
        32632,
    )

    if inside_int.size:
        top_n = max(1, int(inside_int.size * 0.02))
        top_n = min(top_n, 50000)
        idx = np.argsort(inside_int)[-top_n:]
        cand_pts = inside_pts[idx]
        cand_int = inside_int[idx]
    else:
        cand_pts = np.empty((0, 3), dtype=np.float32)
        cand_int = np.empty((0,), dtype=np.uint16)
    write_las(
        pc_dir / "marking_candidate_ground_utm32.laz",
        cand_pts.astype(np.float32),
        cand_int.astype(np.uint16),
        np.full((cand_pts.shape[0],), 2, dtype=np.uint8),
        32632,
    )

    inside_hist = _hist(inside_int)
    bg_hist = _hist(bg_int)
    write_csv(
        tables_dir / "intensity_hist_inside.csv",
        [{"bin": i, "count": int(c)} for i, c in enumerate(inside_hist)],
        ["bin", "count"],
    )
    write_csv(
        tables_dir / "intensity_hist_bg.csv",
        [{"bin": i, "count": int(c)} for i, c in enumerate(bg_hist)],
        ["bin", "count"],
    )

    img_dir = ensure_dir(run_dir / "images")
    _plot_hist_compare(inside_int, bg_int, img_dir / "hist_compare.png")
    _plot_bev(points_xy_ground, mapped_ground, inside_mask, bg_mask, img_dir / "bev_intensity_inside_bg.png")
    _plot_shift_heatmap(shift_rows, img_dir / "shift_heatmap.png", "auc", best_dx, best_dy)

    report = [
        "# LiDAR intensity verify 0010 f280-300",
        "",
        f"- drive_id: {drive_id}",
        f"- frames: {frame_start}-{frame_end}",
        f"- data_root: {data_root}",
        f"- params_hash: {params_hash}",
        f"- intensity_map_rule: {map_rule}",
        f"- truth_layer_config: {truth_layer}",
        f"- truth_layer_resolved: {resolved_layer}",
        f"- truth_layer_resolve_reason: {resolve_reason}",
        f"- intensity_usable: {bool(intensity_usable)}",
        f"- intensity_nonzero_ratio: {intensity_stats['nonzero_ratio']:.4f}",
        f"- intensity_dynamic_range: {intensity_stats['dynamic_range']:.1f}",
        f"- best_shift: dx={best_dx:.2f}, dy={best_dy:.2f}",
        f"- AUC: {auc_final:.4f}",
        f"- d_score: {d_final:.3f}",
        f"- delta_p95: {delta_p95:.1f}",
        f"- intensity_usable_for_markings: {bool(intensity_for_markings)}",
        "",
        "## Outputs",
        f"- {relpath(run_dir, gis_dir / 'truth_original_utm32.gpkg')}",
        f"- {relpath(run_dir, gis_dir / 'truth_aligned_utm32.gpkg')}",
        f"- {relpath(run_dir, gis_dir / 'roi_utm32.gpkg')}",
        f"- {relpath(run_dir, pc_dir / 'roi_points_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'roi_ground_points_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'truth_inside_ground_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'bg_ring_ground_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'marking_candidate_ground_utm32.laz')}",
        f"- {relpath(run_dir, tables_dir / 'shift_search.csv')}",
        f"- {relpath(run_dir, tables_dir / 'intensity_stats.json')}",
        f"- {relpath(run_dir, tables_dir / 'intensity_hist_inside.csv')}",
        f"- {relpath(run_dir, tables_dir / 'intensity_hist_bg.csv')}",
        f"- {relpath(run_dir, img_dir / 'hist_compare.png')}",
        f"- {relpath(run_dir, img_dir / 'bev_intensity_inside_bg.png')}",
        f"- {relpath(run_dir, img_dir / 'shift_heatmap.png')}",
        "",
        "## Notes",
        f"- ground_fallback_frames: {ground_fallback}",
        f"- export_downsample_roi: {roi_ds}",
        f"- export_downsample_ground: {ground_ds}",
        f"- shift_search_sample: {shift_sample_note}",
        f"- frame_random_sampled: {frame_sampled}",
    ]
    report.extend(["", "## Truth Layers"])
    report.extend([f"- {row['name']} ({row['geometry_type']})" for row in layers])
    if warnings:
        report.extend(["", "## Warnings"] + [f"- {w}" for w in warnings])
    if frame_errors:
        report.extend(["", "## Frame Errors"] + [f"- {e}" for e in frame_errors[:10]])
    write_text(run_dir / "report.md", "\n".join(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

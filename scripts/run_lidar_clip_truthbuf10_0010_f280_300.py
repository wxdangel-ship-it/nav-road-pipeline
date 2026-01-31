from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

from pipeline._io import load_yaml
from pipeline.calib.kitti360_world import transform_points_V_to_W
from pipeline.datasets.kitti360_io import _find_velodyne_dir, _resolve_velodyne_path, load_kitti360_lidar_points
from pipeline.lidar_semantic.export_pointcloud import write_las
from scripts.pipeline_common import ensure_overwrite, now_ts, relpath, setup_logging, write_csv, write_gpkg_layer, write_json, write_text


LOG = logging.getLogger("lidar_clip_truthbuf10_0010_f280_300")

TRUTH_GPKG = Path(r"E:\Work\nav-road-pipeline\crosswalk_truth_utm32.gpkg")

REQUIRED_KEYS = [
    "FRAME_START",
    "FRAME_END",
    "BUFFER_M",
    "TARGET_EPSG",
    "HIGH_INT_PCTL",
    "MAX_EXPORT_POINTS",
    "OVERWRITE",
]


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


def _intensity_field(raw: np.ndarray) -> Optional[np.ndarray]:
    if raw.ndim == 2 and raw.shape[1] >= 4:
        return raw[:, 3]
    if raw.dtype.fields:
        for key in ("intensity", "reflectance"):
            if key in raw.dtype.fields:
                return raw[key]
    return None


def _map_intensity(raw: Optional[np.ndarray]) -> Tuple[np.ndarray, str, bool]:
    if raw is None:
        return np.zeros((0,), dtype=np.uint16), "missing", True
    if raw.size == 0:
        return raw.astype(np.uint16), "empty", False
    if raw.dtype.kind == "f":
        min_val = float(np.nanmin(raw))
        max_val = float(np.nanmax(raw))
        if min_val >= 0.0 and max_val <= 1.0:
            scaled = np.round(np.clip(raw, 0.0, 1.0) * 65535.0).astype(np.uint16)
            return scaled, "float01_to_uint16", False
        if min_val >= 0.0 and max_val <= 255.0:
            scaled = np.round(np.clip(raw, 0.0, 255.0) * 256.0).astype(np.uint16)
            return scaled, "float255_to_uint16", False
        return np.clip(raw, 0, 65535).astype(np.uint16), "float_unexpected_clipped", False
    if raw.dtype.kind in {"u", "i"}:
        max_val = float(np.max(raw))
        if max_val <= 255.0:
            return (raw.astype(np.uint16) * 256).astype(np.uint16), "uint8_to_uint16", False
        return np.clip(raw, 0, 65535).astype(np.uint16), "uint16_direct", False
    return np.zeros((raw.shape[0],), dtype=np.uint16), "unsupported_dtype_zeroed", True


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


def _sample_for_export(points_xyz: np.ndarray, intensity: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray, str]:
    if points_xyz.size == 0:
        return points_xyz, intensity, "empty"
    if points_xyz.shape[0] <= max_points:
        return points_xyz, intensity, "none"
    rng = np.random.default_rng(0)
    idx = rng.choice(points_xyz.shape[0], size=int(max_points), replace=False)
    return points_xyz[idx], intensity[idx], f"random_{max_points}"


def _plot_bev(points_xy: np.ndarray, out_path: Path, title: str, res_m: float) -> None:
    if points_xy.size == 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(title)
        ax.set_axis_off()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return
    minx, miny = np.min(points_xy[:, 0]), np.min(points_xy[:, 1])
    maxx, maxy = np.max(points_xy[:, 0]), np.max(points_xy[:, 1])
    width = int(np.ceil((maxx - minx) / res_m)) + 1
    height = int(np.ceil((maxy - miny) / res_m)) + 1
    col = np.floor((points_xy[:, 0] - minx) / res_m).astype(np.int32)
    row = np.floor((maxy - points_xy[:, 1]) / res_m).astype(np.int32)
    valid = (col >= 0) & (row >= 0) & (col < width) & (row < height)
    col = col[valid]
    row = row[valid]
    img = np.zeros((height, width), dtype=np.float32)
    for r, c in zip(row, col):
        img[r, c] += 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, origin="upper", cmap="inferno")
    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg_path = Path("configs/lidar_clip_truthbuf10_0010_f280_300.yaml")
    cfg = _load_cfg(cfg_path)
    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise KeyError(f"Missing required key {key} in {cfg_path}")

    run_id = now_ts()
    run_dir = Path("runs") / f"lidar_clip_truthbuf10_0010_f280_300_{run_id}"
    if bool(cfg.get("OVERWRITE", True)):
        ensure_overwrite(run_dir)
    setup_logging(run_dir / "run.log")
    _write_resolved(run_dir, cfg)

    gis_dir = run_dir / "gis"
    pc_dir = run_dir / "pointcloud"
    tbl_dir = run_dir / "tables"
    img_dir = run_dir / "images"

    layers = _list_layers(TRUTH_GPKG)
    write_csv(tbl_dir / "layers.csv", layers, ["name", "geometry_type"])
    layer_name = _pick_truth_layer(layers)
    if layer_name is None:
        report = [
            "# LiDAR clip truth buffer10 0010 f280-300",
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
        truth_gdf = truth_gdf.set_crs(int(cfg["TARGET_EPSG"]))
    if int(truth_gdf.crs.to_epsg() or 0) != int(cfg["TARGET_EPSG"]):
        truth_gdf = truth_gdf.to_crs(int(cfg["TARGET_EPSG"]))
    truth_geom = _truth_geom(truth_gdf)
    truth_gdf = gpd.GeoDataFrame(geometry=[truth_geom], crs=truth_gdf.crs)
    write_gpkg_layer(gis_dir / "truth_selected_utm32.gpkg", "truth_selected", truth_gdf, int(cfg["TARGET_EPSG"]), [])
    truth_buf = truth_geom.buffer(float(cfg["BUFFER_M"]))
    buf_gdf = gpd.GeoDataFrame(geometry=[truth_buf], crs=truth_gdf.crs)
    write_gpkg_layer(gis_dir / "truth_buffer10_utm32.gpkg", "truth_buffer10", buf_gdf, int(cfg["TARGET_EPSG"]), [])

    scans: List[str] = []
    data_root = _auto_find_kitti_root(scans)
    if data_root is None:
        raise RuntimeError(f"data_root_not_found:scanned={scans}")
    drive_id = _select_drive_0010(data_root)

    frame_ids = _frame_ids(int(cfg["FRAME_START"]), int(cfg["FRAME_END"]))
    bbox = truth_buf.bounds
    bbox_poly = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])

    all_points: List[np.ndarray] = []
    all_intensity: List[np.ndarray] = []
    per_frame_rows: List[Dict[str, object]] = []
    intensity_rule = "unknown"
    intensity_missing = False

    _ = _find_velodyne_dir(data_root, drive_id)
    for fid in frame_ids:
        raw = load_kitti360_lidar_points(data_root, drive_id, fid)
        if raw.size == 0:
            per_frame_rows.append({"frame_id": fid, "clip_count": 0})
            continue
        intensity_raw = _intensity_field(raw)
        if intensity_raw is None:
            intensity_raw = np.zeros((raw.shape[0],), dtype=np.float32)
        else:
            intensity_raw = intensity_raw.astype(np.float32)
        mapped, rule, missing_flag = _map_intensity(intensity_raw)
        if intensity_rule == "unknown":
            intensity_rule = rule
        elif intensity_rule != rule:
            intensity_rule = "mixed"
        if missing_flag:
            intensity_missing = True
        pts_world = transform_points_V_to_W(raw[:, :3], data_root, drive_id, fid)
        finite_mask = np.isfinite(pts_world).all(axis=1)
        pts_world = pts_world[finite_mask]
        mapped = mapped[finite_mask]
        if pts_world.size == 0:
            per_frame_rows.append({"frame_id": fid, "clip_count": 0})
            continue
        bbox_mask = _mask_points_in_polygon(pts_world[:, :2], bbox_poly)
        pts_bbox = pts_world[bbox_mask]
        inten_bbox = mapped[bbox_mask]
        if pts_bbox.size == 0:
            per_frame_rows.append({"frame_id": fid, "clip_count": 0})
            continue
        poly_mask = _mask_points_in_polygon(pts_bbox[:, :2], truth_buf)
        pts_clip = pts_bbox[poly_mask]
        inten_clip = inten_bbox[poly_mask]
        per_frame_rows.append({"frame_id": fid, "clip_count": int(pts_clip.shape[0])})
        if pts_clip.size:
            all_points.append(pts_clip.astype(np.float32))
            all_intensity.append(inten_clip.astype(np.uint16))

    write_csv(tbl_dir / "per_frame_counts.csv", per_frame_rows, ["frame_id", "clip_count"])
    if not all_points:
        decision = {"status": "FAIL", "reason": "no_points_after_clip"}
        write_json(run_dir / "decision.json", decision)
        report = [
            "# LiDAR clip truth buffer10 0010 f280-300",
            "",
            "- status: FAIL",
            "- reason: no_points_after_clip",
            f"- per_frame_counts: {relpath(run_dir, tbl_dir / 'per_frame_counts.csv')}",
        ]
        write_text(run_dir / "report.md", "\n".join(report) + "\n")
        return

    points_all = np.concatenate(all_points, axis=0)
    intensity_all = np.concatenate(all_intensity, axis=0)

    max_export = int(cfg["MAX_EXPORT_POINTS"])
    pts_out, inten_out, ds = _sample_for_export(points_all, intensity_all, max_export)
    write_las(
        pc_dir / "clip_truthbuf10_utm32.laz",
        pts_out,
        inten_out,
        np.ones((pts_out.shape[0],), dtype=np.uint8),
        int(cfg["TARGET_EPSG"]),
    )

    intensity_stats = _intensity_stats(intensity_all)
    write_json(tbl_dir / "intensity_stats.json", intensity_stats)

    hi_thr = float(np.percentile(intensity_all.astype(np.float64), float(cfg["HIGH_INT_PCTL"]))) if intensity_all.size else 0.0
    hi_mask = intensity_all >= hi_thr
    hi_points = points_all[hi_mask]
    hi_intensity = intensity_all[hi_mask]
    hi_pts_out, hi_int_out, hi_ds = _sample_for_export(hi_points, hi_intensity, max_export)
    write_las(
        pc_dir / "clip_truthbuf10_highI_utm32.laz",
        hi_pts_out,
        hi_int_out,
        np.ones((hi_pts_out.shape[0],), dtype=np.uint8),
        int(cfg["TARGET_EPSG"]),
    )

    img_dir.mkdir(parents=True, exist_ok=True)
    _plot_bev(points_all[:, :2], img_dir / "bev_density.png", "BEV density", 0.2)
    _plot_bev(hi_points[:, :2], img_dir / "bev_highI.png", "BEV high intensity", 0.2)

    intensity_all_zero = intensity_stats["nonzero_ratio"] == 0.0
    status = "FAIL" if (points_all.size == 0 or intensity_all_zero) else "PASS"
    decision = {
        "status": status,
        "intensity_all_zero": bool(intensity_all_zero),
        "total_points": int(points_all.shape[0]),
        "high_intensity_threshold": float(hi_thr),
        "high_intensity_points": int(hi_points.shape[0]),
    }
    write_json(run_dir / "decision.json", decision)

    report = [
        "# LiDAR clip truth buffer10 0010 f280-300",
        "",
        f"- status: {status}",
        f"- input_drive: {drive_id}",
        f"- frames: {cfg['FRAME_START']}..{cfg['FRAME_END']}",
        f"- intensity_rule: {intensity_rule}",
        f"- intensity_missing: {bool(intensity_missing)}",
        f"- total_points: {points_all.shape[0]}",
        f"- highI_threshold: {hi_thr:.1f}",
        f"- highI_points: {hi_points.shape[0]}",
        f"- export_sampling_all: {ds}",
        f"- export_sampling_highI: {hi_ds}",
        "",
        "## outputs",
        f"- {relpath(run_dir, pc_dir / 'clip_truthbuf10_utm32.laz')}",
        f"- {relpath(run_dir, pc_dir / 'clip_truthbuf10_highI_utm32.laz')}",
        f"- {relpath(run_dir, img_dir / 'bev_density.png')}",
        f"- {relpath(run_dir, img_dir / 'bev_highI.png')}",
        f"- {relpath(run_dir, tbl_dir / 'intensity_stats.json')}",
        f"- {relpath(run_dir, tbl_dir / 'per_frame_counts.csv')}",
    ]
    write_text(run_dir / "report.md", "\n".join(report) + "\n")


if __name__ == "__main__":
    main()

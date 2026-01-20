from __future__ import annotations
from pathlib import Path
import argparse
import datetime
import json
import math
import os
import re
from typing import Iterable, Optional

import numpy as np
from shapely.geometry import LineString, Point, Polygon, MultiPolygon, box, mapping, shape
from shapely import affinity as shapely_affinity
from shapely.ops import unary_union, linemerge, transform as shapely_transform
from shapely.prepared import prep
from pyproj import Transformer
import yaml

from pipeline._io import ensure_dir, new_run_id
from pipeline.nn.model_registry import load_model as load_nn_model
from pipeline.sat_intersections import run_sat_intersections
from pipeline.intersection_shape import (
    arm_count as _arm_count,
    aspect_ratio as _intersection_aspect_ratio,
    circularity as _intersection_circularity,
    overlap_with_road as _intersection_overlap_with_road,
    refine_intersection_polygon as _refine_intersection_polygon,
)
from pipeline.centerlines_v2 import build_centerlines_v2
from pipeline.evidence.image_feature_provider import load_features as load_image_features

try:
    from shapely import make_valid as _shapely_make_valid
except Exception:
    try:
        from shapely.validation import make_valid as _shapely_make_valid
    except Exception:
        _shapely_make_valid = None


def _find_oxts_dir(data_root: Path, drive: str) -> Path:
    candidates = [
        data_root / "data_poses" / drive / "oxts" / "data",
        data_root / "data_poses" / "oxts" / drive / "oxts" / "data",
        data_root / "data_poses" / "oxts" / drive / "data",
        data_root / "data_poses_oxts" / drive / "oxts" / "data",
        data_root / "data_poses_oxts_extract" / drive / "oxts" / "data",
        data_root / drive / "oxts" / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"ERROR: oxts data not found for drive: {drive}")


def _read_latlon(oxts_dir: Path, max_frames: int) -> list[tuple[float, float]]:
    files = sorted(oxts_dir.glob("*.txt"))
    if max_frames and max_frames > 0:
        files = files[: max_frames]
    if not files:
        raise SystemExit(f"ERROR: no oxts txt files found in {oxts_dir}")
    pts: list[tuple[float, float]] = []
    for fp in files:
        text = fp.read_text(encoding="utf-8").strip()
        if not text:
            continue
        parts = re.split(r"\s+", text)
        if len(parts) < 6:
            continue
        try:
            lat = float(parts[0])
            lon = float(parts[1])
            yaw = float(parts[5])
        except ValueError:
            continue
        pts.append((lat, lon, yaw))
    if len(pts) < 2:
        raise SystemExit("ERROR: insufficient oxts points.")
    return pts


def _project_to_utm32(points: list[tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    lats = np.array([p[0] for p in points], dtype=float)
    lons = np.array([p[1] for p in points], dtype=float)
    yaws = np.array([p[2] for p in points], dtype=float)
    xs, ys = transformer.transform(lons, lats)
    return np.vstack([xs, ys]).T, yaws


def _find_velodyne_dir(data_root: Path, drive: str) -> Path:
    candidates = [
        data_root / "data_3d_raw" / drive / "velodyne_points" / "data",
        data_root / "data_3d_raw" / drive / "velodyne_points" / "data" / "1",
        data_root / drive / "velodyne_points" / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"ERROR: velodyne data not found for drive: {drive}")


def _find_image_dir(data_root: Path, drive: str, camera: str) -> Path:
    candidates = [
        data_root / "data_2d_raw" / drive / camera / "data_rect",
        data_root / "data_2d_raw" / drive / camera / "data",
        data_root / drive / camera / "data_rect",
        data_root / drive / camera / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"ERROR: image data not found for drive: {drive}, camera: {camera}")


def _read_calib_key(path: Path, key: str) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        if k.strip() != key:
            continue
        parts = [p for p in re.split(r"\s+", v.strip()) if p]
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            return None
        if len(vals) == 9:
            return np.asarray(vals, dtype=float).reshape(3, 3)
        if len(vals) == 12:
            return np.asarray(vals, dtype=float).reshape(3, 4)
        if len(vals) == 16:
            return np.asarray(vals, dtype=float).reshape(4, 4)
        return None
    return None


def _read_calib_matrix_file(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    parts = [p for p in re.split(r"\s+", text) if p]
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    if len(vals) == 9:
        return np.asarray(vals, dtype=float).reshape(3, 3)
    if len(vals) == 12:
        return np.asarray(vals, dtype=float).reshape(3, 4)
    if len(vals) == 16:
        return np.asarray(vals, dtype=float).reshape(4, 4)
    return None


def _find_calib_dir(data_root: Path, drive: str) -> Optional[Path]:
    candidates = [
        data_root / "calibration",
        data_root / "calib",
        data_root / "data_2d_raw" / drive / "calibration",
        data_root / "data_2d_raw" / "calibration",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_kitti360_calib(data_root: Path, drive: str, camera: str) -> dict:
    calib_dir = _find_calib_dir(data_root, drive)
    if calib_dir is None:
        raise RuntimeError("Calibration directory not found.")
    cam_id = camera.split("_")[-1]

    cam_to_pose = calib_dir / "calib_cam_to_pose.txt"
    cam_to_velo = calib_dir / "calib_cam_to_velo.txt"
    cam_to_cam = calib_dir / "calib_cam_to_cam.txt"
    perspective = calib_dir / "perspective.txt"

    t_cam_to_velo = _read_calib_key(cam_to_velo, camera)
    if t_cam_to_velo is None and cam_to_velo.exists():
        t_cam_to_velo = _read_calib_key(cam_to_velo, f"image_{cam_id}")
    if t_cam_to_velo is None and cam_to_velo.exists():
        t_cam_to_velo = _read_calib_matrix_file(cam_to_velo)
    if t_cam_to_velo is None:
        raise RuntimeError("Missing camera-to-velodyne calibration.")

    t_cam_to_velo = _ensure_4x4(t_cam_to_velo)
    t_velo_to_cam = np.linalg.inv(t_cam_to_velo)

    r_rect = None
    if cam_to_cam.exists():
        r_rect = _read_calib_key(cam_to_cam, f"R_rect_{cam_id}")
    if r_rect is None and perspective.exists():
        r_rect = _read_calib_key(perspective, f"R_rect_{cam_id}")
    if r_rect is None:
        r_rect = np.eye(3, dtype=float)
    r_rect = _ensure_4x4(r_rect)

    p_rect = None
    if cam_to_cam.exists():
        p_rect = _read_calib_key(cam_to_cam, f"P_rect_{cam_id}")
        if p_rect is None:
            p_rect = _read_calib_key(cam_to_cam, f"K_{cam_id}")
    if p_rect is None and perspective.exists():
        p_rect = _read_calib_key(perspective, f"P_rect_{cam_id}")
        if p_rect is None:
            p_rect = _read_calib_key(perspective, f"K_{cam_id}")
    if p_rect is None:
        raise RuntimeError("Missing camera intrinsics.")

    if p_rect.shape == (3, 3):
        p_rect = np.hstack([p_rect, np.zeros((3, 1), dtype=float)])
    if p_rect.shape != (3, 4):
        raise RuntimeError("Unexpected projection matrix shape.")

    return {
        "calib_dir": str(calib_dir),
        "t_velo_to_cam": t_velo_to_cam,
        "r_rect": r_rect,
        "p_rect": p_rect,
        "cam_to_pose": str(cam_to_pose) if cam_to_pose.exists() else None,
        "cam_to_velo": str(cam_to_velo) if cam_to_velo.exists() else None,
        "cam_to_cam": str(cam_to_cam) if cam_to_cam.exists() else None,
        "perspective": str(perspective) if perspective.exists() else None,
    }


def _ensure_4x4(mat: np.ndarray) -> np.ndarray:
    if mat.shape == (4, 4):
        return mat.astype(float)
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=float)
        out[:3, :4] = mat
        return out
    if mat.shape == (3, 3):
        out = np.eye(4, dtype=float)
        out[:3, :3] = mat
        return out
    raise RuntimeError(f"Unexpected matrix shape: {mat.shape}")


def _read_velodyne_points(fp: Path) -> np.ndarray:
    raw = np.fromfile(str(fp), dtype=np.float32)
    if raw.size % 4 != 0:
        raw = raw[: raw.size - (raw.size % 4)]
    return raw.reshape(-1, 4)


def _ground_mask(z: np.ndarray, z_band: float = 0.6) -> np.ndarray:
    if z.size == 0:
        return z.astype(bool)
    med = float(np.median(z))
    return np.abs(z - med) <= z_band


def _grid_counts(
    xs: np.ndarray,
    ys: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
) -> dict[tuple[int, int], int]:
    dx = xs - origin[0]
    dy = ys - origin[1]
    ix = np.floor(dx / resolution).astype(np.int64)
    iy = np.floor(dy / resolution).astype(np.int64)
    keys = np.stack([ix, iy], axis=1)
    uniq, counts = np.unique(keys, axis=0, return_counts=True)
    out: dict[tuple[int, int], int] = {}
    for k, c in zip(uniq, counts):
        key = (int(k[0]), int(k[1]))
        out[key] = out.get(key, 0) + int(c)
    return out


def _merge_counts(acc: dict[tuple[int, int], int], add: dict[tuple[int, int], int]) -> None:
    for k, v in add.items():
        acc[k] = acc.get(k, 0) + v


def _merge_lines(geom) -> LineString | None:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return geom
    if geom.geom_type == "MultiLineString":
        merged = linemerge(geom)
        if merged.geom_type == "LineString":
            return merged
        if merged.geom_type == "MultiLineString":
            lines = list(merged.geoms)
            if not lines:
                return None
            return max(lines, key=lambda g: g.length)
    return None


def _get_env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_env_str(key: str, default: str) -> str:
    raw = os.environ.get(key, "")
    return raw.strip() if raw else default


def _load_nn_best_cfg(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"ERROR: nn best config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data


def _load_centerlines_cfg(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "centerlines" in data:
        return data.get("centerlines") or {}
    return data


def _centerlines_defaults() -> dict:
    return {
        "mode": "both",
        "step_m": 5.0,
        "probe_m": 25.0,
        "dual_width_threshold_m": 12.0,
        "min_dual_sample_ratio": 0.6,
        "min_segment_length_m": 30.0,
        "dual_offset_mode": "fixed",
        "dual_offset_m": 3.5,
        "dual_offset_max_m": 8.0,
        "dual_offset_margin_m": 1.0,
        "dual_min_keep_ratio": 0.6,
        "dual_fallback_single": True,
        "dual_conf_threshold": 0.0,
        "divider_sources": ["geom"],
        "simplify_m": 0.5,
        "debug_divider_layers": False,
    }


def _load_intersection_refine_cfg(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("shape_refine") or data


def _intersection_refine_defaults() -> dict:
    return {
        "radius_m": 18.0,
        "road_buffer_m": 1.0,
        "arm_length_m": 25.0,
        "arm_buffer_m": 6.0,
        "simplify_m": 0.5,
        "min_area_m2": 30.0,
        "min_part_area_m2": 30.0,
        "min_hole_area_m2": 30.0,
        "max_circularity": 0.85,
        "min_overlap_road": 0.7,
    }


def _fill_small_holes(geom, min_area: float):
    if min_area <= 0:
        return geom
    if geom.is_empty:
        return geom

    def _clean_poly(poly: Polygon) -> Polygon:
        keep = []
        for ring in poly.interiors:
            hole = Polygon(ring)
            if hole.area >= min_area:
                keep.append(ring)
        return Polygon(poly.exterior, keep)

    if isinstance(geom, Polygon):
        return _clean_poly(geom)
    if isinstance(geom, MultiPolygon):
        polys = [_clean_poly(p) for p in geom.geoms]
        polys = [p for p in polys if not p.is_empty]
        return MultiPolygon(polys) if polys else geom
    return geom


def _smooth_geom(geom, buf_m: float):
    if buf_m <= 0:
        return geom
    if geom.is_empty:
        return geom
    return geom.buffer(buf_m).buffer(-buf_m)


def _close_open_geom(geom, close_m: float, open_m: float):
    if close_m > 0:
        geom = geom.buffer(close_m).buffer(-close_m)
    if open_m > 0:
        geom = geom.buffer(-open_m).buffer(open_m)
    return geom


def _make_valid_geom(geom):
    if geom is None or geom.is_empty:
        return geom
    if _shapely_make_valid is None:
        return geom.buffer(0)
    try:
        fixed = _shapely_make_valid(geom)
    except Exception:
        fixed = geom.buffer(0)
    return fixed


def _remove_small_parts(geom, min_area: float):
    if min_area <= 0:
        return geom
    polys = _explode_polygons(geom)
    polys = [p for p in polys if p.area >= min_area]
    if not polys:
        return geom
    return unary_union(polys)


def _polygon_metrics(geom) -> dict:
    if geom is None or geom.is_empty:
        return {
            "area_m2": 0.0,
            "perimeter_m": 0.0,
            "roughness": 0.0,
            "vertex_count": 0,
            "holes_count": 0,
            "part_count": 0,
        }
    area = float(geom.area)
    perim = float(geom.length)
    return {
        "area_m2": round(area, 3),
        "perimeter_m": round(perim, 3),
        "roughness": round(_polygon_roughness(perim, area), 6),
        "vertex_count": int(_polygon_vertex_count(geom)),
        "holes_count": int(_polygon_holes_count(geom)),
        "part_count": int(len(_explode_polygons(geom))),
    }


def _apply_strong_smoothing(
    geom,
    min_part_area: float,
    close_m: float,
    open_m: float,
    simplify_m: float,
    min_hole_area: float,
) -> tuple:
    before = _polygon_metrics(geom)
    out = _make_valid_geom(geom)
    out = _remove_small_parts(out, min_part_area)
    out = _close_open_geom(out, close_m, open_m)
    if simplify_m > 0:
        out = out.simplify(simplify_m, preserve_topology=True)
    out = _fill_small_holes(out, min_hole_area)
    out = _make_valid_geom(out)
    polys = _explode_polygons(out)
    if polys:
        out = unary_union(polys)
    after = _polygon_metrics(out)
    report = {
        "before": before,
        "after": after,
        "params": {
            "min_part_area_m2": float(min_part_area),
            "close_m": float(close_m),
            "open_m": float(open_m),
            "simplify_m": float(simplify_m),
            "min_hole_area_m2": float(min_hole_area),
        },
    }
    return out, report


def _offset_polyline_coords(coords: np.ndarray, offset: float) -> LineString | None:
    if coords.shape[0] < 2:
        return None
    normals = np.zeros_like(coords)
    for i in range(coords.shape[0] - 1):
        dx = coords[i + 1, 0] - coords[i, 0]
        dy = coords[i + 1, 1] - coords[i, 1]
        length = math.hypot(dx, dy)
        if length == 0:
            nx, ny = 0.0, 0.0
        else:
            nx, ny = -dy / length, dx / length
        normals[i] = (nx, ny)
    normals[-1] = normals[-2]
    avg = normals.copy()
    for i in range(1, coords.shape[0] - 1):
        nx = normals[i - 1, 0] + normals[i, 0]
        ny = normals[i - 1, 1] + normals[i, 1]
        length = math.hypot(nx, ny)
        if length != 0:
            avg[i] = (nx / length, ny / length)
    shifted = coords + avg * offset
    return LineString(shifted.tolist())


def _build_centerlines(line: LineString, road_poly, center_offset: float) -> list[LineString]:
    line_len = float(line.length)
    base = line.simplify(0.5)
    coords = np.asarray(base.coords)
    for offset in (center_offset, max(1.0, center_offset * 0.7), max(0.8, center_offset * 0.45)):
        left = _offset_polyline_coords(coords, offset)
        right = _offset_polyline_coords(coords, -offset)
        if left is None or right is None:
            continue
        if road_poly.contains(left) and road_poly.contains(right):
            return [left, right]
        left_clip = _merge_lines(left.intersection(road_poly))
        right_clip = _merge_lines(right.intersection(road_poly))
        if left_clip is None or right_clip is None:
            continue
        if left_clip.length >= 0.7 * line_len and right_clip.length >= 0.7 * line_len:
            return [left_clip, right_clip]
    return [line, line]


def _longest_line(geom) -> LineString | None:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return geom
    if geom.geom_type == "MultiLineString":
        lines = list(geom.geoms)
        if not lines:
            return None
        return max(lines, key=lambda g: g.length)
    return None


def _offset_line(line: LineString, offset: float) -> LineString | None:
    if line is None or line.is_empty:
        return None
    try:
        if hasattr(line, "offset_curve"):
            out = line.offset_curve(offset, join_style=2)
        else:
            out = line.parallel_offset(offset, join_style=2)
    except Exception:
        coords = np.asarray(line.coords)
        out = _offset_polyline_coords(coords, offset)
    return _longest_line(out)


def _clip_line(line: LineString | None, road_poly, min_keep_ratio: float) -> LineString | None:
    if line is None or line.is_empty:
        return None
    clipped = _merge_lines(line.intersection(road_poly))
    if clipped is None:
        return None
    if clipped.length < min_keep_ratio * max(1e-6, line.length):
        return None
    return clipped


def _centerline_width_stats(
    line: LineString,
    road_poly,
    step_m: float,
    probe_m: float,
) -> dict:
    pts, widths = _width_profile(line, road_poly, step_m, probe_m=probe_m)
    valid = [w for w in widths if w > 0]
    if not valid:
        return {
            "width_median": 0.0,
            "width_p95": 0.0,
            "dual_sample_ratio": 0.0,
            "sample_count": 0,
        }
    return {
        "width_median": float(np.median(valid)),
        "width_p95": float(np.percentile(valid, 95)),
        "dual_sample_ratio": 0.0,
        "sample_count": len(valid),
        "widths": valid,
    }


def _centerline_dual_offset(
    width_median: float,
    mode: str,
    offset_fixed: float,
    offset_max: float,
    offset_margin: float,
    center_offset_default: float,
) -> float:
    if mode == "half_width" and width_median > 0:
        half = max(0.0, width_median * 0.5 - offset_margin)
        return max(0.0, min(offset_max, half))
    if offset_fixed > 0:
        return offset_fixed
    return max(0.0, center_offset_default)


def _build_centerline_outputs(
    traj_line: LineString,
    road_poly,
    center_cfg: dict,
    center_offset_default: float,
) -> dict:
    base_line = _merge_lines(traj_line.intersection(road_poly)) or traj_line
    base_len = float(base_line.length)
    step_m = float(center_cfg["step_m"])
    probe_m = float(center_cfg["probe_m"])

    width_stats = _centerline_width_stats(base_line, road_poly, step_m, probe_m)
    widths = width_stats.get("widths") or []
    width_median = float(width_stats["width_median"])
    width_p95 = float(width_stats["width_p95"])
    dual_ratio = 0.0
    if widths:
        dual_ratio = sum(1 for w in widths if w >= center_cfg["dual_width_threshold_m"]) / len(widths)
    width_stats["dual_sample_ratio"] = dual_ratio

    dual_triggered = (
        base_len >= float(center_cfg["min_segment_length_m"])
        and dual_ratio >= float(center_cfg["min_dual_sample_ratio"])
        and widths
    )

    offset_mode = str(center_cfg["dual_offset_mode"]).lower()
    dual_offset = _centerline_dual_offset(
        width_median,
        offset_mode,
        float(center_cfg["dual_offset_m"]),
        float(center_cfg["dual_offset_max_m"]),
        float(center_cfg["dual_offset_margin_m"]),
        center_offset_default,
    )

    outputs: dict[str, list[dict]] = {"single": [], "dual": [], "both": [], "auto": []}
    outputs_lines: dict[str, list[LineString]] = {"single": [], "dual": [], "both": [], "auto": []}
    dual_fallback = False

    def _make_single(mode_label: str) -> dict:
        return {
            "type": "Feature",
            "geometry": mapping(base_line),
            "properties": {
                "lane_mode": "single",
                "line_type": "single",
                "side": "C",
                "mode": mode_label,
                "offset_m": 0.0,
                "dual_triggered": bool(dual_triggered),
            },
        }

    def _make_dual(mode_label: str) -> list[dict]:
        if dual_offset <= 0:
            return []
        left = _offset_line(base_line, dual_offset)
        right = _offset_line(base_line, -dual_offset)
        left = _clip_line(left, road_poly, float(center_cfg["dual_min_keep_ratio"]))
        right = _clip_line(right, road_poly, float(center_cfg["dual_min_keep_ratio"]))
        if left is None or right is None:
            return []
        return [
            {
                "type": "Feature",
                "geometry": mapping(left),
                "properties": {
                    "lane_mode": "dual_left",
                    "line_type": "dual",
                    "side": "L",
                    "mode": mode_label,
                    "offset_m": float(dual_offset),
                    "dual_triggered": bool(dual_triggered),
                },
            },
            {
                "type": "Feature",
                "geometry": mapping(right),
                "properties": {
                    "lane_mode": "dual_right",
                    "line_type": "dual",
                    "side": "R",
                    "mode": mode_label,
                    "offset_m": float(dual_offset),
                    "dual_triggered": bool(dual_triggered),
                },
            },
        ]

    outputs["single"].append(_make_single("single"))
    outputs_lines["single"].append(base_line)

    dual_features = _make_dual("dual")
    if dual_features:
        outputs["dual"].extend(dual_features)
        outputs_lines["dual"].extend([shape(f["geometry"]) for f in dual_features])
    else:
        dual_fallback = True

    mode = str(center_cfg["mode"]).lower()
    if mode == "dual":
        if dual_features:
            outputs["auto"] = dual_features
            outputs_lines["auto"] = outputs_lines["dual"]
        elif center_cfg.get("dual_fallback_single", True):
            outputs["auto"] = outputs["single"]
            outputs_lines["auto"] = outputs_lines["single"]
    elif mode == "both":
        outputs["both"] = outputs["single"] + (dual_features if dual_features else [])
        outputs_lines["both"] = outputs_lines["single"] + (outputs_lines["dual"] if dual_features else [])
        outputs["auto"] = outputs["both"]
        outputs_lines["auto"] = outputs_lines["both"]
    elif mode == "auto":
        if dual_triggered and dual_features:
            outputs["auto"] = dual_features
            outputs_lines["auto"] = outputs_lines["dual"]
        else:
            outputs["auto"] = outputs["single"]
            outputs_lines["auto"] = outputs_lines["single"]
    else:  # single
        outputs["auto"] = outputs["single"]
        outputs_lines["auto"] = outputs_lines["single"]

    if dual_fallback and outputs.get("auto"):
        for feat in outputs["auto"]:
            props = feat.get("properties") or {}
            props["dual_fallback"] = 1
            feat["properties"] = props

    return {
        "outputs": outputs,
        "outputs_lines": outputs_lines,
        "active_features": outputs["auto"],
        "active_lines": outputs_lines["auto"],
        "dual_triggered": bool(dual_triggered),
        "dual_offset_m": float(dual_offset),
        "dual_fallback": bool(dual_fallback),
        "width_median": width_median,
        "width_p95": width_p95,
        "dual_sample_ratio": float(dual_ratio),
    }


def _write_centerlines_outputs(
    out_dir: Path,
    outputs: dict,
    wgs84: Transformer,
) -> None:
    def _with_mode(features: list[dict], mode: str) -> list[dict]:
        out = []
        for feat in features:
            props = dict(feat.get("properties") or {})
            props["mode"] = mode
            out.append({"type": "Feature", "geometry": feat.get("geometry"), "properties": props})
        return out

    single = _with_mode(outputs.get("single") or [], "single")
    dual = _with_mode(outputs.get("dual") or [], "dual")
    both = _with_mode(outputs.get("both") or [], "both")
    auto = _with_mode(outputs.get("auto") or [], "auto")

    _write_geojson(out_dir / "centerlines_single.geojson", single)
    _write_geojson(out_dir / "centerlines_dual.geojson", dual)
    _write_geojson(out_dir / "centerlines_both.geojson", both)
    _write_geojson(out_dir / "centerlines_auto.geojson", auto)
    _write_geojson(out_dir / "centerlines.geojson", auto)
    _write_geojson_wgs84(out_dir / "centerlines_single_wgs84.geojson", single, wgs84)
    _write_geojson_wgs84(out_dir / "centerlines_dual_wgs84.geojson", dual, wgs84)
    _write_geojson_wgs84(out_dir / "centerlines_both_wgs84.geojson", both, wgs84)
    _write_geojson_wgs84(out_dir / "centerlines_auto_wgs84.geojson", auto, wgs84)
    _write_geojson_wgs84(out_dir / "centerlines_wgs84.geojson", auto, wgs84)


def _heading_angles(coords: np.ndarray) -> np.ndarray:
    dxy = np.diff(coords, axis=0)
    return np.arctan2(dxy[:, 1], dxy[:, 0])


def _angle_diff(a1: float, a2: float) -> float:
    d = a2 - a1
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return abs(d)


def _build_intersection_points(
    coords: np.ndarray,
    turn_thr_rad: float = 0.9,
    close_thr_m: float = 12.0,
    min_sep: int = 25,
) -> list[Point]:
    points: list[Point] = []
    if coords.shape[0] < 3:
        return points
    angles = _heading_angles(coords)
    for i in range(1, len(angles)):
        if _angle_diff(angles[i - 1], angles[i]) > turn_thr_rad:
            x, y = coords[i]
            points.append(Point(x, y))

    step = max(1, coords.shape[0] // 400)
    sampled = coords[::step]
    for i in range(len(sampled)):
        x1, y1 = sampled[i]
        for j in range(i + min_sep, len(sampled)):
            x2, y2 = sampled[j]
            if math.hypot(x2 - x1, y2 - y1) < close_thr_m:
                points.append(Point((x1 + x2) * 0.5, (y1 + y2) * 0.5))
    return points


def _road_bbox_dims(road_poly) -> tuple[float, float, float]:
    minx, miny, maxx, maxy = road_poly.bounds
    dx = maxx - minx
    dy = maxy - miny
    diag = math.hypot(dx, dy)
    return dx, dy, diag


def _explode_polygons(geom) -> list:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        polys = []
        for g in geom.geoms:
            polys.extend(_explode_polygons(g))
        return polys
    return []


def _polygon_vertex_count(geom) -> int:
    if geom.is_empty:
        return 0
    if geom.geom_type == "Polygon":
        return len(geom.exterior.coords)
    if geom.geom_type == "MultiPolygon":
        return sum(len(p.exterior.coords) for p in geom.geoms)
    if geom.geom_type == "GeometryCollection":
        return sum(len(p.exterior.coords) for p in _explode_polygons(geom))
    return 0


def _polygon_holes_count(geom) -> int:
    if geom.is_empty:
        return 0
    if geom.geom_type == "Polygon":
        return len(geom.interiors)
    if geom.geom_type == "MultiPolygon":
        return sum(len(p.interiors) for p in geom.geoms)
    if geom.geom_type == "GeometryCollection":
        return sum(len(p.interiors) for p in _explode_polygons(geom))
    return 0


def _polygon_roughness(perimeter: float, area: float) -> float:
    if area <= 0.0:
        return 0.0
    return float((perimeter * perimeter) / area)


def _postprocess_intersections(geom, road_poly, min_area: float, topk: int, simplify_m: float):
    if geom is None or geom.is_empty:
        return [], 0.0
    unioned = unary_union(geom)
    polys = _explode_polygons(unioned)
    if not polys:
        return [], 0.0
    clipped = [p.intersection(road_poly) for p in polys]
    clipped = [p for p in clipped if not p.is_empty]
    if not clipped:
        return [], 0.0
    cleaned = []
    for p in clipped:
        g = p.simplify(simplify_m, preserve_topology=True).buffer(0)
        if not g.is_empty:
            cleaned.extend(_explode_polygons(g))
    cleaned = [p for p in cleaned if p.area >= min_area]
    cleaned.sort(key=lambda p: p.area, reverse=True)
    if topk > 0:
        cleaned = cleaned[:topk]
    total_area = sum(p.area for p in cleaned)
    return cleaned, total_area


def _clean_polygons(geom, min_area: float, topk: int):
    polys = _explode_polygons(geom)
    polys = [p for p in polys if p.area >= min_area]
    polys.sort(key=lambda p: p.area, reverse=True)
    if topk > 0:
        polys = polys[:topk]
    return polys


def _sample_line_points(line: LineString, step_m: float) -> list[tuple[float, float]]:
    if line.length == 0:
        return []
    n = max(2, int(math.ceil(line.length / step_m)) + 1)
    return [line.interpolate(float(i) / (n - 1), normalized=True).coords[0] for i in range(n)]


def _smooth_median(values: list[float], win: int = 5) -> list[float]:
    if not values:
        return values
    out = []
    half = win // 2
    for i in range(len(values)):
        s = max(0, i - half)
        e = min(len(values), i + half + 1)
        v = sorted(values[s:e])
        out.append(v[len(v) // 2])
    return out


def _ray_intersections(road_poly, origin: tuple[float, float], direction: tuple[float, float], max_dist: float = 60.0) -> float | None:
    ox, oy = origin
    dx, dy = direction
    end = (ox + dx * max_dist, oy + dy * max_dist)
    ray = LineString([origin, end])
    inter = ray.intersection(road_poly)
    if inter.is_empty:
        return None
    if inter.geom_type == "LineString":
        return inter.length
    if inter.geom_type == "MultiLineString":
        return max(g.length for g in inter.geoms) if inter.geoms else None
    if inter.geom_type == "GeometryCollection":
        lines = [g for g in inter.geoms if g.geom_type == "LineString"]
        if not lines:
            return None
        return max(g.length for g in lines)
    return None


def _width_profile(
    line: LineString,
    road_poly,
    step_m: float,
    probe_m: float = 60.0,
) -> tuple[list[tuple[float, float]], list[float]]:
    pts = _sample_line_points(line, step_m)
    if len(pts) < 2:
        return pts, []
    widths: list[float] = []
    coords = np.asarray(pts)
    for i in range(len(coords)):
        if i == 0:
            dx, dy = coords[1] - coords[0]
        elif i == len(coords) - 1:
            dx, dy = coords[-1] - coords[-2]
        else:
            dx, dy = coords[i + 1] - coords[i - 1]
        length = math.hypot(dx, dy)
        if length == 0:
            widths.append(0.0)
            continue
        nx, ny = -dy / length, dx / length
        w1 = _ray_intersections(road_poly, (coords[i, 0], coords[i, 1]), (nx, ny), max_dist=probe_m)
        w2 = _ray_intersections(road_poly, (coords[i, 0], coords[i, 1]), (-nx, -ny), max_dist=probe_m)
        if w1 is None or w2 is None:
            widths.append(0.0)
        else:
            widths.append(float(w1 + w2))
    return pts, widths


def _cluster_peaks(points: list[tuple[float, float]], mask: list[bool], merge_dist: float = 20.0) -> list[tuple[float, float]]:
    clusters = []
    current: list[tuple[float, float]] = []
    last = None
    for pt, is_peak in zip(points, mask):
        if not is_peak:
            if current:
                clusters.append(current)
                current = []
            last = None
            continue
        if last is None:
            current.append(pt)
            last = pt
        else:
            if math.hypot(pt[0] - last[0], pt[1] - last[1]) <= merge_dist:
                current.append(pt)
            else:
                clusters.append(current)
                current = [pt]
            last = pt
    if current:
        clusters.append(current)
    centers = []
    for c in clusters:
        xs = [p[0] for p in c]
        ys = [p[1] for p in c]
        centers.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return centers


def _write_geojson(path: Path, features: list[dict]) -> None:
    fc = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_geojson_wgs84(path: Path, features: list[dict], transformer: Transformer) -> None:
    def _proj(x, y, z=None):
        lon, lat = transformer.transform(x, y)
        if z is None:
            return lon, lat
        return lon, lat, z

    out_features = []
    for feat in features:
        geom = feat.get("geometry")
        if geom is None:
            continue
        shp = shapely_transform(_proj, shape(geom))
        out_features.append(
            {
                "type": "Feature",
                "geometry": mapping(shp),
                "properties": feat.get("properties", {}),
            }
        )
    _write_geojson(path, out_features)


def _project_points_to_image(
    pts_velo: np.ndarray,
    t_velo_to_cam: np.ndarray,
    r_rect: np.ndarray,
    p_rect: np.ndarray,
    img_w: int,
    img_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pts_velo.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=bool)
    pts_h = np.hstack([pts_velo[:, :3], np.ones((pts_velo.shape[0], 1), dtype=float)])
    cam = (r_rect @ (t_velo_to_cam @ pts_h.T)).T
    in_front = cam[:, 2] > 0.1
    proj = (p_rect @ cam.T).T
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]
    u_i = np.round(u).astype(np.int64)
    v_i = np.round(v).astype(np.int64)
    valid = (u_i >= 0) & (u_i < img_w) & (v_i >= 0) & (v_i < img_h)
    return u_i, v_i, in_front & valid


def _load_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "NN backend requires torch, transformers, opencv-python, and pillow. "
            "Install optional deps (see requirements_nn.txt) or set GEOM_BACKEND=algo."
        ) from exc
    return cv2


def _grid_from_algo(
    bin_files: list[Path],
    poses_xy: np.ndarray,
    yaws: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
) -> dict[tuple[int, int], int]:
    grid: dict[tuple[int, int], int] = {}
    for i in range(len(bin_files)):
        pts = _read_velodyne_points(bin_files[i])
        if pts.size == 0:
            continue
        z = pts[:, 2]
        mask = _ground_mask(z, z_band=0.6)
        if not np.any(mask):
            continue
        pts = pts[mask]
        x = pts[:, 0]
        y = pts[:, 1]
        yaw = float(yaws[i])
        c = math.cos(yaw)
        s = math.sin(yaw)
        xw = c * x - s * y + poses_xy[i, 0]
        yw = s * x + c * y + poses_xy[i, 1]
        counts = _grid_counts(xw, yw, origin, resolution)
        _merge_counts(grid, counts)
    return grid


def _grid_from_nn(
    bin_files: list[Path],
    img_files: list[Optional[Path]],
    poses_xy: np.ndarray,
    yaws: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    stride: int,
    args,
    out_dir: Path,
    data_root: Path,
    drive: str,
    camera: str,
) -> dict[tuple[int, int], int]:
    cv2 = _load_cv2()
    calib = _load_kitti360_calib(data_root, drive, camera)
    t_velo_to_cam = calib["t_velo_to_cam"]
    r_rect = calib["r_rect"]
    p_rect = calib["p_rect"]
    class_names = [c.strip().lower() for c in str(args.nn_drivable_classes).split(",") if c.strip()]
    model_cfg = {
        "model_family": args.nn_model_family,
        "model_id": args.nn_model,
        "mask_threshold": float(args.nn_mask_threshold),
        "postprocess_params": {"drivable_classes": class_names},
    }
    model_res = load_nn_model(model_cfg)
    if not model_res.available or not model_res.predictor:
        raise RuntimeError(model_res.reason or "model_unavailable")
    predictor = model_res.predictor

    masks_dir = ensure_dir(out_dir / "masks")
    grid: dict[tuple[int, int], int] = {}

    for i in range(0, len(bin_files), max(1, stride)):
        img_path = img_files[i] if i < len(img_files) else None
        if img_path is None or not img_path.exists():
            continue
        mask_path = masks_dir / f"{img_path.stem}.png"
        if bool(args.nn_cache) and mask_path.exists():
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                continue
            mask = mask_img > 0
        else:
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mask = predictor.predict(img_rgb)
            if bool(args.nn_cache):
                cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))

        pts = _read_velodyne_points(bin_files[i])
        if pts.size == 0:
            continue
        z = pts[:, 2]
        ground = _ground_mask(z, z_band=0.6)
        if not np.any(ground):
            continue
        pts = pts[ground]

        img_h, img_w = mask.shape[:2]
        u, v, valid = _project_points_to_image(pts, t_velo_to_cam, r_rect, p_rect, img_w, img_h)
        if u.size == 0:
            continue
        mask_vals = mask[v[valid], u[valid]]
        if not np.any(mask_vals):
            continue
        pts = pts[valid][mask_vals]

        x = pts[:, 0]
        y = pts[:, 1]
        yaw = float(yaws[i])
        c = math.cos(yaw)
        s = math.sin(yaw)
        xw = c * x - s * y + poses_xy[i, 0]
        yw = s * x + c * y + poses_xy[i, 1]
        counts = _grid_counts(xw, yw, origin, resolution)
        _merge_counts(grid, counts)

    return grid

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="2013_05_28_drive_0000_sync", help="KITTI-360 drive name")
    ap.add_argument("--max-frames", type=int, default=2000, help="max frames to read")
    ap.add_argument("--frame-start", type=int, default=0, help="start frame index")
    ap.add_argument("--frame-count", type=int, default=0, help="number of frames (0=use max-frames)")
    ap.add_argument("--road-min-area", type=float, default=50.0, help="min road polygon area to keep")
    ap.add_argument("--road-topk", type=int, default=1, help="keep top-k road components by area")
    ap.add_argument("--inter-min-area", type=float, default=100.0, help="min intersection area to keep")
    ap.add_argument("--inter-topk", type=int, default=10, help="keep top-k intersections by area")
    ap.add_argument("--inter-simplify", type=float, default=0.8, help="intersection simplify meters")
    ap.add_argument("--inter-mode", choices=["widen", "turn"], default="widen", help="intersection candidate mode")
    ap.add_argument("--width-sample-step", type=float, default=3.0, help="sampling step along centerline (m)")
    ap.add_argument("--width-peak-ratio", type=float, default=1.6, help="width peak ratio vs local median")
    ap.add_argument("--width-buffer-mult", type=float, default=0.8, help="buffer radius multiplier for width peaks")
    ap.add_argument("--grid-resolution", type=float, default=0.5, help="BEV grid resolution (m)")
    ap.add_argument("--density-thr", type=int, default=3, help="grid density threshold")
    ap.add_argument("--corridor-m", type=float, default=15.0, help="trajectory corridor width (m)")
    ap.add_argument("--simplify-m", type=float, default=1.2, help="geometry simplify meters")
    ap.add_argument("--centerline-offset-m", type=float, default=3.5, help="centerline offset (m)")
    ap.add_argument("--centerlines-config", default="configs/centerlines.yaml", help="centerlines config yaml")
    ap.add_argument("--intersection-refine-config", default="configs/intersections_shape_refine.yaml", help="intersection refine config yaml")
    ap.add_argument("--allow-empty-intersections", type=int, default=1, help="allow empty intersections without failing")
    ap.add_argument(
        "--intersection-backend",
        choices=["algo", "sat", "hybrid"],
        default="hybrid",
        help="intersection backend selection",
    )
    ap.add_argument("--sat-patch-m", type=float, default=256.0, help="SAT patch size in meters")
    ap.add_argument("--sat-conf-thr", type=float, default=0.3, help="SAT confidence threshold")
    ap.add_argument("--nn-model-family", default="segformer", help="NN model family (segformer/mask2former)")
    ap.add_argument("--nn-model", default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024", help="HF model id")
    ap.add_argument("--nn-camera", default="image_00", help="camera folder name (image_00/image_01)")
    ap.add_argument("--nn-stride", type=int, default=5, help="frame stride for NN inference")
    ap.add_argument("--nn-cache", type=int, default=1, help="cache segmentation masks to outputs/masks")
    ap.add_argument("--nn-drivable-classes", default="road,sidewalk", help="comma-separated drivable class names")
    ap.add_argument("--nn-mask-threshold", type=float, default=0.5, help="mask threshold (if applicable)")
    args = ap.parse_args()

    run_id = new_run_id("geom")
    run_dir = ensure_dir(Path(__file__).resolve().parents[1] / "runs" / run_id)
    out_dir = ensure_dir(run_dir / "outputs")
    status = "FAIL"
    reason: Optional[str] = None
    model_id: Optional[str] = None
    backend_used = "unknown"
    geom_summary = {
        "run_id": run_id,
        "drive": args.drive,
        "backend_used": backend_used,
        "backend": backend_used,
        "model_family": args.nn_model_family,
        "model_id": model_id,
        "camera": args.nn_camera,
        "stride": int(args.nn_stride),
        "internal_epsg": 32632,
        "wgs84_epsg": 4326,
        "status": status,
        "reason": reason,
        "nn_fixed": False,
        "nn_best_cfg_path": None,
        "mask_threshold": float(args.nn_mask_threshold),
    }

    try:
        data_root = os.environ.get("POC_DATA_ROOT", "")
        if not data_root:
            raise SystemExit("ERROR: POC_DATA_ROOT is not set.")
        data_root = Path(data_root)

        nn_fixed = os.environ.get("GEOM_NN_FIXED", "0").strip() == "1"
        nn_best_cfg_path = Path(os.environ.get("GEOM_NN_BEST_CFG", "configs/geom_nn_best.yaml"))
        if nn_fixed and os.environ.get("GEOM_BACKEND", "auto").strip().lower() == "nn":
            best_cfg = _load_nn_best_cfg(nn_best_cfg_path)
            args.nn_model = best_cfg.get("model_id", args.nn_model)
            args.nn_model_family = best_cfg.get("model_family", args.nn_model_family)
            args.nn_camera = best_cfg.get("camera", args.nn_camera)
            args.nn_stride = int(best_cfg.get("stride", args.nn_stride))
            args.nn_mask_threshold = float(best_cfg.get("mask_threshold", args.nn_mask_threshold))
            if "postprocess_params" in best_cfg and isinstance(best_cfg["postprocess_params"], dict):
                drivable = best_cfg["postprocess_params"].get("drivable_classes")
                if drivable:
                    args.nn_drivable_classes = ",".join(drivable)

        oxts_dir = _find_oxts_dir(data_root, args.drive)
        velodyne_dir = _find_velodyne_dir(data_root, args.drive)
        frame_start = max(0, int(args.frame_start))
        frame_count = int(args.frame_count) if args.frame_count and args.frame_count > 0 else int(args.max_frames)
        if frame_count <= 0:
            frame_count = 2000

        oxts_pts = _read_latlon(oxts_dir, max_frames=0)
        poses_xy, yaws = _project_to_utm32(oxts_pts)

        bin_files = sorted(velodyne_dir.glob("*.bin"))
        if frame_count and frame_count > 0:
            bin_files = bin_files[frame_start: frame_start + frame_count]
        n = min(len(bin_files), poses_xy.shape[0])
        if n < 2:
            raise SystemExit("ERROR: insufficient velodyne frames.")
        bin_files = bin_files[:n]
        poses_xy = poses_xy[frame_start: frame_start + n]
        yaws = yaws[frame_start: frame_start + n]

        resolution = float(args.grid_resolution)
        density_thr = int(args.density_thr)
        corridor_m = float(args.corridor_m)
        simplify_m = float(args.simplify_m)
        center_offset_m = float(args.centerline_offset_m)
        post_mask_close_m = _get_env_float("POST_MASK_CLOSE_M", 1.0)
        post_mask_open_m = _get_env_float("POST_MASK_OPEN_M", 0.5)
        post_min_component_m2 = _get_env_float("POST_MIN_COMPONENT_M2", 50.0)
        post_fill_holes_m2 = _get_env_float("POST_FILL_HOLES_M2", 200.0)
        poly_simplify_m = _get_env_float("POLY_SIMPLIFY_M", 0.5)
        poly_smooth_buf_m = _get_env_float("POLY_SMOOTH_BUF_M", 1.0)
        inter_simplify_m = _get_env_float("INTER_SIMPLIFY_M", 0.5)
        inter_smooth_buf_m = _get_env_float("INTER_SMOOTH_BUF_M", 1.0)
        smooth_profile = _get_env_str("SMOOTH_PROFILE", "default").lower()
        strong_min_part_area_m2 = _get_env_float("STRONG_MIN_PART_AREA_M2", 120.0)
        strong_close_m = _get_env_float("STRONG_CLOSE_M", 2.5)
        strong_open_m = _get_env_float("STRONG_OPEN_M", 1.2)
        strong_simplify_m = _get_env_float("STRONG_SIMPLIFY_M", 1.2)
        strong_min_hole_area_m2 = _get_env_float("STRONG_MIN_HOLE_AREA_M2", 350.0)
        center_cfg = _centerlines_defaults()
        center_cfg_path = Path(os.environ.get("CENTERLINES_CONFIG", "") or args.centerlines_config)
        center_cfg.update(_load_centerlines_cfg(center_cfg_path))
        env_mode = os.environ.get("CENTERLINE_MODE", "").strip()
        if env_mode:
            center_cfg["mode"] = env_mode
        env_dual_width = os.environ.get("DUAL_WIDTH_THRESH_M", "").strip()
        if env_dual_width:
            center_cfg["dual_width_threshold_m"] = float(env_dual_width)
        env_dual_len = os.environ.get("DUAL_MIN_LEN_M", "").strip()
        if env_dual_len:
            center_cfg["min_segment_length_m"] = float(env_dual_len)
        env_dual_offset = os.environ.get("DUAL_OFFSET_M", "").strip().lower()
        if env_dual_offset:
            if env_dual_offset not in {"auto", ""}:
                center_cfg["dual_offset_m"] = float(env_dual_offset)
                center_cfg["dual_offset_mode"] = "fixed"
        env_feature_store = os.environ.get("FEATURE_STORE_DIR", "").strip()
        if env_feature_store and not center_cfg.get("seg_divider_feature_store_dir"):
            center_cfg["seg_divider_feature_store_dir"] = env_feature_store

        centerline_mode = str(center_cfg["mode"]).lower()
        refine_cfg = _intersection_refine_defaults()
        refine_cfg_path = Path(os.environ.get("INTERSECTION_REFINE_CONFIG", "") or args.intersection_refine_config)
        refine_cfg.update(_load_intersection_refine_cfg(refine_cfg_path))
        inter_backend = _get_env_str("INTERSECTION_BACKEND", args.intersection_backend).lower()
        sat_patch_m = _get_env_float("SAT_PATCH_M", float(args.sat_patch_m))
        sat_conf_thr = _get_env_float("SAT_CONF_THR", float(args.sat_conf_thr))
        dop20_root = Path(os.environ.get("DOP20_ROOT", r"E:\KITTI360\KITTI-360\_lglbw_dop20"))

        origin = (float(poses_xy[0, 0]), float(poses_xy[0, 1]))
        backend_env = os.environ.get("GEOM_BACKEND", "auto").strip().lower()
        if backend_env not in {"auto", "nn", "algo"}:
            raise SystemExit("ERROR: GEOM_BACKEND must be auto|nn|algo")
        backend_used = "algo"
        model_id: Optional[str] = None
        grid: dict[tuple[int, int], int] = {}
        if backend_env in {"auto", "nn"}:
            try:
                img_dir = _find_image_dir(data_root, args.drive, args.nn_camera)
                img_map = {p.stem: p for p in sorted(img_dir.glob("*.png"))}
                img_files: list[Optional[Path]] = [img_map.get(p.stem) for p in bin_files]
                if not any(p is not None for p in img_files):
                    raise RuntimeError(f"no images found for {args.nn_camera} in {img_dir}")
                print(
                    f"[GEOM] using NN backend ({args.nn_model_family}:{args.nn_model}) "
                    f"on {args.nn_camera}, stride={args.nn_stride}"
                )
                grid = _grid_from_nn(
                    bin_files=bin_files,
                    img_files=img_files,
                    poses_xy=poses_xy,
                    yaws=yaws,
                    origin=origin,
                    resolution=resolution,
                    stride=int(args.nn_stride),
                    args=args,
                    out_dir=out_dir,
                    data_root=data_root,
                    drive=args.drive,
                    camera=args.nn_camera,
                )
                backend_used = "nn"
                model_id = args.nn_model
            except Exception as exc:
                if backend_env == "nn":
                    raise SystemExit(f"ERROR: NN backend failed: {exc}") from exc
                print(f"[GEOM][WARN] NN backend unavailable, fallback to algo: {exc}")
                grid = _grid_from_algo(
                    bin_files=bin_files,
                    poses_xy=poses_xy,
                    yaws=yaws,
                    origin=origin,
                    resolution=resolution,
                )
                backend_used = "algo"
        else:
            print("[GEOM] using ALGO backend")
            grid = _grid_from_algo(
                bin_files=bin_files,
                poses_xy=poses_xy,
                yaws=yaws,
                origin=origin,
                resolution=resolution,
            )

        print(
            f"[GEOM] backend_used={backend_used} camera={args.nn_camera} stride={args.nn_stride} "
            f"model_family={args.nn_model_family} model_id={model_id or 'none'}"
        )

        traj_line = LineString(poses_xy.tolist())
        corridor = traj_line.buffer(corridor_m, cap_style=2, join_style=2)
        corridor_prep = prep(corridor)

        cells = []
        for (ix, iy), count in grid.items():
            if count < density_thr:
                continue
            cx = origin[0] + (ix + 0.5) * resolution
            cy = origin[1] + (iy + 0.5) * resolution
            if not corridor_prep.contains(Point(cx, cy)):
                continue
            cells.append(box(
                origin[0] + ix * resolution,
                origin[1] + iy * resolution,
                origin[0] + (ix + 1) * resolution,
                origin[1] + (iy + 1) * resolution,
            ))

        if cells:
            road_poly = unary_union(cells)
        else:
            road_poly = corridor

        road_poly = unary_union([road_poly, traj_line.buffer(6.0, cap_style=2, join_style=2)])
        road_poly = _close_open_geom(road_poly, post_mask_close_m, post_mask_open_m)
        road_poly = _smooth_geom(road_poly, poly_smooth_buf_m)
        road_poly = road_poly.simplify(poly_simplify_m, preserve_topology=True)
        road_poly = _fill_small_holes(road_poly, post_fill_holes_m2)

        smooth_compare = None
        if smooth_profile == "strong":
            road_poly, smooth_compare = _apply_strong_smoothing(
                road_poly,
                min_part_area=strong_min_part_area_m2,
                close_m=strong_close_m,
                open_m=strong_open_m,
                simplify_m=strong_simplify_m,
                min_hole_area=strong_min_hole_area_m2,
            )

        road_before = len(_explode_polygons(road_poly))
        road_min_area = max(float(args.road_min_area), post_min_component_m2)
        road_topk = int(args.road_topk)
        road_keep = _clean_polygons(road_poly, road_min_area, road_topk)
        road_after = len(road_keep)
        road_poly = unary_union(road_keep) if road_keep else road_poly

        base_centerline = _merge_lines(traj_line.intersection(road_poly)) or traj_line
        width_stats = _centerline_width_stats(
            base_centerline,
            road_poly,
            float(center_cfg["step_m"]),
            float(center_cfg["probe_m"]),
        )
        center_cfg["width_median_m"] = float(width_stats["width_median"])
        center_cfg["width_p95_m"] = float(width_stats["width_p95"])

        divider_lines = []
        divider_src_hint = None
        divider_sources = center_cfg.get("divider_sources", ["geom"])
        if isinstance(divider_sources, str):
            divider_sources = [divider_sources]
        use_seg = "seg" in {str(s).lower() for s in divider_sources}
        feature_store_dir = center_cfg.get("seg_divider_feature_store_dir")
        if use_seg and feature_store_dir:
            features = load_image_features(args.drive, None, Path(feature_store_dir))
            divider_feats = features.get("divider_median") or []
            if not divider_feats:
                lane_feats = features.get("lane_marking") or []
                lane_subtypes = set(
                    s.lower()
                    for s in center_cfg.get("seg_divider_lane_subtypes", ["double_yellow", "solid_double"])
                )
                divider_feats = [
                    f for f in lane_feats
                    if str((f.get("properties") or {}).get("subtype") or "").lower() in lane_subtypes
                ]
                if divider_feats:
                    divider_src_hint = "lane_marking"
            if divider_feats:
                divider_src_hint = divider_src_hint or "divider_median"
                accepted = []
                for feat in divider_feats:
                    props = feat.get("properties") or {}
                    geom = feat.get("geometry")
                    frame = str(props.get("geometry_frame") or "")
                    if geom is None or geom.is_empty:
                        continue
                    if frame in {"map", "ego"}:
                        accepted.append(geom)
                    elif frame == "image_px" and center_cfg.get("accept_image_px", False):
                        accepted.append(geom)
                if accepted and center_cfg.get("image_px_to_map") == "fit_road_bbox":
                    minx, miny, maxx, maxy = road_poly.bounds
                    geom_bounds = unary_union(accepted).bounds
                    gx0, gy0, gx1, gy1 = geom_bounds
                    sx = (maxx - minx) / max(1e-6, gx1 - gx0)
                    sy = (maxy - miny) / max(1e-6, gy1 - gy0)
                    scaled = []
                    for g in accepted:
                        g2 = shapely_affinity.scale(g, xfact=sx, yfact=sy, origin=(gx0, gy0))
                        g2 = shapely_affinity.translate(g2, xoff=minx - gx0 * sx, yoff=miny - gy0 * sy)
                        scaled.append(g2)
                    accepted = scaled
                divider_lines = accepted

        center_result = build_centerlines_v2(
            traj_line=traj_line,
            road_poly=road_poly,
            center_cfg=center_cfg,
            center_offset_default=center_offset_m,
            divider_lines=divider_lines,
            divider_src_hint=divider_src_hint,
        )
        center_outputs = center_result["outputs"]
        for feats in center_outputs.values():
            for feat in feats:
                props = feat.get("properties") or {}
                props["drive_id"] = args.drive
                props["tile_id"] = args.drive
                feat["properties"] = props

        wgs84 = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
        _write_centerlines_outputs(out_dir, center_outputs, wgs84)
        if center_cfg.get("debug_divider_layers", False):
            debug_dir = out_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            divider_line = center_result.get("divider_line")
            if divider_line is not None and not divider_line.is_empty:
                _write_geojson(
                    debug_dir / "divider_debug.geojson",
                    [
                        {
                            "type": "Feature",
                            "geometry": mapping(divider_line),
                            "properties": {"divider_src": center_result.get("divider_src")},
                        }
                    ],
                )
            carriageways = center_result.get("carriageways") or []
            if len(carriageways) >= 1:
                _write_geojson(
                    debug_dir / "carriageway_split_L.geojson",
                    [{"type": "Feature", "geometry": mapping(carriageways[0]), "properties": {}}],
                )
            if len(carriageways) >= 2:
                _write_geojson(
                    debug_dir / "carriageway_split_R.geojson",
                    [{"type": "Feature", "geometry": mapping(carriageways[1]), "properties": {}}],
                )
            dual_feats = center_outputs.get("dual") or []
            if dual_feats:
                _write_geojson(debug_dir / "dual_centerlines_debug.geojson", dual_feats)
        center_features = center_result["active_features"]
        center_lines = center_result["active_lines"]
        dual_offset_used = center_result.get("dual_sep_m")
        dual_triggered = bool(center_result["dual_triggered"])
        center_dual_ratio = 0.0

        single_feats = center_outputs.get("single") or []
        dual_feats = center_outputs.get("dual") or []
        single_lines = [shape(f.get("geometry")) for f in single_feats if f.get("geometry")]
        dual_lines = [shape(f.get("geometry")) for f in dual_feats if f.get("geometry")]
        single_len = float(sum(l.length for l in single_lines))
        dual_len = float(sum(l.length for l in dual_lines))
        avg_offset = None
        if dual_offset_used is not None:
            avg_offset = float(dual_offset_used)
        if single_len > 0:
            center_dual_ratio = float(dual_len / max(1e-6, single_len))
        dual_triggered_segments = 1 if dual_triggered else 0
        total_segments = 1 if center_lines else 0

        inter_mode = str(args.inter_mode).lower()
        inter_pts: list[Point] = []
        width_median = 0.0
        width_p95 = 0.0
        peak_point_count = 0
        cluster_count = 0
        if inter_mode == "widen":
            sample_step = float(args.width_sample_step)
            pts, widths = _width_profile(traj_line, road_poly, sample_step)
            if widths:
                med = float(np.median(widths))
                width_median = med
                width_p95 = float(np.percentile(widths, 95))
                smoothed = _smooth_median(widths, win=5)
                peaks = [w > (med * float(args.width_peak_ratio)) for w in smoothed]
                peak_point_count = sum(1 for p in peaks if p)
                centers = _cluster_peaks(pts, peaks, merge_dist=20.0)
                cluster_count = len(centers)
                inter_pts = [Point(c) for c in centers]
        else:
            inter_pts = _build_intersection_points(poses_xy, turn_thr_rad=0.9, close_thr_m=12.0, min_sep=25)
        inter_polys = []
        inter_area_total = 0.0
        inter_min_area = float(args.inter_min_area)
        inter_topk = int(args.inter_topk)
        inter_simplify = float(args.inter_simplify)
        if inter_pts:
            if inter_mode == "widen":
                buffers = []
                for p in inter_pts:
                    radius = max(8.0, width_median * float(args.width_buffer_mult))
                    buffers.append(p.buffer(radius))
                inter_union = unary_union(buffers)
            else:
                inter_union = unary_union([p.buffer(15.0) for p in inter_pts])
            inter_union = _smooth_geom(inter_union, inter_smooth_buf_m)
            inter_polys, inter_area_total = _postprocess_intersections(
                inter_union,
                road_poly,
                min_area=inter_min_area,
                topk=inter_topk,
                simplify_m=inter_simplify_m,
            )
            inter_before = len(inter_polys)
            inter_union2 = unary_union(inter_polys) if inter_polys else inter_union
            inter_clean = _clean_polygons(inter_union2, inter_min_area, inter_topk)
            inter_clean = [
                p.simplify(inter_simplify_m, preserve_topology=True).buffer(0)
                for p in inter_clean
                if not p.is_empty
            ]
            inter_clean = [p for p in inter_clean if p.area >= inter_min_area]
            inter_polys = inter_clean
            inter_area_total = sum(p.area for p in inter_polys)
            inter_after = len(inter_polys)
        inter_polys_algo = list(inter_polys)
        debug_dir = out_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        seed_features = []
        local_features = []
        arms_features = []
        refined_features = []
        refined_algo_polys: list[Polygon] = []
        for idx, poly in enumerate(inter_polys_algo):
            seed = inter_pts[idx] if idx < len(inter_pts) else poly.centroid
            seed_features.append(
                {"type": "Feature", "geometry": mapping(seed), "properties": {"idx": idx}}
            )
            refined, meta = _refine_intersection_polygon(
                seed_pt=seed,
                poly_candidate=poly,
                road_polygon=road_poly,
                centerlines=center_lines,
                cfg=refine_cfg,
            )
            if meta.get("local") is not None:
                local_features.append(
                    {"type": "Feature", "geometry": mapping(meta["local"]), "properties": {"idx": idx}}
                )
            if meta.get("arms") is not None and not meta["arms"].is_empty:
                arms_features.append(
                    {"type": "Feature", "geometry": mapping(meta["arms"]), "properties": {"idx": idx}}
                )
            if refined is not None:
                refined_algo_polys.append(refined)
                refined_features.append(
                    {"type": "Feature", "geometry": mapping(refined), "properties": {"idx": idx, "reason": meta.get("reason")}}
                )
        if seed_features:
            _write_geojson(debug_dir / "intersections_seed_points.geojson", seed_features)
        if local_features:
            _write_geojson(debug_dir / "intersections_local_clip.geojson", local_features)
        if arms_features:
            _write_geojson(debug_dir / "intersections_arms.geojson", arms_features)
        if refined_features:
            _write_geojson(debug_dir / "intersections_refined.geojson", refined_features)
        if refined_algo_polys:
            inter_polys_algo = refined_algo_polys
        inter_features_algo = []
        for p in inter_polys_algo:
            circ = _intersection_circularity(p)
            aspect = _intersection_aspect_ratio(p)
            overlap = _intersection_overlap_with_road(p, road_poly)
            arms = _arm_count(p, center_lines, float(refine_cfg["arm_buffer_m"]))
            inter_features_algo.append(
                {
                    "type": "Feature",
                    "geometry": mapping(p),
                    "properties": {
                        "backend_used": "algo",
                        "src": "algo",
                        "shape_refined": 1,
                        "circularity": round(circ, 4),
                        "aspect_ratio": round(aspect, 4),
                        "overlap_road": round(overlap, 4),
                        "arm_count": int(arms),
                    },
                }
            )

        sat_present = False
        sat_avg_conf = 0.0
        inter_polys_sat: list[Polygon] = []
        sat_reason = ""
        if inter_backend in {"sat", "hybrid"}:
            sat_result = run_sat_intersections(
                drive=args.drive,
                candidates=inter_pts,
                traj_points=[(float(x), float(y)) for x, y in poses_xy.tolist()],
                outputs_dir=out_dir,
                crs_epsg=32632,
                patch_m=sat_patch_m,
                conf_thr=sat_conf_thr,
                dop20_root=dop20_root,
            )
            sat_present = bool(sat_result.get("present"))
            sat_avg_conf = float(sat_result.get("avg_confidence") or 0.0)
            inter_polys_sat = sat_result.get("polys") or []
            sat_reason = sat_result.get("reason") or ""

        inter_polys_final = inter_polys_algo
        intersection_backend_used = "algo"
        if inter_backend == "sat":
            if sat_present:
                inter_polys_final = inter_polys_sat
                intersection_backend_used = "sat"
            else:
                print(f"[GEOM][WARN] SAT intersections unavailable; fallback to algo ({sat_reason or 'no_sat'})")
        elif inter_backend == "hybrid":
            if sat_present and sat_avg_conf >= sat_conf_thr:
                inter_polys_final = inter_polys_sat
                intersection_backend_used = "sat"

        inter_features = []
        refined_final_polys: list[Polygon] = []
        for p in inter_polys_final:
            seed = p.centroid
            pre_circ = _intersection_circularity(p)
            refined, meta = _refine_intersection_polygon(
                seed_pt=seed,
                poly_candidate=p,
                road_polygon=road_poly,
                centerlines=center_lines,
                cfg=refine_cfg,
            )
            if refined is None:
                refined = p
            post_circ = _intersection_circularity(refined)
            if post_circ > float(refine_cfg["max_circularity"]) and _arm_count(refined, center_lines, float(refine_cfg["arm_buffer_m"])) >= 3:
                shrink_cfg = dict(refine_cfg)
                shrink_cfg["radius_m"] = float(refine_cfg["radius_m"]) * 0.7
                refined2, _ = _refine_intersection_polygon(
                    seed_pt=seed,
                    poly_candidate=p,
                    road_polygon=road_poly,
                    centerlines=center_lines,
                    cfg=shrink_cfg,
                )
                if refined2 is not None:
                    refined = refined2
                    post_circ = _intersection_circularity(refined)
            refined_final_polys.append(refined)
            circ = post_circ
            aspect = _intersection_aspect_ratio(refined)
            overlap = _intersection_overlap_with_road(refined, road_poly)
            arms = _arm_count(refined, center_lines, float(refine_cfg["arm_buffer_m"]))
            inter_features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(refined),
                    "properties": {
                        "backend_used": intersection_backend_used,
                        "src": intersection_backend_used,
                        "reason": f"backend_{intersection_backend_used}",
                        "sat_present": bool(sat_present),
                        "sat_confidence": round(sat_avg_conf, 4),
                        "shape_refined": 1,
                        "pre_circularity": round(pre_circ, 4),
                        "post_circularity": round(post_circ, 4),
                        "circularity": round(circ, 4),
                        "aspect_ratio": round(aspect, 4),
                        "overlap_road": round(overlap, 4),
                        "arm_count": int(arms),
                    },
                }
            )
        inter_polys_final = refined_final_polys

        _write_geojson(out_dir / "road_polygon.geojson", [{"type": "Feature", "geometry": mapping(road_poly), "properties": {}}])
        _write_geojson(out_dir / "intersections.geojson", inter_features)
        _write_geojson(out_dir / "intersections_algo.geojson", inter_features_algo)

        road_dx, road_dy, road_diag = _road_bbox_dims(road_poly)
        line_lengths = [float(line.length) for line in center_lines]
        total_len = float(sum(line_lengths))
        inter_len = 0.0
        for line in center_lines:
            if line.length > 0:
                inter_len += float(line.intersection(road_poly).length)
        center_in_poly_ratio = (inter_len / total_len) if total_len > 0 else 0.0
        print(f"[QC] road bbox dx={road_dx:.2f}m dy={road_dy:.2f}m diag={road_diag:.2f}m")
        for i, ln in enumerate(line_lengths, 1):
            print(f"[QC] centerline_{i} length={ln:.2f}m")
        print(f"[QC] centerline_total length={total_len:.2f}m")
        inter_area_total_final = sum(p.area for p in inter_polys_final)
        inter_before_final = len(inter_polys_final)
        inter_after_final = len(inter_polys_final)
        print(f"[QC] road_component_count_before={road_before} after={road_after}")
        print(f"[QC] inter_component_count_before={inter_before_final} after={inter_after_final}")
        print(f"[QC] width_median={width_median:.2f} width_p95={width_p95:.2f} peak_point_count={peak_point_count} cluster_count={cluster_count}")
        top5 = sorted([p.area for p in inter_polys_final], reverse=True)[:5]
        print(f"[QC] intersections_count={len(inter_polys_final)}")
        print(f"[QC] intersections_area_total={inter_area_total_final:.2f}m2")
        print(f"[QC] intersections_top5_area={','.join(f'{a:.2f}' for a in top5)}")
        print(f"[QC] centerlines_in_polygon_ratio={center_in_poly_ratio:.3f}")
        road_area_m2 = float(road_poly.area)
        road_perim_m = float(road_poly.length)
        road_vertex_count = _polygon_vertex_count(road_poly)
        road_holes_count = _polygon_holes_count(road_poly)
        road_roughness = _polygon_roughness(road_perim_m, road_area_m2)

        inter_geom = unary_union(inter_polys_final) if inter_polys_final else Polygon()
        inter_area_m2 = inter_area_total_final
        inter_perim_m = float(inter_geom.length) if not inter_geom.is_empty else 0.0
        inter_vertex_count = _polygon_vertex_count(inter_geom)
        inter_holes_count = _polygon_holes_count(inter_geom)
        inter_roughness = _polygon_roughness(inter_perim_m, inter_area_m2)
        inter_circularities = [_intersection_circularity(p) for p in inter_polys_final]
        inter_aspects = [_intersection_aspect_ratio(p) for p in inter_polys_final]
        inter_overlaps = [_intersection_overlap_with_road(p, road_poly) for p in inter_polys_final]
        inter_arm_counts = [
            _arm_count(p, center_lines, float(refine_cfg["arm_buffer_m"])) for p in inter_polys_final
        ]

        def _mean(vals: list[float]) -> float:
            return float(sum(vals) / len(vals)) if vals else 0.0

        intersections_circularity = _mean(inter_circularities)
        intersections_aspect = _mean(inter_aspects)
        intersections_overlap = _mean(inter_overlaps)
        intersections_arms = _mean([float(v) for v in inter_arm_counts]) if inter_arm_counts else 0.0
        if total_len < 200.0 or (road_diag > 0 and total_len < 0.2 * road_diag):
            raise SystemExit("ERROR: centerlines too short; check offset/clip/trajectory coverage.")
        if len(inter_polys_final) > max(20, inter_topk):
            msg = "intersections unstable; check candidates/postprocess thresholds."
            if bool(args.allow_empty_intersections):
                print(f"[GEOM][WARN] {msg}")
            else:
                raise SystemExit(f"ERROR: {msg}")
        if inter_area_total_final < max(100.0, inter_min_area):
            msg = "intersections unstable; check candidates/postprocess thresholds."
            if bool(args.allow_empty_intersections):
                print(f"[GEOM][WARN] {msg}")
            else:
                raise SystemExit(f"ERROR: {msg}")

        snap = {
            "run_id": run_id,
            "drive": args.drive,
            "max_frames": args.max_frames,
            "frame_start": frame_start,
            "frame_count": frame_count,
            "crs_epsg": 32632,
            "backend": backend_used,
            "data_root": str(data_root),
            "oxts_dir": str(oxts_dir),
            "velodyne_dir": str(velodyne_dir),
            "grid_resolution_m": resolution,
            "density_threshold": density_thr,
            "corridor_m": corridor_m,
            "outputs": {
                "road_polygon": "road_polygon.geojson",
                "centerlines": "centerlines.geojson",
                "centerlines_single": "centerlines_single.geojson",
                "centerlines_dual": "centerlines_dual.geojson",
                "centerlines_both": "centerlines_both.geojson",
                "centerlines_auto": "centerlines_auto.geojson",
                "intersections": "intersections.geojson",
                "intersections_algo": "intersections_algo.geojson",
                "intersections_sat": "intersections_sat.geojson",
                "road_polygon_wgs84": "road_polygon_wgs84.geojson",
                "centerlines_wgs84": "centerlines_wgs84.geojson",
                "centerlines_single_wgs84": "centerlines_single_wgs84.geojson",
                "centerlines_dual_wgs84": "centerlines_dual_wgs84.geojson",
                "centerlines_both_wgs84": "centerlines_both_wgs84.geojson",
                "centerlines_auto_wgs84": "centerlines_auto_wgs84.geojson",
                "intersections_wgs84": "intersections_wgs84.geojson",
                "intersections_algo_wgs84": "intersections_algo_wgs84.geojson",
                "intersections_sat_wgs84": "intersections_sat_wgs84.geojson",
                "crs": "crs.json",
            },
        }
        (out_dir / "StateSnapshot.md").write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")

        qc = {
            "road_bbox_dx_m": round(road_dx, 3),
            "road_bbox_dy_m": round(road_dy, 3),
            "road_bbox_diag_m": round(road_diag, 3),
            "frame_start": int(frame_start),
            "frame_count": int(frame_count),
            "road_component_count_before": int(road_before),
            "road_component_count_after": int(road_after),
            "centerline_1_length_m": round(line_lengths[0], 3) if len(line_lengths) > 0 else 0.0,
            "centerline_2_length_m": round(line_lengths[1], 3) if len(line_lengths) > 1 else 0.0,
            "centerline_total_length_m": round(total_len, 3),
            "centerlines_single_cnt": int(len(single_lines)),
            "centerlines_single_len_m": round(single_len, 3),
            "centerlines_dual_cnt": int(len(dual_lines)),
            "centerlines_dual_len_m": round(dual_len, 3),
            "dual_ratio": round((dual_len / max(1e-6, single_len)), 4),
            "dual_triggered_segments": int(dual_triggered_segments),
            "centerlines_segments_total": int(total_segments),
            "centerlines_avg_offset_m": round(avg_offset, 3) if avg_offset is not None else None,
            "centerlines_in_polygon_ratio": round(center_in_poly_ratio, 4),
            "divider_found": 1 if center_result.get("divider_found") else 0,
            "divider_src": center_result.get("divider_src"),
            "split_success": 1 if center_result.get("divider_found") and dual_triggered else 0,
            "intersections_count": int(len(inter_polys_final)),
            "intersections_area_total_m2": round(inter_area_total_final, 3),
            "intersections_top5_area_m2": [round(a, 3) for a in top5],
            "polygon_area_m2": round(road_area_m2, 3),
            "polygon_vertex_count": int(road_vertex_count),
            "polygon_roughness": round(road_roughness, 6),
            "polygon_holes_count": int(road_holes_count),
            "vertex_count": int(road_vertex_count),
            "roughness": round(road_roughness, 6),
            "intersections_vertex_count": int(inter_vertex_count),
            "intersections_roughness": round(inter_roughness, 6),
            "intersections_holes_count": int(inter_holes_count),
            "intersections_circularity": round(intersections_circularity, 4),
            "intersections_aspect_ratio": round(intersections_aspect, 4),
            "intersections_overlap_road": round(intersections_overlap, 4),
            "intersections_arm_count": round(intersections_arms, 3),
            "width_median_m": round(width_median, 3),
            "width_p95_m": round(width_p95, 3),
            "peak_point_count": int(peak_point_count),
            "cluster_count": int(cluster_count),
            "inter_component_count_before": int(inter_before_final),
            "inter_component_count_after": int(inter_after_final),
            "grid_resolution_m": round(resolution, 3),
            "density_threshold": int(density_thr),
            "corridor_m": round(corridor_m, 3),
            "simplify_m": round(poly_simplify_m, 3),
            "centerline_offset_m": round(center_offset_m, 3),
        }
        (out_dir / "qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")

        smooth_compare_path = None
        if smooth_compare:
            smooth_compare_path = out_dir / "smooth_compare.json"
            smooth_compare_path.write_text(
                json.dumps(smooth_compare, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        geom_summary.update(
            {
                "backend_used": backend_used,
                "backend": backend_used,
                "model_id": model_id,
                "mask_threshold": float(args.nn_mask_threshold),
                "nn_fixed": bool(nn_fixed),
                "nn_best_cfg_path": str(nn_best_cfg_path) if nn_fixed else None,
                "bbox_dx_m": qc["road_bbox_dx_m"],
                "bbox_dy_m": qc["road_bbox_dy_m"],
                "bbox_diag_m": qc["road_bbox_diag_m"],
                "road_bbox_dx_m": qc["road_bbox_dx_m"],
                "road_bbox_dy_m": qc["road_bbox_dy_m"],
                "road_bbox_diag_m": qc["road_bbox_diag_m"],
                "centerline_total_len_m": qc["centerline_total_length_m"],
                "centerline_total_length_m": qc["centerline_total_length_m"],
                "centerlines_in_polygon_ratio": qc["centerlines_in_polygon_ratio"],
                "centerlines_single_cnt": qc["centerlines_single_cnt"],
                "centerlines_single_len_m": qc["centerlines_single_len_m"],
                "centerlines_dual_cnt": qc["centerlines_dual_cnt"],
                "centerlines_dual_len_m": qc["centerlines_dual_len_m"],
                "dual_ratio": qc["dual_ratio"],
                "dual_triggered_segments": qc["dual_triggered_segments"],
                "centerlines_segments_total": qc["centerlines_segments_total"],
                "centerlines_avg_offset_m": qc["centerlines_avg_offset_m"],
                "road_component_count_before": qc["road_component_count_before"],
                "road_component_count_after": qc["road_component_count_after"],
                "intersections_count": qc["intersections_count"],
                "intersections_area_total_m2": qc["intersections_area_total_m2"],
                "polygon_area_m2": qc["polygon_area_m2"],
                "polygon_vertex_count": qc["polygon_vertex_count"],
                "polygon_roughness": qc["polygon_roughness"],
                "polygon_holes_count": qc["polygon_holes_count"],
                "vertex_count": qc["vertex_count"],
                "roughness": qc["roughness"],
                "intersections_vertex_count": qc["intersections_vertex_count"],
                "intersections_roughness": qc["intersections_roughness"],
                "intersections_holes_count": qc["intersections_holes_count"],
                "intersections_circularity": qc["intersections_circularity"],
                "intersections_aspect_ratio": qc["intersections_aspect_ratio"],
                "intersections_overlap_road": qc["intersections_overlap_road"],
                "intersections_arm_count": qc["intersections_arm_count"],
                "width_median_m": qc["width_median_m"],
                "width_p95_m": qc["width_p95_m"],
                "centerline_mode": centerline_mode,
                "dual_offset_m": round(dual_offset_used, 3) if dual_offset_used is not None else None,
                "dual_sep_m": round(dual_offset_used, 3) if dual_offset_used is not None else None,
                "dual_width_thresh_m": round(center_cfg["dual_width_threshold_m"], 3),
                "dual_min_len_m": round(center_cfg["min_segment_length_m"], 3),
                "dual_conf_threshold": round(float(center_cfg.get("dual_conf_threshold", 0.0)), 3),
                "divider_sources": center_cfg.get("divider_sources"),
                "divider_found": 1 if center_result.get("divider_found") else 0,
                "divider_src": center_result.get("divider_src"),
                "split_success": 1 if center_result.get("divider_found") and dual_triggered else 0,
                "centerline_dual_ratio": round(center_dual_ratio, 4),
                "post_mask_close_m": round(post_mask_close_m, 3),
                "post_mask_open_m": round(post_mask_open_m, 3),
                "post_min_component_m2": round(post_min_component_m2, 3),
                "post_fill_holes_m2": round(post_fill_holes_m2, 3),
                "poly_simplify_m": round(poly_simplify_m, 3),
                "poly_smooth_buf_m": round(poly_smooth_buf_m, 3),
                "smooth_profile": smooth_profile,
                "strong_min_part_area_m2": round(strong_min_part_area_m2, 3),
                "strong_close_m": round(strong_close_m, 3),
                "strong_open_m": round(strong_open_m, 3),
                "strong_simplify_m": round(strong_simplify_m, 3),
                "strong_min_hole_area_m2": round(strong_min_hole_area_m2, 3),
                "smooth_compare_path": str(smooth_compare_path) if smooth_compare_path else None,
                "inter_simplify_m": round(inter_simplify_m, 3),
                "inter_smooth_buf_m": round(inter_smooth_buf_m, 3),
                "intersection_backend_used": intersection_backend_used,
                "sat_present": bool(sat_present),
                "sat_confidence_avg": round(sat_avg_conf, 4),
            }
        )

        _write_geojson_wgs84(out_dir / "road_polygon_wgs84.geojson", [{"type": "Feature", "geometry": mapping(road_poly), "properties": {}}], wgs84)
        _write_geojson_wgs84(out_dir / "intersections_wgs84.geojson", inter_features, wgs84)
        _write_geojson_wgs84(out_dir / "intersections_algo_wgs84.geojson", inter_features_algo, wgs84)
        crs_meta = {
            "internal_epsg": 32632,
            "wgs84": 4326,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "backend": backend_used,
        }
        (out_dir / "crs.json").write_text(json.dumps(crs_meta, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[GEOM] DONE -> {out_dir}")
        print("outputs:")
        print(f"- {out_dir / 'road_polygon.geojson'}")
        print(f"- {out_dir / 'centerlines.geojson'}")
        print(f"- {out_dir / 'centerlines_single.geojson'}")
        print(f"- {out_dir / 'centerlines_dual.geojson'}")
        print(f"- {out_dir / 'centerlines_both.geojson'}")
        print(f"- {out_dir / 'centerlines_auto.geojson'}")
        print(f"- {out_dir / 'intersections.geojson'}")
        print(f"- {out_dir / 'intersections_algo.geojson'}")
        print(f"- {out_dir / 'intersections_sat.geojson'}")
        print(f"- {out_dir / 'road_polygon_wgs84.geojson'}")
        print(f"- {out_dir / 'centerlines_wgs84.geojson'}")
        print(f"- {out_dir / 'centerlines_single_wgs84.geojson'}")
        print(f"- {out_dir / 'centerlines_dual_wgs84.geojson'}")
        print(f"- {out_dir / 'centerlines_both_wgs84.geojson'}")
        print(f"- {out_dir / 'centerlines_auto_wgs84.geojson'}")
        print(f"- {out_dir / 'intersections_wgs84.geojson'}")
        print(f"- {out_dir / 'intersections_algo_wgs84.geojson'}")
        print(f"- {out_dir / 'intersections_sat_wgs84.geojson'}")
        print(f"- {out_dir / 'crs.json'}")
        status = "PASS"
    except BaseException as exc:
        reason = str(exc) or exc.__class__.__name__
        print(f"[GEOM][ERROR] {reason}")
    finally:
        geom_summary["backend_used"] = backend_used
        geom_summary["backend"] = backend_used
        geom_summary["model_id"] = model_id
        if "nn_fixed" in locals():
            geom_summary["nn_fixed"] = bool(nn_fixed)
            geom_summary["nn_best_cfg_path"] = str(nn_best_cfg_path) if nn_fixed else None
        geom_summary["status"] = status
        geom_summary["reason"] = reason
        (out_dir / "GeomSummary.json").write_text(
            json.dumps(geom_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if status != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations
from pathlib import Path
import argparse
import json
import math
import os
import re
from typing import Iterable

import numpy as np
from shapely.geometry import LineString, Point, box, mapping
from shapely.ops import unary_union, linemerge
from shapely.prepared import prep
from pyproj import Transformer

from pipeline._io import ensure_dir, new_run_id


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


def _build_centerlines(line: LineString, road_poly) -> list[LineString]:
    line_len = float(line.length)
    base = line.simplify(0.5)
    coords = np.asarray(base.coords)
    for offset in (3.5, 2.5, 1.5, 0.8):
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
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    return []


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


def _width_profile(line: LineString, road_poly, step_m: float) -> tuple[list[tuple[float, float]], list[float]]:
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
        w1 = _ray_intersections(road_poly, (coords[i, 0], coords[i, 1]), (nx, ny))
        w2 = _ray_intersections(road_poly, (coords[i, 0], coords[i, 1]), (-nx, -ny))
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="2013_05_28_drive_0000_sync", help="KITTI-360 drive name")
    ap.add_argument("--max-frames", type=int, default=2000, help="max frames to read")
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
    args = ap.parse_args()

    data_root = os.environ.get("POC_DATA_ROOT", "")
    if not data_root:
        raise SystemExit("ERROR: POC_DATA_ROOT is not set.")
    data_root = Path(data_root)

    run_id = new_run_id("geom")
    run_dir = ensure_dir(Path(__file__).resolve().parents[1] / "runs" / run_id)
    out_dir = ensure_dir(run_dir / "outputs")

    oxts_dir = _find_oxts_dir(data_root, args.drive)
    velodyne_dir = _find_velodyne_dir(data_root, args.drive)
    oxts_pts = _read_latlon(oxts_dir, args.max_frames)
    poses_xy, yaws = _project_to_utm32(oxts_pts)

    bin_files = sorted(velodyne_dir.glob("*.bin"))
    if args.max_frames and args.max_frames > 0:
        bin_files = bin_files[: args.max_frames]
    n = min(len(bin_files), poses_xy.shape[0])
    if n < 2:
        raise SystemExit("ERROR: insufficient velodyne frames.")
    bin_files = bin_files[:n]
    poses_xy = poses_xy[:n]
    yaws = yaws[:n]

    resolution = float(args.grid_resolution)
    density_thr = int(args.density_thr)
    corridor_m = float(args.corridor_m)
    simplify_m = float(args.simplify_m)

    origin = (float(poses_xy[0, 0]), float(poses_xy[0, 1]))
    grid: dict[tuple[int, int], int] = {}

    for i in range(n):
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
    road_poly = road_poly.simplify(simplify_m, preserve_topology=True)

    road_before = len(_explode_polygons(road_poly))
    road_min_area = float(args.road_min_area)
    road_topk = int(args.road_topk)
    road_keep = _clean_polygons(road_poly, road_min_area, road_topk)
    road_after = len(road_keep)
    road_poly = unary_union(road_keep) if road_keep else road_poly

    center_lines = _build_centerlines(traj_line, road_poly)
    center_features = [
        {"type": "Feature", "geometry": mapping(center_lines[0]), "properties": {"name": "left"}},
        {"type": "Feature", "geometry": mapping(center_lines[1]), "properties": {"name": "right"}},
    ]

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
        inter_polys, inter_area_total = _postprocess_intersections(
            inter_union,
            road_poly,
            min_area=inter_min_area,
            topk=inter_topk,
            simplify_m=inter_simplify,
        )
        inter_before = len(inter_polys)
        inter_union2 = unary_union(inter_polys) if inter_polys else inter_union
        inter_clean = _clean_polygons(inter_union2, inter_min_area, inter_topk)
        inter_clean = [
            p.simplify(inter_simplify, preserve_topology=True).buffer(0)
            for p in inter_clean
            if not p.is_empty
        ]
        inter_clean = [p for p in inter_clean if p.area >= inter_min_area]
        inter_polys = inter_clean
        inter_area_total = sum(p.area for p in inter_polys)
        inter_after = len(inter_polys)
    inter_features = [{"type": "Feature", "geometry": mapping(p), "properties": {}} for p in inter_polys]

    _write_geojson(out_dir / "centerlines.geojson", center_features)
    _write_geojson(out_dir / "road_polygon.geojson", [{"type": "Feature", "geometry": mapping(road_poly), "properties": {}}])
    _write_geojson(out_dir / "intersections.geojson", inter_features)

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
    print(f"[QC] road_component_count_before={road_before} after={road_after}")
    print(f"[QC] inter_component_count_before={inter_before if inter_pts else 0} after={inter_after if inter_pts else 0}")
    print(f"[QC] width_median={width_median:.2f} width_p95={width_p95:.2f} peak_point_count={peak_point_count} cluster_count={cluster_count}")
    top5 = sorted([p.area for p in inter_polys], reverse=True)[:5]
    print(f"[QC] intersections_count={len(inter_polys)}")
    print(f"[QC] intersections_area_total={inter_area_total:.2f}m2")
    print(f"[QC] intersections_top5_area={','.join(f'{a:.2f}' for a in top5)}")
    print(f"[QC] centerlines_in_polygon_ratio={center_in_poly_ratio:.3f}")
    if total_len < 200.0 or (road_diag > 0 and total_len < 0.2 * road_diag):
        raise SystemExit("ERROR: centerlines too short; check offset/clip/trajectory coverage.")
    if len(inter_polys) > max(20, inter_topk):
        raise SystemExit("ERROR: intersections unstable; check candidates/postprocess thresholds.")
    if inter_area_total < max(100.0, inter_min_area):
        raise SystemExit("ERROR: intersections unstable; check candidates/postprocess thresholds.")

    snap = {
        "run_id": run_id,
        "drive": args.drive,
        "max_frames": args.max_frames,
        "crs_epsg": 32632,
        "data_root": str(data_root),
        "oxts_dir": str(oxts_dir),
        "velodyne_dir": str(velodyne_dir),
        "grid_resolution_m": resolution,
        "density_threshold": density_thr,
        "corridor_m": corridor_m,
        "outputs": {
            "road_polygon": "road_polygon.geojson",
            "centerlines": "centerlines.geojson",
            "intersections": "intersections.geojson",
        },
    }
    (out_dir / "StateSnapshot.md").write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")

    qc = {
        "road_bbox_dx_m": round(road_dx, 3),
        "road_bbox_dy_m": round(road_dy, 3),
        "road_bbox_diag_m": round(road_diag, 3),
        "road_component_count_before": int(road_before),
        "road_component_count_after": int(road_after),
        "centerline_1_length_m": round(line_lengths[0], 3) if len(line_lengths) > 0 else 0.0,
        "centerline_2_length_m": round(line_lengths[1], 3) if len(line_lengths) > 1 else 0.0,
        "centerline_total_length_m": round(total_len, 3),
        "centerlines_in_polygon_ratio": round(center_in_poly_ratio, 4),
        "intersections_count": int(len(inter_polys)),
        "intersections_area_total_m2": round(inter_area_total, 3),
        "intersections_top5_area_m2": [round(a, 3) for a in top5],
        "width_median_m": round(width_median, 3),
        "width_p95_m": round(width_p95, 3),
        "peak_point_count": int(peak_point_count),
        "cluster_count": int(cluster_count),
        "inter_component_count_before": int(inter_before if inter_pts else 0),
        "inter_component_count_after": int(inter_after if inter_pts else 0),
        "grid_resolution_m": round(resolution, 3),
        "density_threshold": int(density_thr),
        "corridor_m": round(corridor_m, 3),
        "simplify_m": round(simplify_m, 3),
    }
    (out_dir / "qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[GEOM] DONE -> {out_dir}")
    print("outputs:")
    print(f"- {out_dir / 'road_polygon.geojson'}")
    print(f"- {out_dir / 'centerlines.geojson'}")
    print(f"- {out_dir / 'intersections.geojson'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

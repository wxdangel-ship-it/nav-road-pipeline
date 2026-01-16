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
from shapely.ops import unary_union
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


def _build_centerlines(line: LineString, road_poly) -> list[LineString]:
    for offset in (3.5, 2.0, 1.5):
        left = line.parallel_offset(offset, "left", join_style=2)
        right = line.parallel_offset(offset, "right", join_style=2)
        left_clip = left.intersection(road_poly)
        right_clip = right.intersection(road_poly)
        left_line = _longest_line(left_clip) if not left_clip.is_empty else None
        right_line = _longest_line(right_clip) if not right_clip.is_empty else None
        if left_line is not None and right_line is not None:
            return [left_line, right_line]
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


def _write_geojson(path: Path, features: list[dict]) -> None:
    fc = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="2013_05_28_drive_0000_sync", help="KITTI-360 drive name")
    ap.add_argument("--max-frames", type=int, default=2000, help="max frames to read")
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

    resolution = 0.5
    density_thr = 3
    corridor_m = 15.0
    simplify_m = 1.2

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
        road_poly = unary_union(cells).simplify(simplify_m, preserve_topology=True)
    else:
        road_poly = corridor.simplify(simplify_m, preserve_topology=True)

    center_lines = _build_centerlines(traj_line, road_poly)
    center_features = [
        {"type": "Feature", "geometry": mapping(center_lines[0]), "properties": {"name": "left"}},
        {"type": "Feature", "geometry": mapping(center_lines[1]), "properties": {"name": "right"}},
    ]

    inter_pts = _build_intersection_points(poses_xy, turn_thr_rad=0.9, close_thr_m=12.0, min_sep=25)
    if inter_pts:
        inter_union = unary_union([p.buffer(15.0) for p in inter_pts]).intersection(road_poly)
        if inter_union.is_empty:
            inter_features = []
        else:
            inter_features = [{"type": "Feature", "geometry": mapping(inter_union), "properties": {}}]
    else:
        inter_features = []

    _write_geojson(out_dir / "centerlines.geojson", center_features)
    _write_geojson(out_dir / "road_polygon.geojson", [{"type": "Feature", "geometry": mapping(road_poly), "properties": {}}])
    _write_geojson(out_dir / "intersections.geojson", inter_features)

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

    print(f"[GEOM] DONE -> {out_dir}")
    print("outputs:")
    print(f"- {out_dir / 'road_polygon.geojson'}")
    print(f"- {out_dir / 'centerlines.geojson'}")
    print(f"- {out_dir / 'intersections.geojson'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

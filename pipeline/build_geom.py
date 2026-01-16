from __future__ import annotations
from pathlib import Path
import argparse
import json
import math
import os
import re
from copy import deepcopy

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
    pts = []
    for fp in files:
        text = fp.read_text(encoding="utf-8").strip()
        if not text:
            continue
        parts = re.split(r"\s+", text)
        if len(parts) < 2:
            continue
        try:
            lat = float(parts[0])
            lon = float(parts[1])
        except ValueError:
            continue
        pts.append((lat, lon))
    if len(pts) < 2:
        raise SystemExit("ERROR: insufficient oxts points.")
    return pts


def _wgs84_to_utm32n(lat: float, lon: float) -> tuple[float, float]:
    a = 6378137.0
    f = 1.0 / 298.257223563
    k0 = 0.9996
    e2 = f * (2.0 - f)
    ep2 = e2 / (1.0 - e2)

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lon0 = math.radians(9.0)  # UTM zone 32 central meridian

    n = a / math.sqrt(1.0 - e2 * math.sin(lat_rad) ** 2)
    t = math.tan(lat_rad) ** 2
    c = ep2 * math.cos(lat_rad) ** 2
    a_ = (lon_rad - lon0) * math.cos(lat_rad)

    m = (
        a
        * (
            (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat_rad
            - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * lat_rad)
            + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * lat_rad)
            - (35 * e2 ** 3 / 3072) * math.sin(6 * lat_rad)
        )
    )

    x = k0 * n * (
        a_
        + (1 - t + c) * a_ ** 3 / 6
        + (5 - 18 * t + t ** 2 + 72 * c - 58 * ep2) * a_ ** 5 / 120
    ) + 500000.0
    y = k0 * (
        m
        + n
        * math.tan(lat_rad)
        * (
            a_ ** 2 / 2
            + (5 - t + 9 * c + 4 * c ** 2) * a_ ** 4 / 24
            + (61 - 58 * t + t ** 2 + 600 * c - 330 * ep2) * a_ ** 6 / 720
        )
    )
    return x, y


def _project_to_utm32(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [_wgs84_to_utm32n(lat, lon) for lat, lon in points]


def _offset_polyline(points: list[tuple[float, float]], offset: float) -> list[tuple[float, float]]:
    n = len(points)
    if n < 2:
        return points
    normals = []
    for i in range(n - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            normals.append((0.0, 0.0))
        else:
            normals.append((-dy / length, dx / length))
    out = []
    for i in range(n):
        if i == 0:
            nx, ny = normals[0]
        elif i == n - 1:
            nx, ny = normals[-1]
        else:
            n1 = normals[i - 1]
            n2 = normals[i]
            nx = n1[0] + n2[0]
            ny = n1[1] + n2[1]
            length = math.hypot(nx, ny)
            if length != 0:
                nx /= length
                ny /= length
        x, y = points[i]
        out.append((x + nx * offset, y + ny * offset))
    return out


def _rdp(points: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points
    x1, y1 = points[0]
    x2, y2 = points[-1]
    dx = x2 - x1
    dy = y2 - y1
    denom = math.hypot(dx, dy)
    max_dist = -1.0
    idx = -1
    for i in range(1, len(points) - 1):
        px, py = points[i]
        if denom == 0:
            dist = math.hypot(px - x1, py - y1)
        else:
            dist = abs(dy * px - dx * py + x2 * y1 - y2 * x1) / denom
        if dist > max_dist:
            max_dist = dist
            idx = i
    if max_dist > eps and idx != -1:
        left = _rdp(points[: idx + 1], eps)
        right = _rdp(points[idx:], eps)
        return left[:-1] + right
    return [points[0], points[-1]]


def _build_intersections(
    points: list[tuple[float, float]],
    threshold_m: float = 12.0,
    min_sep: int = 20,
) -> list[tuple[float, float]]:
    n = len(points)
    if n < min_sep + 2:
        return []
    step = max(1, n // 400)
    sampled = points[::step]
    centers = []
    for i in range(len(sampled)):
        x1, y1 = sampled[i]
        for j in range(i + min_sep, len(sampled)):
            x2, y2 = sampled[j]
            d = math.hypot(x2 - x1, y2 - y1)
            if d < threshold_m:
                centers.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
    return centers


def _circle_polygon(center: tuple[float, float], radius: float, steps: int = 16) -> list[tuple[float, float]]:
    cx, cy = center
    pts = []
    for i in range(steps):
        ang = 2.0 * math.pi * i / steps
        pts.append((cx + math.cos(ang) * radius, cy + math.sin(ang) * radius))
    pts.append(pts[0])
    return pts


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
    ll = _read_latlon(oxts_dir, args.max_frames)
    xy = _project_to_utm32(ll)

    center_offset_m = 3.5
    road_buffer_m = 10.0
    simplify_m = 1.0

    left = _offset_polyline(xy, center_offset_m)
    right = _offset_polyline(xy, -center_offset_m)
    left = _rdp(left, simplify_m)
    right = _rdp(right, simplify_m)

    road_left = _offset_polyline(xy, road_buffer_m)
    road_right = _offset_polyline(xy, -road_buffer_m)
    road_poly = road_left + list(reversed(road_right))
    if road_poly[0] != road_poly[-1]:
        road_poly.append(road_poly[0])
    road_poly = _rdp(road_poly, simplify_m)

    inter_centers = _build_intersections(xy, threshold_m=12.0, min_sep=20)
    inter_features = []
    for c in inter_centers:
        ring = _circle_polygon(c, 8.0, steps=16)
        inter_features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": {}})

    center_features = [
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": left}, "properties": {"name": "left"}},
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": right}, "properties": {"name": "right"}},
    ]

    _write_geojson(out_dir / "centerlines.geojson", center_features)
    _write_geojson(out_dir / "road_polygon.geojson", [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [road_poly]}, "properties": {}}])
    _write_geojson(out_dir / "intersections.geojson", inter_features)

    snap = {
        "run_id": run_id,
        "drive": args.drive,
        "max_frames": args.max_frames,
        "crs_epsg": 32632,
        "data_root": str(data_root),
        "oxts_dir": str(oxts_dir),
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

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import geopandas as gpd
import pandas as pd
try:
    import pyogrio
except Exception:
    pyogrio = None
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Polygon

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.build_geom import _find_oxts_dir, _load_kitti360_calib


def _read_oxts_frame(oxts_dir: Path, frame_id: str) -> Optional[Tuple[float, float, float]]:
    path = oxts_dir / f"{frame_id}.txt"
    if not path.exists():
        return None
    parts = path.read_text(encoding="utf-8").strip().split()
    if len(parts) < 6:
        return None
    lat = float(parts[0])
    lon = float(parts[1])
    yaw = float(parts[5])
    return lat, lon, yaw


def _utm32_transform():
    return Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)


def _pixel_to_world(
    u: float,
    v: float,
    k: np.ndarray,
    r_rect: np.ndarray,
    cam_to_velo: np.ndarray,
    pose_xy: Tuple[float, float],
    yaw: float,
) -> Optional[Tuple[float, float]]:
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    if fx == 0 or fy == 0:
        return None
    dir_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)
    r_rect_inv = np.linalg.inv(r_rect)
    dir_cam = r_rect_inv.dot(dir_cam)
    dir_velo = cam_to_velo[:3, :3].dot(dir_cam)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    r_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    dir_world = r_yaw.dot(dir_velo)
    cam_offset = cam_to_velo[:3, 3]
    origin_z = -float(cam_offset[2])
    origin = np.array(
        [pose_xy[0] + c * cam_offset[0] - s * cam_offset[1], pose_xy[1] + s * cam_offset[0] + c * cam_offset[1], origin_z],
        dtype=float,
    )
    if dir_world[2] >= -1e-6:
        return None
    t = -origin[2] / dir_world[2]
    if t <= 0:
        return None
    hit = origin + t * dir_world
    return float(hit[0]), float(hit[1])


def _project_coords(
    coords: Iterable[Tuple[float, float]],
    k: np.ndarray,
    r_rect: np.ndarray,
    cam_to_velo: np.ndarray,
    pose_xy: Tuple[float, float],
    yaw: float,
) -> list[Tuple[float, float]]:
    out = []
    for u, v in coords:
        pt = _pixel_to_world(u, v, k, r_rect, cam_to_velo, pose_xy, yaw)
        if pt is not None:
            out.append(pt)
    return out


def _project_geometry(
    geom,
    k: np.ndarray,
    r_rect: np.ndarray,
    cam_to_velo: np.ndarray,
    pose_xy: Tuple[float, float],
    yaw: float,
):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        coords = _project_coords(geom.coords, k, r_rect, cam_to_velo, pose_xy, yaw)
        if len(coords) < 2:
            return None
        return LineString(coords)
    if geom.geom_type == "Polygon":
        coords = _project_coords(geom.exterior.coords, k, r_rect, cam_to_velo, pose_xy, yaw)
        if len(coords) < 3:
            return None
        return Polygon(coords)
    if geom.geom_type == "MultiLineString":
        parts = []
        for part in geom.geoms:
            proj = _project_geometry(part, k, r_rect, cam_to_velo, pose_xy, yaw)
            if proj is not None:
                parts.append(proj)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return LineString([pt for geom in parts for pt in geom.coords])
    if geom.geom_type == "MultiPolygon":
        parts = []
        for part in geom.geoms:
            proj = _project_geometry(part, k, r_rect, cam_to_velo, pose_xy, yaw)
            if proj is not None:
                parts.append(proj)
        if not parts:
            return None
        return parts[0]
    return None


def _write_layers(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    if gdf.empty:
        return
    for cls, sub in gdf.groupby("class"):
        sub.to_file(out_path, layer=str(cls), driver="GPKG")


def _read_all_layers(path: Path) -> gpd.GeoDataFrame:
    layers = []
    if pyogrio is not None:
        try:
            layers = [name for name, _ in pyogrio.list_layers(path)]
        except Exception:
            layers = []
    if not layers:
        try:
            layers = gpd.io.file.fiona.listlayers(str(path))
        except Exception:
            layers = []
    if not layers:
        return gpd.GeoDataFrame()
    frames = []
    for layer in layers:
        try:
            gdf = gpd.read_file(path, layer=layer)
        except Exception:
            continue
        if "class" not in gdf.columns:
            gdf["class"] = layer
        frames.append(gdf)
    if not frames:
        return gpd.GeoDataFrame()
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    ap.add_argument("--feature-store", required=True, help="image_px feature_store dir")
    ap.add_argument("--out-store", required=True, help="output feature_store_map dir")
    ap.add_argument("--data-root", default="", help="KITTI-360 root (default=POC_DATA_ROOT)")
    ap.add_argument("--camera", default="image_00")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    data_root = Path(args.data_root) if args.data_root else Path(os.environ.get("POC_DATA_ROOT", ""))
    if not data_root.exists():
        raise SystemExit("ERROR: POC_DATA_ROOT not set or invalid.")

    feature_store = Path(args.feature_store)
    out_store = Path(args.out_store)
    out_store.mkdir(parents=True, exist_ok=True)

    drive_dir = feature_store / args.drive
    if not drive_dir.exists():
        raise SystemExit(f"ERROR: feature_store missing drive: {drive_dir}")

    oxts_dir = _find_oxts_dir(data_root, args.drive)
    calib = _load_kitti360_calib(data_root, args.drive, args.camera)
    p_rect = calib["p_rect"]
    r_rect = calib["r_rect"][:3, :3]
    cam_to_velo = np.linalg.inv(calib["t_velo_to_cam"])
    k = p_rect[:3, :3]
    ll_to_utm = _utm32_transform()

    frames = sorted([p for p in drive_dir.iterdir() if p.is_dir()])
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    counts = {}
    frames_with = {}
    for frame_dir in frames:
        frame_id = frame_dir.name
        pose = _read_oxts_frame(oxts_dir, frame_id)
        if pose is None:
            continue
        lat, lon, yaw = pose
        x, y = ll_to_utm.transform(lon, lat)
        pose_xy = (float(x), float(y))

        in_gpkg = frame_dir / "image_features.gpkg"
        in_geojson = frame_dir / "image_features.geojson"
        if in_gpkg.exists():
            gdf = _read_all_layers(in_gpkg)
        elif in_geojson.exists():
            gdf = gpd.read_file(in_geojson)
        else:
            continue
        if gdf.empty:
            continue

        mapped_rows = []
        for _, row in gdf.iterrows():
            props = dict(row.drop(labels=["geometry"], errors="ignore"))
            geom = row.geometry
            frame = str(props.get("geometry_frame") or "image_px").lower()
            if frame != "image_px":
                mapped = geom
            else:
                mapped = _project_geometry(geom, k, r_rect, cam_to_velo, pose_xy, yaw)
            if mapped is None or mapped.is_empty:
                continue
            props["geometry_frame"] = "map"
            mapped_rows.append({**props, "geometry": mapped})
            cls = str(props.get("class") or "unknown")
            counts[cls] = counts.get(cls, 0) + 1
            frames_with[cls] = frames_with.get(cls, 0) + 1

        if not mapped_rows:
            continue
        out_frame = out_store / args.drive / frame_id
        out_frame.mkdir(parents=True, exist_ok=True)
        out_gpkg = out_frame / "image_features.gpkg"
        out_gdf = gpd.GeoDataFrame(mapped_rows, geometry="geometry", crs="EPSG:32632")
        _write_layers(out_gdf, out_gpkg)

    index = {
        "drive_id": args.drive,
        "geometry_frame": "map",
        "counts": counts,
        "frames": frames_with,
    }
    (out_store / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[FEATURE_STORE_MAP] wrote: {out_store}")
    print(f"[FEATURE_STORE_MAP] counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

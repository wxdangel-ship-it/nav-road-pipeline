from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import unary_union

try:
    import pyogrio
except Exception:
    pyogrio = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.datasets.kitti360_io import (
    load_kitti360_calib,
    load_kitti360_cam_to_pose_key,
    load_kitti360_lidar_points_world,
    load_kitti360_pose,
    load_kitti360_pose_full,
)
from pipeline.calib.kitti360_backproject import BackprojectContext, configure_default_context, pixel_to_world_on_ground
from pipeline.calib.kitti360_projection import project_velo_to_image
from tools.build_image_sample_index import _find_image_dir


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("project_feature_store_to_map")


def _find_latest_clean_dtm(drive_id: str) -> Optional[Path]:
    drive_tag = drive_id.split("_")[-2] if "_" in drive_id else drive_id
    patterns = [
        f"lidar_ground_{drive_tag}_f250_500_*",
        f"lidar_ground_{drive_tag}_*",
    ]
    candidates: list[Path] = []
    for pat in patterns:
        for run_dir in Path("runs").glob(pat):
            cand = run_dir / "rasters" / "dtm_median_utm32.tif"
            if cand.exists():
                candidates.append(cand)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_camera_defaults(path: Path) -> dict:
    if not path.exists():
        return {"default_camera": "image_00", "enforce_camera": True, "allow_override": False}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = {
        "default_camera": "image_00",
        "enforce_camera": True,
        "allow_override": False,
    }
    if isinstance(data, dict):
        defaults.update({k: v for k, v in data.items() if v is not None})
    return defaults


def _assert_camera_consistency(
    camera: str,
    image_dir: Path | None,
    calib: dict | None,
    cam_to_pose_key: str,
    lidar_world_mode: str,
    default_camera: str,
    enforce_camera: bool,
    allow_override: bool,
) -> None:
    if enforce_camera and not allow_override and camera != default_camera:
        raise SystemExit(f"ERROR: camera={camera} expected={default_camera}")
    if image_dir is None or not image_dir.exists():
        raise SystemExit("ERROR: image_dir_missing")
    image_dir_text = str(image_dir).replace("\\", "/").lower()
    if enforce_camera and "image_00" not in image_dir_text:
        raise SystemExit(f"ERROR: image_dir_not_image_00:{image_dir}")
    if enforce_camera and "data_rect" not in image_dir_text:
        raise SystemExit(f"ERROR: image_dir_not_data_rect:{image_dir}")
    if calib is None:
        raise SystemExit("ERROR: calib_missing")
    p_key = str(calib.get("p_rect_key", ""))
    r_key = str(calib.get("r_rect_key", ""))
    if enforce_camera and (p_key != "P_rect_00" or r_key != "R_rect_00"):
        raise SystemExit(f"ERROR: calib_rect_key_mismatch:p={p_key} r={r_key}")
    if enforce_camera and cam_to_pose_key != "image_00":
        raise SystemExit(f"ERROR: cam_to_pose_key_mismatch:{cam_to_pose_key}")
    if enforce_camera and lidar_world_mode != "fullpose":
        raise SystemExit(f"ERROR: lidar_world_mode={lidar_world_mode} expected=fullpose")


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


def _find_velodyne_dir(data_root: Path, drive: str) -> Path:
    candidates = [
        data_root / "data_3d_raw" / drive / "velodyne_points" / "data",
        data_root / "data_3d_raw" / drive / "velodyne_points" / "data" / "1",
        data_root / drive / "velodyne_points" / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"missing_file:velodyne_dir:{drive}")


def _resolve_velodyne_path(velodyne_dir: Path, frame_id: str) -> Optional[Path]:
    direct = velodyne_dir / f"{frame_id}.bin"
    if direct.exists():
        return direct
    if frame_id.isdigit():
        pad = velodyne_dir / f"{int(frame_id):010d}.bin"
        if pad.exists():
            return pad
    return None


def _read_velodyne_points(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size % 4 != 0:
        raw = raw[: raw.size - (raw.size % 4)]
    return raw.reshape(-1, 4)


def _velo_to_world(points: np.ndarray, pose_xy: Tuple[float, float], yaw: float) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    xw = c * x - s * y + pose_xy[0]
    yw = s * x + c * y + pose_xy[1]
    return np.stack([xw, yw, z], axis=1)


def _project_velodyne_to_image(points: np.ndarray, proj_ctx: BackprojectContext) -> Tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return np.empty((0, 2), dtype=float), np.zeros((0,), dtype=bool)
    proj = project_velo_to_image(points[:, :3], proj_ctx.calib, use_rect=True, y_flip_mode="fixed_true", sanity=False)
    uv = np.stack([proj["u"], proj["v"]], axis=1)
    valid = proj["valid"].astype(bool)
    return uv, valid


def _project_geometry_ground_plane(
    geom,
    frame_id: str,
    ground_model: Dict[str, object],
    proj_ctx: BackprojectContext,
):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        coords = []
        for u, v in geom.coords:
            pt = pixel_to_world_on_ground(frame_id, float(u), float(v), ground_model, ctx=proj_ctx)
            if pt is not None:
                coords.append((float(pt[0]), float(pt[1])))
        if len(coords) < 2:
            return None
        return LineString(coords)
    if geom.geom_type == "Polygon":
        coords = []
        for u, v in geom.exterior.coords:
            pt = pixel_to_world_on_ground(frame_id, float(u), float(v), ground_model, ctx=proj_ctx)
            if pt is not None:
                coords.append((float(pt[0]), float(pt[1])))
        if len(coords) < 3:
            return None
        return Polygon(coords)
    if geom.geom_type == "MultiLineString":
        parts = []
        for part in geom.geoms:
            proj = _project_geometry_ground_plane(part, frame_id, ground_model, proj_ctx)
            if proj is not None:
                parts.append(proj)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return LineString([pt for g in parts for pt in g.coords])
    if geom.geom_type == "MultiPolygon":
        parts = []
        for part in geom.geoms:
            proj = _project_geometry_ground_plane(part, frame_id, ground_model, proj_ctx)
            if proj is not None:
                parts.append(proj)
        if not parts:
            return None
        return parts[0]
    return None


def _points_to_linestring(points: np.ndarray) -> Optional[LineString]:
    if points.shape[0] < 2:
        return None
    coords = points[:, :2]
    mean = coords.mean(axis=0)
    cov = np.cov((coords - mean).T)
    vals, vecs = np.linalg.eig(cov)
    idx = int(np.argmax(vals))
    direction = vecs[:, idx]
    proj = (coords - mean) @ direction
    order = np.argsort(proj)
    line = LineString(coords[order].tolist())
    return line


def _points_to_geometry(points: np.ndarray, geom_type: str, min_length: float) -> Optional[object]:
    if points.shape[0] == 0:
        return None
    if geom_type == "LineString":
        line = _points_to_linestring(points)
        if line is None or line.length < min_length:
            return None
        return line
    if geom_type == "Polygon":
        poly = MultiPoint(points[:, :2]).convex_hull
        if poly.is_empty or (hasattr(poly, "area") and poly.area <= 0):
            return None
        return poly
    return MultiPoint(points[:, :2])


def _filter_points_in_geom(uv: np.ndarray, world_pts: np.ndarray, geom, line_buffer_px: float) -> np.ndarray:
    if geom is None or geom.is_empty:
        return np.empty((0, 3), dtype=float)
    if geom.geom_type in {"LineString", "MultiLineString"} and line_buffer_px > 0:
        geom = geom.buffer(line_buffer_px)
    minx, miny, maxx, maxy = geom.bounds
    mask = (
        (uv[:, 0] >= minx)
        & (uv[:, 0] <= maxx)
        & (uv[:, 1] >= miny)
        & (uv[:, 1] <= maxy)
    )
    if not np.any(mask):
        return np.empty((0, 3), dtype=float)
    idxs = np.where(mask)[0]
    pts = []
    for i in idxs:
        if geom.contains(Point(float(uv[i, 0]), float(uv[i, 1]))):
            pts.append(world_pts[i])
    if not pts:
        return np.empty((0, 3), dtype=float)
    return np.asarray(pts, dtype=float)


def _mask_area_px(geom, line_buffer_px: float) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    if geom.geom_type in {"Polygon", "MultiPolygon"}:
        return float(geom.area)
    if geom.geom_type in {"LineString", "MultiLineString"}:
        return float(geom.length) * max(line_buffer_px * 2.0, 1.0)
    return 0.0


def _confidence_value(props: dict) -> float:
    for key in ("conf", "score", "confidence"):
        val = props.get(key)
        if val is not None:
            try:
                return float(val)
            except Exception:
                continue
    return 0.0


def _cap_instances_per_class(
    gdf: gpd.GeoDataFrame, max_instances_per_class: int
) -> gpd.GeoDataFrame:
    if max_instances_per_class <= 0 or gdf.empty:
        return gdf
    if "class" not in gdf.columns:
        return gdf
    rows = []
    for cls, part in gdf.groupby("class"):
        if len(part) <= max_instances_per_class:
            rows.append(part)
            continue
        tmp = part.copy()
        tmp["__conf"] = tmp.apply(lambda r: _confidence_value(dict(r)), axis=1)
        tmp = tmp.sort_values("__conf", ascending=False).head(max_instances_per_class)
        rows.append(tmp.drop(columns=["__conf"], errors="ignore"))
    return gpd.GeoDataFrame(pd.concat(rows, ignore_index=True), geometry="geometry")


def _apply_mask_area_filter(
    gdf: gpd.GeoDataFrame, min_mask_area_px: float, line_buffer_px: float
) -> gpd.GeoDataFrame:
    if min_mask_area_px <= 0 or gdf.empty:
        return gdf
    areas = gdf.geometry.apply(lambda g: _mask_area_px(g, line_buffer_px))
    return gdf.loc[areas >= min_mask_area_px]


def _write_layers(path: Path, by_class: Dict[str, list]) -> None:
    if path.exists():
        path.unlink()
    for cls, feats in by_class.items():
        if not feats:
            continue
        gdf = gpd.GeoDataFrame(
            [f["properties"] for f in feats],
            geometry=[f["geometry"] for f in feats],
            crs="EPSG:32632",
        )
        gdf.to_file(path, layer=cls, driver="GPKG")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    ap.add_argument("--feature-store", required=True, help="image_px feature_store dir")
    ap.add_argument("--out-store", required=True, help="output feature_store_map dir")
    ap.add_argument("--data-root", default="", help="KITTI-360 root (default=POC_DATA_ROOT)")
    ap.add_argument("--camera", default="image_00")
    ap.add_argument("--pose-mode", default="", choices=["", "legacy", "fullpose"])
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--map-mode", default="auto", choices=["lidar_project", "ground_plane", "auto"])
    ap.add_argument("--min-points", type=int, default=30)
    ap.add_argument("--min-length", type=float, default=0.8)
    ap.add_argument("--min-mask-area-px", type=float, default=500.0)
    ap.add_argument("--dtm-path", default="", help="optional clean DTM path (dtm_median_utm32.tif)")
    ap.add_argument("--fixed-plane-z0", type=float, default=0.0, help="fallback ground plane z0 in UTM32")
    ap.add_argument("--max-instances-per-class", type=int, default=200)
    ap.add_argument("--line-buffer-px", type=float, default=2.0)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--debug-root", default="")
    ap.add_argument("--out-name", default="map_evidence_utm32.gpkg")
    args = ap.parse_args()

    log = _setup_logger()
    data_root = Path(args.data_root) if args.data_root else Path(os.environ.get("POC_DATA_ROOT", ""))
    if not data_root.exists():
        log.error("POC_DATA_ROOT not set or invalid.")
        return 2

    defaults = _load_camera_defaults(Path("configs/camera_defaults.yaml"))
    default_camera = str(defaults.get("default_camera") or "image_00")
    enforce_camera = bool(defaults.get("enforce_camera", True))
    allow_override = bool(defaults.get("allow_override", False))
    if str(os.environ.get("ALLOW_CAMERA_OVERRIDE", "0")).strip() == "1":
        allow_override = True
    camera = str(args.camera or default_camera)
    pose_mode = str(args.pose_mode or os.environ.get("LIDAR_WORLD_MODE") or "").strip().lower()
    if str(os.environ.get("USE_FULLPOSE_LIDAR", "0")).strip() == "1":
        pose_mode = "fullpose"
    if not pose_mode:
        pose_mode = "fullpose"

    image_dir = _find_image_dir(data_root, args.drive, camera)
    try:
        calib = load_kitti360_calib(data_root, camera)
    except Exception as exc:
        log.error("calib load failed: %s", exc)
        return 4
    cam_to_pose_key = ""
    try:
        _cam_to_pose, cam_to_pose_key = load_kitti360_cam_to_pose_key(data_root, camera)
    except Exception:
        cam_to_pose_key = ""
    _assert_camera_consistency(
        camera,
        image_dir,
        calib,
        cam_to_pose_key,
        pose_mode,
        default_camera,
        enforce_camera,
        allow_override,
    )

    dtm_path = Path(args.dtm_path) if str(args.dtm_path).strip() else _find_latest_clean_dtm(args.drive)
    ground_mode = "lidar_clean_dtm" if dtm_path else "fixed_plane"
    ground_model = {"mode": ground_mode, "dtm_path": str(dtm_path) if dtm_path else None, "z0": float(args.fixed_plane_z0)}
    proj_ctx = configure_default_context(data_root, args.drive, cam_id=camera, dtm_path=dtm_path)

    feature_store = Path(args.feature_store)
    out_store = Path(args.out_store)
    out_store.mkdir(parents=True, exist_ok=True)

    drive_dir = feature_store / args.drive
    if not drive_dir.exists():
        log.error("feature_store missing drive: %s", drive_dir)
        return 3

    try:
        velodyne_dir = _find_velodyne_dir(data_root, args.drive)
    except Exception as exc:
        log.warning("velodyne not available: %s", exc)
        velodyne_dir = None

    frames = sorted([p for p in drive_dir.iterdir() if p.is_dir()])
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    by_class_all: Dict[str, list] = {}
    errors = []
    debug_samples: Dict[str, list] = {}
    ground_classes = {
        "lane_marking",
        "stop_line",
        "crosswalk",
        "gore_marking",
        "arrow",
        "arrow_marking",
    }

    for frame_dir in frames:
        frame_id = frame_dir.name
        out_frame = out_store / args.drive / frame_id
        out_frame.mkdir(parents=True, exist_ok=True)
        out_path = out_frame / args.out_name
        if args.resume and out_path.exists():
            continue

        try:
            if pose_mode == "fullpose":
                pose = load_kitti360_pose_full(data_root, args.drive, frame_id)
            else:
                pose = load_kitti360_pose(data_root, args.drive, frame_id)
        except Exception as exc:
            errors.append(f"{args.drive}:{frame_id}:pose:{exc}")
            continue

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

        gdf = _apply_mask_area_filter(gdf, args.min_mask_area_px, args.line_buffer_px)
        gdf = _cap_instances_per_class(gdf, args.max_instances_per_class)
        if gdf.empty:
            continue

        lidar_ok = False
        uv = None
        world_pts = None
        if velodyne_dir is not None:
            bin_path = _resolve_velodyne_path(velodyne_dir, frame_id)
            if bin_path and bin_path.exists():
                pts = _read_velodyne_points(bin_path)
                if pose_mode == "fullpose":
                    world_pts = load_kitti360_lidar_points_world(
                        data_root,
                        args.drive,
                        frame_id,
                        mode="fullpose",
                        cam_id=camera,
                    )
                else:
                    world_pts = _velo_to_world(pts[:, :3], (pose[0], pose[1]), pose[2])
                uv, valid = _project_velodyne_to_image(pts[:, :3], proj_ctx)
                uv = uv[valid]
                world_pts = world_pts[valid]
                lidar_ok = True

        by_class: Dict[str, list] = {}
        for _, row in gdf.iterrows():
            props = dict(row.drop(labels=["geometry"], errors="ignore"))
            geom = row.geometry
            cls = str(props.get("class") or "unknown")
            geom_type = geom.geom_type
            map_mode = args.map_mode
            map_reason = ""
            mapped = None
            points_count = 0
            evidence_strength = ""
            map_mode_used = ""
            weak_reason = ""
            is_ground_class = cls in ground_classes

            if map_mode in {"lidar_project", "auto"} and lidar_ok:
                pts = _filter_points_in_geom(uv, world_pts, geom, args.line_buffer_px)
                points_count = int(pts.shape[0])
                if points_count >= args.min_points:
                    mapped = _points_to_geometry(pts, geom_type, args.min_length)
                    if mapped is not None and not mapped.is_empty:
                        map_mode_used = "lidar_project"
                        evidence_strength = "strong"
                else:
                    map_reason = "lidar_points_insufficient"
                    if map_mode == "lidar_project":
                        mapped = None
            if mapped is None and map_mode in {"ground_plane", "auto"} and is_ground_class:
                mapped = _project_geometry_ground_plane(geom, frame_id, ground_model, proj_ctx)
                if mapped is not None:
                    map_mode_used = "ground_plane"
                    evidence_strength = "weak"
                    if map_reason == "lidar_points_insufficient":
                        weak_reason = "insufficient_lidar_points"
                    elif not lidar_ok:
                        weak_reason = "no_lidar"
                    else:
                        weak_reason = "plane_assumed"
            if mapped is None or mapped.is_empty:
                continue
            if mapped.geom_type in {"LineString", "MultiLineString"} and mapped.length < args.min_length:
                continue

            if map_mode_used == "":
                map_mode_used = map_mode
            if evidence_strength == "":
                evidence_strength = "strong" if map_mode_used == "lidar_project" else "weak"

            props["geometry_frame"] = "map"
            props["map_mode_used"] = map_mode_used
            props["evidence_strength"] = evidence_strength
            props["weak_reason"] = weak_reason
            props["map_reason"] = map_reason
            props["points_count"] = points_count
            props["drive_id"] = props.get("drive_id") or args.drive
            props["frame_id"] = props.get("frame_id") or frame_id
            feat = {"geometry": mapped, "properties": props}
            by_class.setdefault(cls, []).append(feat)
            by_class_all.setdefault(cls, []).append(feat)

            if args.debug_root:
                debug_root = Path(args.debug_root)
                overlay = debug_root / args.drive / f"{frame_id}_overlay.png"
                if overlay.exists():
                    debug_samples.setdefault(cls, []).append(
                        {
                            "drive_id": args.drive,
                            "frame_id": frame_id,
                            "class": cls,
                            "points_count": points_count,
                            "overlay": str(overlay),
                        }
                    )

        _write_layers(out_path, by_class)

    drive_root = out_store / args.drive
    drive_root.mkdir(parents=True, exist_ok=True)
    out_drive = drive_root / args.out_name
    _write_layers(out_drive, by_class_all)

    index = {
        "drive_id": args.drive,
        "geometry_frame": "map",
        "map_mode": args.map_mode,
        "counts": {cls: len(feats) for cls, feats in by_class_all.items()},
    }
    (out_store / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    if debug_samples:
        (drive_root / "map_debug_samples.json").write_text(json.dumps(debug_samples, indent=2), encoding="utf-8")

    if errors:
        (drive_root / "map_errors.txt").write_text("\n".join(errors), encoding="utf-8")

    log.info("feature_store_map written: %s", out_drive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

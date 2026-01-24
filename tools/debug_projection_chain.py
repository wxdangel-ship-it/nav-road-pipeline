import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from pipeline.datasets.kitti360_io import (
    load_kitti360_calib,
    load_kitti360_lidar_points_world,
    load_kitti360_pose,
)
from tools.build_image_sample_index import _find_image_dir
from tools.run_crosswalk_monitor_range import _load_crosswalk_raw, _normalize_frame_id, _parse_frame_id


def _load_yaml(path: Path) -> dict:
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _resolve_candidate_source(candidate_source: str) -> Optional[Path]:
    if not candidate_source:
        return None
    path = Path(candidate_source)
    if path.is_dir():
        for cand in [
            path / "outputs" / "frame_candidates_utm32.gpkg",
            path / "frame_candidates_utm32.gpkg",
            path / "outputs" / "crosswalk_entities_utm32.gpkg",
        ]:
            if cand.exists():
                return cand
        return None
    if path.exists():
        return path
    return None


def _list_layers(path: Path) -> List[str]:
    try:
        import pyogrio

        return [name for name, _ in pyogrio.list_layers(path)]
    except Exception:
        return []


def _load_candidate_geom(
    candidate_path: Path,
    drive_id: str,
    frame_id: str,
) -> Optional[Polygon]:
    layers = _list_layers(candidate_path)
    layer = None
    for name in ("frame_candidates", "crosswalk_candidate_poly"):
        if name in layers:
            layer = name
            break
    if layer is None and layers:
        layer = layers[0]
    if layer is None:
        return None
    gdf = gpd.read_file(candidate_path, layer=layer)
    if gdf.empty:
        return None
    if "drive_id" in gdf.columns:
        gdf = gdf[gdf["drive_id"].astype(str) == drive_id]
    if "frame_id" in gdf.columns:
        gdf = gdf[gdf["frame_id"].astype(str) == frame_id]
    if gdf.empty:
        return None
    subset = gdf
    if "proj_method" in gdf.columns:
        lidar = gdf[gdf["proj_method"].astype(str) == "lidar"]
        if not lidar.empty:
            subset = lidar
    if "geom_ok" in subset.columns:
        ok = subset[subset["geom_ok"] == 1]
        if not ok.empty:
            subset = ok
    geom = unary_union(subset.geometry.values)
    if geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    if geom.geom_type == "MultiPolygon":
        polys = list(geom.geoms)
        return polys[0] if polys else None
    return None


def _project_world_to_image(
    points: np.ndarray,
    pose_xy_yaw: Tuple[float, float, float],
    calib: Dict[str, np.ndarray],
    mode: str,
) -> np.ndarray:
    x0, y0, yaw = pose_xy_yaw
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    dx = points[:, 0] - x0
    dy = points[:, 1] - y0
    x_ego = c * dx + s * dy
    y_ego = -s * dx + c * dy
    z_ego = points[:, 2]
    ones = np.ones_like(x_ego)
    pts_h = np.stack([x_ego, y_ego, z_ego, ones], axis=0)
    cam = calib["t_velo_to_cam"] @ pts_h

    if mode.startswith("p_rect"):
        proj = calib["p_rect"] @ np.vstack([cam[:3, :], np.ones((1, cam.shape[1]))])
        zs = proj[2, :]
        valid = zs > 1e-3
        us = np.zeros_like(zs)
        vs = np.zeros_like(zs)
        us[valid] = proj[0, valid] / zs[valid]
        vs[valid] = proj[1, valid] / zs[valid]
        return np.stack([us, vs, valid], axis=1)

    xyz = cam[:3, :].T
    if mode.endswith("rrect"):
        xyz = (calib["r_rect"] @ xyz.T).T
    zs = xyz[:, 2]
    valid = zs > 1e-3
    us = np.zeros_like(zs)
    vs = np.zeros_like(zs)
    k = calib["k"]
    us[valid] = (k[0, 0] * xyz[valid, 0] / zs[valid]) + k[0, 2]
    vs[valid] = (k[1, 1] * xyz[valid, 1] / zs[valid]) + k[1, 2]
    return np.stack([us, vs, valid], axis=1)


def _polygon_from_points(points: List[Tuple[float, float]]) -> Optional[Polygon]:
    if not points:
        return None
    try:
        poly = Polygon(points)
    except Exception:
        return None
    if poly.is_empty or not poly.is_valid:
        return None
    return poly


def _bbox_from_poly(poly: Polygon) -> Optional[List[float]]:
    if poly is None or poly.is_empty:
        return None
    minx, miny, maxx, maxy = poly.bounds
    return [float(minx), float(miny), float(maxx), float(maxy)]


def _bbox_iou(b1: List[float], b2: List[float]) -> float:
    if not b1 or not b2 or len(b1) != 4 or len(b2) != 4:
        return 0.0
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _draw_points(img: Image.Image, pts: np.ndarray, color: Tuple[int, int, int], radius: int = 1) -> None:
    draw = ImageDraw.Draw(img)
    for u, v, valid in pts:
        if not valid:
            continue
        x = int(round(float(u)))
        y = int(round(float(v)))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=color)


def _draw_poly(img: Image.Image, poly: Optional[Polygon], color: Tuple[int, int, int]) -> None:
    if poly is None or poly.is_empty:
        return
    draw = ImageDraw.Draw(img)
    if poly.geom_type == "MultiPolygon":
        polys = list(poly.geoms)
    else:
        polys = [poly]
    for item in polys:
        coords = list(item.exterior.coords)
        draw.line(coords + [coords[0]], fill=color, width=2)


def _roundtrip_metrics(
    raw_poly: Optional[Polygon],
    raw_bbox: Optional[List[float]],
    reproj_poly: Optional[Polygon],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if raw_poly is not None and reproj_poly is not None and not raw_poly.is_empty and not reproj_poly.is_empty:
        inter = raw_poly.intersection(reproj_poly).area
        union = raw_poly.union(reproj_poly).area
        iou = float(inter / union) if union > 0 else 0.0
        area_ratio = float(reproj_poly.area / raw_poly.area) if raw_poly.area > 0 else None
    else:
        iou = None
        area_ratio = None
        if raw_bbox and reproj_poly is not None:
            reproj_bbox = _bbox_from_poly(reproj_poly)
            iou = _bbox_iou(raw_bbox, reproj_bbox or [])
            if reproj_bbox:
                area_raw = max(0.0, (raw_bbox[2] - raw_bbox[0]) * (raw_bbox[3] - raw_bbox[1]))
                if area_raw > 0:
                    area_ratio = (
                        (reproj_bbox[2] - reproj_bbox[0]) * (reproj_bbox[3] - reproj_bbox[1])
                    ) / area_raw
    center_err = None
    if raw_bbox and reproj_poly is not None:
        reproj_bbox = _bbox_from_poly(reproj_poly)
        if reproj_bbox:
            cx1 = (raw_bbox[0] + raw_bbox[2]) / 2.0
            cy1 = (raw_bbox[1] + raw_bbox[3]) / 2.0
            cx2 = (reproj_bbox[0] + reproj_bbox[2]) / 2.0
            cy2 = (reproj_bbox[1] + reproj_bbox[3]) / 2.0
            center_err = float(math.hypot(cx1 - cx2, cy1 - cy2))
    return iou, center_err, area_ratio


def _mask_bounds_ok(raw_bbox: Optional[List[float]], width: int, height: int) -> bool:
    if not raw_bbox:
        return False
    minx, miny, maxx, maxy = raw_bbox
    if minx < -1 or miny < -1:
        return False
    if maxx > width + 1 or maxy > height + 1:
        return False
    return True


def _proj_in_image_ratio(pts: np.ndarray, width: int, height: int) -> float:
    if pts.size == 0:
        return 0.0
    us = pts[:, 0]
    vs = pts[:, 1]
    valid = pts[:, 2] > 0
    in_image = valid & (us >= 0) & (us < width) & (vs >= 0) & (vs < height)
    return float(np.mean(in_image)) if pts.shape[0] > 0 else 0.0


def _points_in_mask(pts: np.ndarray, raw_poly: Optional[Polygon]) -> int:
    if raw_poly is None or raw_poly.is_empty or pts.size == 0:
        return 0
    mask_pts = []
    for u, v, valid in pts:
        if not valid:
            continue
        mask_pts.append((float(u), float(v)))
    if not mask_pts:
        return 0
    count = 0
    for pt in mask_pts:
        if raw_poly.contains(Point(pt)):
            count += 1
    return count


def _draw_outputs(
    image_path: Path,
    lidar_proj: np.ndarray,
    raw_poly: Optional[Polygon],
    reproj_poly: Optional[Polygon],
    out_dir: Path,
    frame_id: str,
) -> None:
    base = Image.open(image_path).convert("RGB")
    lidar_img = base.copy()
    _draw_points(lidar_img, lidar_proj, (0, 255, 0), radius=1)
    lidar_img.save(out_dir / f"{frame_id}_lidar_on_image.png")

    mask_img = base.copy()
    _draw_points(mask_img, lidar_proj, (0, 255, 0), radius=1)
    _draw_poly(mask_img, raw_poly, (255, 165, 0))
    mask_img.save(out_dir / f"{frame_id}_mask_and_lidar.png")

    reproj_img = base.copy()
    _draw_poly(reproj_img, raw_poly, (255, 165, 0))
    _draw_poly(reproj_img, reproj_poly, (255, 0, 0))
    reproj_img.save(out_dir / f"{frame_id}_reproj_vs_mask.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--drive", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--camera", default="image_00")
    ap.add_argument("--candidate-source", default="")
    ap.add_argument("--image-run", default="")
    ap.add_argument("--image-provider", default="")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    kitti_root = Path(args.kitti_root)
    drive_id = str(args.drive)
    frame_id = _normalize_frame_id(str(args.frame))
    camera = str(args.camera)
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / f"proj_debug_{frame_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_dir = _find_image_dir(kitti_root, drive_id, camera)
    image_path = image_dir / f"{frame_id}.png"
    if not image_path.exists():
        return 2
    image = Image.open(image_path)
    width, height = image.size

    cfg = _load_yaml(Path("configs/crosswalk_fix_range.yaml"))
    image_run = Path(args.image_run or cfg.get("image_run") or "")
    image_provider = str(args.image_provider or cfg.get("image_provider") or "")
    if not image_run.exists():
        return 2
    feature_store_root = image_run / f"feature_store_{image_provider}"
    raw_gdf, raw_score, raw_status = _load_crosswalk_raw(feature_store_root, drive_id, frame_id, {})
    raw_poly = unary_union(raw_gdf.geometry.values) if not raw_gdf.empty else None
    raw_bbox = _bbox_from_poly(raw_poly) if raw_poly is not None else None

    candidate_path = _resolve_candidate_source(args.candidate_source)
    world_geom = _load_candidate_geom(candidate_path, drive_id, frame_id) if candidate_path else None

    pose = load_kitti360_pose(kitti_root, drive_id, frame_id)
    calib = load_kitti360_calib(kitti_root, camera)
    calib_01 = load_kitti360_calib(kitti_root, "image_01")

    lidar_world = load_kitti360_lidar_points_world(kitti_root, drive_id, frame_id)
    z_vals = lidar_world[:, 2] if lidar_world.size > 0 else np.array([], dtype=float)
    ground_z = float(np.percentile(z_vals, 10)) if z_vals.size > 0 else 0.0
    lidar_proj = _project_world_to_image(lidar_world, pose, calib, "k_rrect")
    proj_ratio = _proj_in_image_ratio(lidar_proj, width, height)
    points_in_bbox = 0
    if raw_bbox:
        minx, miny, maxx, maxy = raw_bbox
        in_bbox = (
            (lidar_proj[:, 0] >= minx)
            & (lidar_proj[:, 0] <= maxx)
            & (lidar_proj[:, 1] >= miny)
            & (lidar_proj[:, 1] <= maxy)
            & (lidar_proj[:, 2] > 0)
        )
        points_in_bbox = int(np.count_nonzero(in_bbox))

    modes = [
        ("p_rect_cam00", calib, "p_rect"),
        ("k_rrect_cam00", calib, "k_rrect"),
        ("k_no_rect_cam00", calib, "k_no_rrect"),
        ("p_rect_cam01", calib_01, "p_rect"),
        ("k_rrect_cam01", calib_01, "k_rrect"),
    ]
    variant_rows = []
    reproj_poly_best = None
    best_iou = -1.0
    best_center = None
    best_ratio = None
    for name, c, mode in modes:
        reproj_poly = None
        if world_geom is not None:
            coords = np.array(list(world_geom.exterior.coords), dtype=float)
            pts = np.column_stack([coords[:, 0], coords[:, 1], np.full(coords.shape[0], ground_z)])
            proj = _project_world_to_image(pts, pose, c, mode)
            points = [(float(u), float(v)) for u, v, valid in proj if valid]
            reproj_poly = _polygon_from_points(points)
        iou, center_err, area_ratio = _roundtrip_metrics(raw_poly, raw_bbox, reproj_poly)
        variant_rows.append(
            {
                "mode": name,
                "reproj_iou": iou,
                "center_err_px": center_err,
                "area_ratio": area_ratio,
            }
        )
        if iou is not None and iou > best_iou:
            best_iou = iou
            reproj_poly_best = reproj_poly
            best_center = center_err
            best_ratio = area_ratio

    offset_rows = []
    for offset in range(-5, 6):
        frame_num = _parse_frame_id(frame_id) or int(frame_id)
        target = _normalize_frame_id(str(frame_num + offset))
        try:
            pose_off = load_kitti360_pose(kitti_root, drive_id, target)
            lidar_world_off = load_kitti360_lidar_points_world(kitti_root, drive_id, target)
        except Exception:
            continue
        proj_off = _project_world_to_image(lidar_world_off, pose_off, calib, "k_rrect")
        in_bbox = 0
        if raw_bbox:
            minx, miny, maxx, maxy = raw_bbox
            mask = (
                (proj_off[:, 0] >= minx)
                & (proj_off[:, 0] <= maxx)
                & (proj_off[:, 1] >= miny)
                & (proj_off[:, 1] <= maxy)
                & (proj_off[:, 2] > 0)
            )
            in_bbox = int(np.count_nonzero(mask))
        offset_iou = None
        if world_geom is not None:
            coords = np.array(list(world_geom.exterior.coords), dtype=float)
            pts = np.column_stack([coords[:, 0], coords[:, 1], np.zeros(coords.shape[0], dtype=float)])
            proj = _project_world_to_image(pts, pose_off, calib, "k_rrect")
            points = [(float(u), float(v)) for u, v, valid in proj if valid]
            reproj_poly = _polygon_from_points(points)
            offset_iou, _center, _ratio = _roundtrip_metrics(raw_poly, raw_bbox, reproj_poly)
        offset_rows.append(
            {
                "offset": offset,
                "points_in_bbox": in_bbox,
                "reproj_iou": offset_iou,
            }
        )

    debug_json = {
        "drive_id": drive_id,
        "frame_id": frame_id,
        "image_path": str(image_path),
        "image_shape": [width, height],
        "raw_status": raw_status,
        "raw_score": raw_score,
        "roi_bbox": None,
        "scale_x": None,
        "scale_y": None,
        "offset_x": None,
        "offset_y": None,
        "mask_bbox": raw_bbox,
        "mask_in_image_bounds": _mask_bounds_ok(raw_bbox, width, height),
        "proj_in_image_ratio": proj_ratio,
        "points_in_bbox": points_in_bbox,
        "points_in_mask": _points_in_mask(lidar_proj, raw_poly),
        "ground_z": ground_z,
        "reproj_iou": best_iou if best_iou >= 0 else None,
        "center_err_px": best_center,
        "area_ratio": best_ratio,
        "calib": {
            "p_rect": calib["p_rect"].tolist(),
            "k": calib["k"].tolist(),
            "r_rect": calib["r_rect"].tolist(),
            "t_velo_to_cam": calib["t_velo_to_cam"].tolist(),
        },
    }
    debug_path = out_dir / f"proj_debug_{frame_id}.json"
    debug_path.write_text(json.dumps(debug_json, indent=2), encoding="utf-8")

    pd.DataFrame(variant_rows).to_csv(out_dir / f"proj_debug_{frame_id}_variants.csv", index=False)
    pd.DataFrame(offset_rows).to_csv(out_dir / f"proj_debug_{frame_id}_offsets.csv", index=False)

    _draw_outputs(image_path, lidar_proj, raw_poly, reproj_poly_best, out_dir, frame_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

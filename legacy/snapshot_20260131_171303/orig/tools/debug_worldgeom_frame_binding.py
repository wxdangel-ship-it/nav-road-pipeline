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
from shapely.geometry import Polygon

from pipeline.datasets.kitti360_io import load_kitti360_calib, load_kitti360_pose, load_kitti360_pose_full
from tools.build_image_sample_index import _find_image_dir
from tools.run_crosswalk_monitor_range import _normalize_frame_id, _project_world_to_image


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _list_layers(path: Path) -> List[str]:
    try:
        import pyogrio

        return [name for name, _ in pyogrio.list_layers(path)]
    except Exception:
        return []


def _resolve_outputs_dir(run_dir: Path) -> Path:
    if (run_dir / "outputs").exists():
        return run_dir / "outputs"
    return run_dir


def _find_trace_path(outputs_dir: Path) -> Optional[Path]:
    for cand in [outputs_dir / "crosswalk_trace.csv", outputs_dir.parent / "crosswalk_trace.csv"]:
        if cand.exists():
            return cand
    return None


def _find_cluster_summary(outputs_dir: Path) -> Optional[Path]:
    cand = outputs_dir / "cluster_summary.csv"
    return cand if cand.exists() else None


def _find_candidate_path(outputs_dir: Path) -> Optional[Path]:
    for cand in [
        outputs_dir / "frame_candidates_utm32.gpkg",
        outputs_dir / "crosswalk_entities_utm32.gpkg",
    ]:
        if cand.exists():
            return cand
    return None


def _load_candidates(candidate_path: Path, drive_id: str, frame_id: str) -> gpd.GeoDataFrame:
    layers = _list_layers(candidate_path)
    layer = None
    for name in ("frame_candidates", "crosswalk_candidate_poly"):
        if name in layers:
            layer = name
            break
    if layer is None and layers:
        layer = layers[0]
    if layer is None:
        return gpd.GeoDataFrame()
    gdf = gpd.read_file(candidate_path, layer=layer)
    if gdf.empty:
        return gdf
    if "drive_id" in gdf.columns:
        gdf = gdf[gdf["drive_id"].astype(str) == drive_id]
    if "frame_id" in gdf.columns:
        gdf = gdf[gdf["frame_id"].astype(str) == frame_id]
    return gdf


def _pick_primary_candidate(candidates: gpd.GeoDataFrame) -> Optional[pd.Series]:
    if candidates.empty:
        return None
    def _safe_int(value: object) -> int:
        try:
            if value is None:
                return 0
            if isinstance(value, float) and math.isnan(value):
                return 0
            return int(value)
        except Exception:
            return 0
    for _, row in candidates.iterrows():
        if str(row.get("proj_method") or "") == "lidar" and _safe_int(row.get("geom_ok")) == 1:
            return row
    for _, row in candidates.iterrows():
        if _safe_int(row.get("geom_ok")) == 1:
            return row
    return candidates.iloc[0]


def _parse_bbox(value: object) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [float(v) for v in parsed]
        except Exception:
            return None
    return None


def _find_seed_for_cluster(outputs_dir: Path, cluster_id: str) -> Optional[dict]:
    seeds_path = outputs_dir / "seeds.jsonl"
    if not seeds_path.exists():
        return None
    for line in seeds_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if str(obj.get("cluster_id") or "") == cluster_id:
            return obj
    return None


def _load_pose(kitti_root: Path, drive_id: str, frame_id: str, lidar_world_mode: str) -> Optional[Tuple[float, ...]]:
    try:
        if str(lidar_world_mode).lower() == "fullpose":
            x, y, z, roll, pitch, yaw = load_kitti360_pose_full(kitti_root, drive_id, frame_id)
            return (x, y, z, roll, pitch, yaw)
        x, y, yaw = load_kitti360_pose(kitti_root, drive_id, frame_id)
        return (x, y, yaw)
    except Exception:
        return None


def _world_to_ego_matrix(pose: Tuple[float, ...]) -> np.ndarray:
    if len(pose) == 6:
        x, y, z, roll, pitch, yaw = pose
        c1 = float(np.cos(yaw))
        s1 = float(np.sin(yaw))
        c2 = float(np.cos(pitch))
        s2 = float(np.sin(pitch))
        c3 = float(np.cos(roll))
        s3 = float(np.sin(roll))
        r_z = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        r_y = np.array([[c2, 0.0, s2], [0.0, 1.0, 0.0], [-s2, 0.0, c2]], dtype=float)
        r_x = np.array([[1.0, 0.0, 0.0], [0.0, c3, -s3], [0.0, s3, c3]], dtype=float)
        r_world_pose = r_z @ r_y @ r_x
        r_world_ego = r_world_pose.T
        t = -r_world_ego @ np.array([x, y, z], dtype=float)
    else:
        x, y, yaw = pose
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        r_world_ego = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        t = -r_world_ego @ np.array([x, y, 0.0], dtype=float)
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = r_world_ego
    mat[:3, 3] = t
    return mat


def _centroid_ego(pose: Tuple[float, ...], centroid_world: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if len(pose) == 6:
        x, y, z, _roll, _pitch, yaw = pose
        dx = centroid_world[0] - x
        dy = centroid_world[1] - y
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        x_ego = c * dx + s * dy
        y_ego = -s * dx + c * dy
        return float(x_ego), float(y_ego), float(centroid_world[2] - z)
    x, y, yaw = pose
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    dx = centroid_world[0] - x
    dy = centroid_world[1] - y
    x_ego = c * dx + s * dy
    y_ego = -s * dx + c * dy
    return float(x_ego), float(y_ego), float(centroid_world[2])


def _draw_utm_plot(
    pose_xy: Tuple[float, float],
    centroid_xy: Tuple[float, float],
    out_path: Path,
) -> None:
    x_vals = [pose_xy[0], centroid_xy[0]]
    y_vals = [pose_xy[1], centroid_xy[1]]
    minx, maxx = min(x_vals), max(x_vals)
    miny, maxy = min(y_vals), max(y_vals)
    pad = max(10.0, 0.2 * max(maxx - minx, maxy - miny, 1.0))
    minx -= pad
    maxx += pad
    miny -= pad
    maxy += pad
    size = 800
    img = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    def to_px(x: float, y: float) -> Tuple[int, int]:
        if maxx == minx or maxy == miny:
            return size // 2, size // 2
        px = int((x - minx) / (maxx - minx) * (size - 1))
        py = int((maxy - y) / (maxy - miny) * (size - 1))
        return px, py

    pose_px = to_px(pose_xy[0], pose_xy[1])
    cent_px = to_px(centroid_xy[0], centroid_xy[1])
    draw.line([pose_px, cent_px], fill=(0, 0, 0), width=2)
    draw.ellipse((pose_px[0] - 5, pose_px[1] - 5, pose_px[0] + 5, pose_px[1] + 5), fill=(0, 128, 255))
    draw.ellipse((cent_px[0] - 5, cent_px[1] - 5, cent_px[0] + 5, cent_px[1] + 5), fill=(255, 0, 0))
    dx = centroid_xy[0] - pose_xy[0]
    dy = centroid_xy[1] - pose_xy[1]
    dist = math.hypot(dx, dy)
    label = f"dx={dx:.2f} dy={dy:.2f} dist={dist:.2f}"
    draw.text((10, 10), label, fill=(0, 0, 0))
    img.save(out_path)


def _draw_candidate_on_image(
    image_path: Path,
    bbox: Optional[List[float]],
    mask_path: Optional[Path],
    out_path: Path,
) -> Optional[str]:
    if not image_path.exists():
        return "image_missing"
    base = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(base)
    if bbox and len(bbox) == 4:
        draw.rectangle(bbox, outline=(255, 0, 0, 255), width=3)
    if mask_path and mask_path.exists():
        mask = Image.open(mask_path).convert("L")
        overlay = Image.new("RGBA", base.size, (255, 0, 0, 80))
        base = Image.composite(overlay, base, mask)
        draw = ImageDraw.Draw(base)
    base.convert("RGB").save(out_path)
    return None


def _reproj_worldgeom(
    image_path: Path,
    geom: Optional[Polygon],
    pose: Optional[Tuple[float, ...]],
    calib: Optional[Dict[str, np.ndarray]],
    out_path: Path,
) -> Optional[str]:
    if not image_path.exists():
        return "image_missing"
    base = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(base)
    if geom is None or geom.is_empty:
        draw.text((10, 10), "reproj_fail: world_geom_missing", fill=(255, 0, 0))
        base.save(out_path)
        return "world_geom_missing"
    if pose is None or calib is None:
        draw.text((10, 10), "reproj_fail: pose_or_calib_missing", fill=(255, 0, 0))
        base.save(out_path)
        return "pose_or_calib_missing"
    coords = np.array(list(geom.exterior.coords), dtype=float)
    pts = np.column_stack([coords[:, 0], coords[:, 1], np.zeros(coords.shape[0], dtype=float)])
    proj = _project_world_to_image(pts, pose, calib)
    poly_pts = [(float(u), float(v)) for u, v, valid in proj if valid]
    if len(poly_pts) < 3:
        draw.text((10, 10), "reproj_fail: no_valid_points", fill=(255, 0, 0))
        base.save(out_path)
        return "no_valid_points"
    draw.line(poly_pts + [poly_pts[0]], fill=(255, 0, 0), width=3)
    base.save(out_path)
    return None


def _find_kitti_root(run_dir: Path) -> Optional[Path]:
    candidates = []
    debug_dir = run_dir / "debug"
    if debug_dir.exists():
        candidates.extend(sorted(debug_dir.glob("*.yaml")))
    candidates.extend(sorted(run_dir.glob("*.yaml")))
    for path in candidates:
        cfg = _load_yaml(path)
        val = cfg.get("kitti_root")
        if val:
            return Path(str(val))
        if isinstance(cfg.get("run_overrides"), dict):
            val = cfg["run_overrides"].get("kitti_root")
            if val:
                return Path(str(val))
    cfg = _load_yaml(Path("configs/crosswalk_fix_range.yaml"))
    val = cfg.get("kitti_root")
    return Path(str(val)) if val else None


def _image_path_for_frame(kitti_root: Path, drive_id: str, frame_id: str, camera: str) -> Path:
    image_dir = _find_image_dir(kitti_root, drive_id, camera)
    return image_dir / f"{frame_id}.png"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--drive", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--camera", default="image_00")
    ap.add_argument("--lidar-world-mode", default="fullpose")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    outputs_dir = _resolve_outputs_dir(run_dir)
    drive_id = str(args.drive)
    frame_id = _normalize_frame_id(str(args.frame))
    camera = str(args.camera)
    lidar_world_mode = str(args.lidar_world_mode).lower()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_path = _find_trace_path(outputs_dir)
    if trace_path is None:
        raise SystemExit("ERROR: crosswalk_trace.csv not found")
    trace_df = pd.read_csv(trace_path)
    trace_df["frame_id_norm"] = trace_df["frame_id"].astype(str).apply(_normalize_frame_id)
    trace_row = trace_df[
        (trace_df["frame_id_norm"] == frame_id) & (trace_df["drive_id"].astype(str) == drive_id)
    ]
    trace_info = trace_row.iloc[0].to_dict() if not trace_row.empty else {}

    candidate_path = _find_candidate_path(outputs_dir)
    candidates = _load_candidates(candidate_path, drive_id, frame_id) if candidate_path else gpd.GeoDataFrame()
    primary = _pick_primary_candidate(candidates)
    primary_info = primary.to_dict() if primary is not None else {}

    cluster_id = str(primary_info.get("cluster_id") or trace_info.get("cluster_id") or "")
    seed_info = _find_seed_for_cluster(outputs_dir, cluster_id) if cluster_id else None

    source_frame_id = str(primary_info.get("frame_id") or frame_id)
    source_stage = "stage1"
    proj_method = str(primary_info.get("proj_method") or trace_info.get("proj_method") or "")
    reject_reasons = str(primary_info.get("reject_reasons") or "")
    if "stage2" in str(primary_info.get("candidate_id") or "") or proj_method == "stage2_video" or "stage2_video" in reject_reasons:
        source_stage = "stage2_video"

    kitti_root = _find_kitti_root(run_dir)
    if kitti_root is None or not kitti_root.exists():
        raise SystemExit("ERROR: kitti_root not found")

    image_path = Path(str(trace_info.get("image_path") or ""))
    if not image_path.exists():
        image_path = _image_path_for_frame(kitti_root, drive_id, frame_id, camera)
    source_image_path = image_path
    if source_frame_id != frame_id:
        source_image_path = _image_path_for_frame(kitti_root, drive_id, source_frame_id, camera)

    bbox = _parse_bbox(primary_info.get("bbox_px"))
    mask_path = None
    if cluster_id:
        stage2_mask = outputs_dir / "stage2_masks" / cluster_id / f"{frame_id}.png"
        if stage2_mask.exists():
            mask_path = stage2_mask

    geom = primary_info.get("geometry") if primary is not None else None
    centroid = geom.centroid if geom is not None and not geom.is_empty else None
    centroid_world = (float(centroid.x), float(centroid.y), 0.0) if centroid is not None else None

    pose = _load_pose(kitti_root, drive_id, frame_id, lidar_world_mode)
    pose_world = None
    if pose is not None:
        pose_world = [float(v) for v in pose]
    dx = dy = dist = None
    if centroid_world is not None and pose is not None:
        dx = centroid_world[0] - pose[0]
        dy = centroid_world[1] - pose[1]
        dist = float(math.hypot(dx, dy))
    t_world_to_ego = _world_to_ego_matrix(pose).tolist() if pose is not None else None
    centroid_ego = _centroid_ego(pose, centroid_world) if pose is not None and centroid_world else None

    verdict = "OK"
    if centroid_ego is not None and centroid_ego[0] < 0:
        verdict = "FRAME_MISMATCH_OR_WRONG_TRANSFORM"

    calib = load_kitti360_calib(kitti_root, camera) if kitti_root is not None else None

    utm_out = out_dir / f"{frame_id}_worldgeom_on_utm32.png"
    if centroid_world is not None and pose is not None:
        _draw_utm_plot((pose[0], pose[1]), (centroid_world[0], centroid_world[1]), utm_out)

    candidate_img_out = out_dir / f"{frame_id}_candidate_on_image.png"
    candidate_draw_err = _draw_candidate_on_image(image_path, bbox, mask_path, candidate_img_out)

    reproj_out = out_dir / f"{frame_id}_reproj_worldgeom.png"
    reproj_err = _reproj_worldgeom(image_path, geom, pose, calib, reproj_out)

    debug_json = {
        "drive_id": drive_id,
        "frame_id": frame_id,
        "source_frame_id": source_frame_id,
        "source_stage": source_stage,
        "source_image_path": str(source_image_path),
        "source_mask_path": str(mask_path) if mask_path else "",
        "source_bbox_px": bbox,
        "candidate_id": str(primary_info.get("candidate_id") or ""),
        "cluster_id": cluster_id,
        "seed_info": seed_info or {},
        "pose_world_utm32": pose_world,
        "candidate_centroid_world_utm32": list(centroid_world) if centroid_world else None,
        "delta_world": {"dx": dx, "dy": dy, "dist": dist},
        "t_world_to_ego": t_world_to_ego,
        "centroid_ego": list(centroid_ego) if centroid_ego else None,
        "verdict": verdict,
        "visuals": {
            "utm32": str(utm_out) if utm_out.exists() else "",
            "candidate_on_image": str(candidate_img_out) if candidate_img_out.exists() else "",
            "reproj_worldgeom": str(reproj_out) if reproj_out.exists() else "",
        },
        "errors": {
            "candidate_on_image": candidate_draw_err or "",
            "reproj_worldgeom": reproj_err or "",
        },
    }
    json_path = out_dir / f"worldgeom_debug_{frame_id}.json"
    json_path.write_text(json.dumps(debug_json, indent=2), encoding="utf-8")

    print(json.dumps(debug_json, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple


def _list_stems(p: Path, pattern: str) -> Set[str]:
    if not p.exists():
        return set()
    return {x.stem for x in p.glob(pattern) if x.is_file()}


def _prefer_pose_root(data_root: Path) -> Optional[Path]:
    # 优先顺序：extract > oxts > poses
    for name in ["data_poses_oxts_extract", "data_poses_oxts", "data_poses"]:
        cand = data_root / name
        if cand.exists():
            return cand
    return None


def discover_drives(data_root: Path) -> List[str]:
    d3 = data_root / "data_3d_raw"
    if not d3.exists():
        return []
    drives = []
    for d in d3.iterdir():
        if not d.is_dir():
            continue
        if not d.name.endswith("_sync"):
            continue
        lidar_dir = d / "velodyne_points" / "data"
        if lidar_dir.exists():
            drives.append(d.name)
    drives.sort()
    return drives


def infer_priors(data_root: Path, prior_root: Optional[Path] = None) -> Dict[str, Optional[Path]]:
    # 默认先验就放在 data_root 内（你现在就是这种结构）
    pr = prior_root or data_root

    osm_layers = pr / "_osm_download" / "aoi_kitti360_sample" / "_layers"
    sat_tiles = pr / "_lglbw_dop20" / "tiles_utm32"

    return {
        "osm_layers": osm_layers if osm_layers.exists() else None,
        "sat_tiles": sat_tiles if sat_tiles.exists() else None,
    }


def index_drive(data_root: Path, drive: str, max_frames: Optional[int] = None) -> Dict[str, Any]:
    # lidar
    lidar_dir = data_root / "data_3d_raw" / drive / "velodyne_points" / "data"
    lidar_ids = _list_stems(lidar_dir, "*.bin")

    # images (cam 00/01)
    img00_dir = data_root / "data_2d_raw" / drive / "image_00" / "data_rect"
    img01_dir = data_root / "data_2d_raw" / drive / "image_01" / "data_rect"
    img00_ids = _list_stems(img00_dir, "*.png")
    img01_ids = _list_stems(img01_dir, "*.png")

    # pose (oxts per-frame txt, if exists)
    pose_root = _prefer_pose_root(data_root)
    pose_dir = None
    pose_ids: Set[str] = set()
    if pose_root is not None:
        cand = pose_root / drive / "oxts" / "data"
        if cand.exists():
            pose_dir = cand
            pose_ids = _list_stems(pose_dir, "*.txt")

    # Optional speed: limit to earliest N frames
    if max_frames is not None and max_frames > 0 and len(lidar_ids) > max_frames:
        lidar_sorted = sorted(lidar_ids)
        lidar_ids = set(lidar_sorted[:max_frames])

    # intersections
    img00_match = len(lidar_ids & img00_ids)
    img01_match = len(lidar_ids & img01_ids)
    img_any_match = len(lidar_ids & (img00_ids | img01_ids))
    pose_match = len(lidar_ids & pose_ids)

    lidar_cnt = len(lidar_ids)
    def _ratio(x: int, y: int) -> float:
        return float(x) / float(y) if y > 0 else 0.0

    out = {
        "tile_id": drive,
        "lidar_dir": str(lidar_dir),
        "img00_dir": str(img00_dir),
        "img01_dir": str(img01_dir),
        "pose_dir": str(pose_dir) if pose_dir else None,
        "lidar_count": lidar_cnt,
        "img00_match": img00_match,
        "img01_match": img01_match,
        "img_any_match": img_any_match,
        "pose_match": pose_match,
        "image_coverage": _ratio(img_any_match, lidar_cnt),
        "pose_coverage": _ratio(pose_match, lidar_cnt),
        "has_img00": img00_dir.exists(),
        "has_img01": img01_dir.exists(),
        "has_pose": pose_dir is not None,
    }
    return out


def summarize_dataset(data_root: Path, drives: Optional[List[str]] = None, max_frames: Optional[int] = None) -> Dict[str, Any]:
    drives2 = drives or discover_drives(data_root)
    tiles = []
    for d in drives2:
        tiles.append(index_drive(data_root, d, max_frames=max_frames))

    # aggregate
    total_lidar = sum(t["lidar_count"] for t in tiles)
    total_img_any = sum(t["img_any_match"] for t in tiles)
    total_pose = sum(t["pose_match"] for t in tiles)

    image_cov = (total_img_any / total_lidar) if total_lidar > 0 else 0.0
    pose_cov = (total_pose / total_lidar) if total_lidar > 0 else 0.0

    missing_pose_drives = [t["tile_id"] for t in tiles if not t["has_pose"]]
    return {
        "drive_count": len(tiles),
        "drives": drives2,
        "total_lidar": total_lidar,
        "total_img_any": total_img_any,
        "total_pose": total_pose,
        "image_coverage": round(image_cov, 4),
        "pose_coverage": round(pose_cov, 4),
        "missing_pose_drives": missing_pose_drives,
        "tiles": tiles,  # 可用于排查（量不大时保留）
    }

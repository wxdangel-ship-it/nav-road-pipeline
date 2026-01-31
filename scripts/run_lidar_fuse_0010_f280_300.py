from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.datasets.kitti360_io import (
    _find_velodyne_dir,
    _resolve_velodyne_path,
    load_kitti360_lidar_points,
)
from pipeline.calib.kitti360_world import transform_points_V_to_W
from pipeline.lidar_semantic.accum_points_world import _voxel_downsample
from pipeline.lidar_semantic.export_pointcloud import write_las
from scripts.pipeline_common import (
    ensure_dir,
    ensure_overwrite,
    now_ts,
    relpath,
    setup_logging,
    write_json,
    write_text,
)


LOG = logging.getLogger("lidar_fuse_0010_f280_300")

REQUIRED_KEYS = [
    "FRAME_START",
    "FRAME_END",
    "STRIDE",
    "TARGET_EPSG",
    "OUTPUT_FORMAT",
    "VOXEL_SIZE_M",
    "RANGE_FILTER",
    "OVERWRITE",
]


def _load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    import yaml

    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


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


def _dump_resolved(run_dir: Path, cfg: Dict[str, object]) -> None:
    import yaml

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8"
    )


def _write_resolved(run_dir: Path, cfg: Dict[str, object]) -> str:
    _dump_resolved(run_dir, cfg)
    params_hash = _hash_cfg(cfg)
    (run_dir / "params_hash.txt").write_text(params_hash + "\n", encoding="utf-8")
    return params_hash


def _resolve_config(base: Dict[str, object], run_dir: Path) -> Tuple[Dict[str, object], str]:
    cfg = dict(base)
    defaults = {
        "FRAME_START": 280,
        "FRAME_END": 300,
        "STRIDE": 1,
        "TARGET_EPSG": 32632,
        "OUTPUT_FORMAT": "laz",
        "VOXEL_SIZE_M": 0.0,
        "RANGE_FILTER": [0, 0],
        "OVERWRITE": True,
    }
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys: {missing}")
    params_hash = _write_resolved(run_dir, cfg)
    return cfg, params_hash


def _auto_find_kitti_root(cfg: Dict[str, object], scans: List[str]) -> Optional[Path]:
    env_root = str(cfg.get("KITTI_ROOT") or "").strip()
    if env_root:
        scans.append(env_root)
        p = Path(env_root)
        if p.exists():
            return p
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


def _frame_ids(start: int, end: int, stride: int) -> List[str]:
    return [f"{i:010d}" for i in range(int(start), int(end) + 1, max(1, int(stride)))]


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
            "p99": 0.0,
            "nonzero_ratio": 0.0,
        }
    vals = inten.astype(np.float64)
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "p99": float(np.percentile(vals, 99)),
        "nonzero_ratio": float(np.mean(vals > 0.0)),
    }


def _bbox(points_xyz: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    if points_xyz.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    return (float(mins[0]), float(mins[1]), float(mins[2]), float(maxs[0]), float(maxs[1]), float(maxs[2]))


def _bbox_ok(bounds: Tuple[float, float, float, float, float, float]) -> Tuple[bool, str]:
    minx, miny, minz, maxx, maxy, maxz = bounds
    if all(abs(v) < 1e-9 for v in bounds):
        return False, "bbox_all_zero"
    if any(np.isnan(v) for v in bounds):
        return False, "bbox_has_nan"
    if minx >= maxx or miny >= maxy:
        return False, "bbox_invalid_order"
    if minx < 100000.0 or maxx > 900000.0:
        return False, "bbox_easting_out_of_range"
    if miny < 0.0 or maxy > 10000000.0:
        return False, "bbox_northing_out_of_range"
    if minz == maxz == 0.0:
        return False, "bbox_z_all_zero"
    return True, "ok"


def main() -> int:
    base_cfg = _load_yaml(Path("configs/lidar_fuse_0010_f280_300.yaml"))
    run_id = now_ts()
    run_dir = Path("runs") / f"lidar_fuse_0010_f280_300_{run_id}"
    if bool(base_cfg.get("OVERWRITE", True)):
        ensure_overwrite(run_dir)
    ensure_dir(run_dir)
    setup_logging(run_dir / "run.log")
    LOG.info("run_start")

    cfg, params_hash = _resolve_config(base_cfg, run_dir)
    scans: List[str] = []
    data_root = _auto_find_kitti_root(cfg, scans)
    if data_root is None:
        write_text(run_dir / "report.md", "data_root_missing")
        write_json(run_dir / "stats.json", {"status": "FAIL", "reason": "data_root_missing", "scan_paths": scans})
        return 2

    drive_id = _select_drive_0010(data_root)
    if drive_id != "2013_05_28_drive_0010_sync":
        write_text(run_dir / "report.md", f"drive_id_mismatch:{drive_id}")
        write_json(run_dir / "stats.json", {"status": "FAIL", "reason": "drive_id_mismatch", "drive_id": drive_id})
        return 2

    frame_start = int(cfg["FRAME_START"])
    frame_end = int(cfg["FRAME_END"])
    stride = int(cfg["STRIDE"])
    frame_ids = _frame_ids(frame_start, frame_end, stride)

    velodyne_dir = _find_velodyne_dir(data_root, drive_id)
    missing = []
    for fid in frame_ids:
        if _resolve_velodyne_path(velodyne_dir, fid) is None:
            missing.append(fid)
    if missing:
        write_text(run_dir / "report.md", f"missing_frames:{missing}")
        write_json(run_dir / "stats.json", {"status": "FAIL", "reason": "missing_frames", "missing": missing})
        return 2

    voxel_size = float(cfg["VOXEL_SIZE_M"])
    range_min = float(cfg.get("RANGE_FILTER", [0, 0])[0])
    range_max = float(cfg.get("RANGE_FILTER", [0, 0])[1])
    use_range = range_max > 0 and range_max > range_min

    points_list: List[np.ndarray] = []
    intensity_list: List[np.ndarray] = []
    intensity_missing = False
    intensity_rule = "unknown"
    frame_errors: List[str] = []

    for idx_f, fid in enumerate(frame_ids):
        try:
            raw = load_kitti360_lidar_points(data_root, drive_id, fid)
            world = transform_points_V_to_W(raw[:, :3], data_root, drive_id, fid, cam_id="image_00")
            if raw.size == 0 or world.size == 0:
                continue
            intensity_raw = _intensity_field(raw)
            missing_this_frame = False
            if intensity_raw is None:
                missing_this_frame = True
                intensity_raw = np.zeros((raw.shape[0],), dtype=np.float32)
            else:
                intensity_raw = intensity_raw.astype(np.float32)

            if use_range:
                xyz_velo = raw[:, :3].astype(np.float32)
                dist = np.linalg.norm(xyz_velo, axis=1)
                mask = (dist >= range_min) & (dist <= range_max)
                if not np.any(mask):
                    continue
                world = world[mask]
                intensity_raw = intensity_raw[mask]

            if missing_this_frame:
                mapped = np.zeros((intensity_raw.shape[0],), dtype=np.uint16)
                rule = "missing_zero"
                missing_flag = True
            else:
                mapped, rule, missing_flag = _map_intensity(intensity_raw)
            if intensity_rule == "unknown":
                intensity_rule = rule
            elif intensity_rule != rule:
                intensity_rule = "mixed"
            if missing_flag:
                intensity_missing = True

            points_list.append(world.astype(np.float32))
            intensity_list.append(mapped.astype(np.uint16))
        except Exception as exc:
            frame_errors.append(f"{fid}:{exc}")
        if (idx_f + 1) % 5 == 0:
            LOG.info("frame_progress: %s/%s", idx_f + 1, len(frame_ids))

    if not points_list:
        write_text(run_dir / "report.md", "no_points_accumulated")
        write_json(run_dir / "stats.json", {"status": "FAIL", "reason": "no_points_accumulated", "errors": frame_errors[:5]})
        return 2

    points_xyz = np.vstack(points_list)
    intensity = np.concatenate(intensity_list)
    if voxel_size > 0.0:
        points_xyz, intensity = _voxel_downsample(points_xyz, intensity, voxel_size)

    bounds = _bbox(points_xyz)
    bbox_ok, bbox_reason = _bbox_ok(bounds)
    intensity_stats = _intensity_stats(intensity)

    status = "PASS" if bbox_ok else "FAIL"
    intensity_all_zero = intensity_stats["nonzero_ratio"] == 0.0
    stats = {
        "status": status,
        "drive_id": drive_id,
        "frames": [frame_start, frame_end],
        "stride": stride,
        "total_points": int(points_xyz.shape[0]),
        "bbox": {
            "xmin": bounds[0],
            "ymin": bounds[1],
            "zmin": bounds[2],
            "xmax": bounds[3],
            "ymax": bounds[4],
            "zmax": bounds[5],
        },
        "bbox_check": {"ok": bool(bbox_ok), "reason": bbox_reason},
        "intensity": {
            "mapping_rule": intensity_rule,
            "missing": bool(intensity_missing),
            "all_zero": bool(intensity_all_zero),
            "stats": intensity_stats,
        },
        "range_filter_m": [range_min, range_max],
        "voxel_size_m": voxel_size,
        "params_hash": params_hash,
    }
    write_json(run_dir / "stats.json", stats)

    if not bbox_ok:
        write_text(run_dir / "report.md", f"bbox_invalid:{bbox_reason}")
        return 2

    out_dir = ensure_dir(run_dir / "fused")
    out_path = out_dir / "fused_points_utm32.laz"
    write_las(
        out_path,
        points_xyz.astype(np.float32),
        intensity.astype(np.uint16),
        np.ones((points_xyz.shape[0],), dtype=np.uint8),
        int(cfg["TARGET_EPSG"]),
    )

    report = [
        "# LiDAR fuse 0010 f280-300",
        "",
        f"- drive_id: {drive_id}",
        f"- frames: {frame_start}-{frame_end}",
        f"- stride: {stride}",
        f"- data_root: {data_root}",
        f"- total_points: {int(points_xyz.shape[0])}",
        f"- bbox: xmin={bounds[0]:.3f}, xmax={bounds[3]:.3f}, ymin={bounds[1]:.3f}, ymax={bounds[4]:.3f}, zmin={bounds[2]:.3f}, zmax={bounds[5]:.3f}",
        f"- intensity_map_rule: {intensity_rule}",
        f"- intensity_missing: {bool(intensity_missing)}",
        f"- intensity_min: {intensity_stats['min']:.1f}",
        f"- intensity_max: {intensity_stats['max']:.1f}",
        f"- intensity_p50: {intensity_stats['p50']:.1f}",
        f"- intensity_p90: {intensity_stats['p90']:.1f}",
        f"- intensity_p99: {intensity_stats['p99']:.1f}",
        f"- intensity_nonzero_ratio: {intensity_stats['nonzero_ratio']:.4f}",
        "",
        "## Outputs",
        f"- {relpath(run_dir, out_path)}",
        f"- {relpath(run_dir, out_path.with_suffix('.las'))}",
    ]
    if intensity_missing or intensity_all_zero:
        report.append("")
        report.append("## Notes")
        if intensity_missing:
            report.append("- intensity_missing: true (filled with zeros)")
        if intensity_all_zero:
            report.append("- intensity_all_zero: true")
    if frame_errors:
        report.append("")
        report.append("## Frame Errors")
        report.extend([f"- {e}" for e in frame_errors[:10]])
    write_text(run_dir / "report.md", "\n".join(report))
    LOG.info("run_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

from pipeline.datasets.kitti360_io import (
    _find_velodyne_dir,
    load_kitti360_lidar_points,
    load_kitti360_lidar_points_world_full,
    load_kitti360_pose,
)
from pipeline.lidar_semantic.export_pointcloud import write_las
from scripts.pipeline_common import (
    LOG,
    ensure_dir,
    ensure_overwrite,
    relpath,
    setup_logging,
    validate_output_crs,
    write_csv,
    write_json,
    write_text,
    write_gpkg_layer,
)


def _detect_kitti_root() -> Path:
    env = Path(str(Path("E:/KITTI360/KITTI-360")))
    if env.exists():
        return env
    cand = Path(str(Path("D:/KITTI360/KITTI-360")))
    if cand.exists():
        return cand
    env2 = Path(str(Path("C:/KITTI360/KITTI-360")))
    if env2.exists():
        return env2
    from os import environ

    env_root = environ.get("POC_DATA_ROOT", "").strip()
    if env_root:
        p = Path(env_root)
        if p.exists():
            return p
    return env


def _load_drives(data_root: Path) -> List[str]:
    drives_file = Path("configs/golden_drives.txt")
    if drives_file.exists():
        drives = [ln.strip() for ln in drives_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if drives:
            return drives
    raw_root = data_root / "data_3d_raw"
    if raw_root.exists():
        return sorted([p.name for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("2013_05_28_drive_")])
    return []


def _select_drive_0010(drives: List[str]) -> str:
    for d in drives:
        if "_0010_" in d:
            return d
    raise RuntimeError("no_0010_drive_found")


def _latest_v2_run() -> Optional[Path]:
    runs = sorted(Path("runs").glob("lidar_semantic_v2_0010_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _read_marking_score_center(run_dir: Path) -> Optional[Tuple[float, float]]:
    score_path = next(run_dir.rglob("marking_score_utm32.tif"), None)
    if score_path is None:
        return None
    try:
        with rasterio.open(score_path) as ds:
            arr = ds.read(1)
            mask = np.isfinite(arr)
            if not np.any(mask):
                return None
            vals = arr[mask]
            if vals.size == 0:
                return None
            thr = np.quantile(vals, 0.999)
            hot = (arr >= thr) & mask
            if not np.any(hot):
                return None
            ys, xs = np.where(hot)
            xs = xs.astype(np.float64)
            ys = ys.astype(np.float64)
            if xs.size == 0:
                return None
            xs = xs + 0.5
            ys = ys + 0.5
            coords = rasterio.transform.xy(ds.transform, ys, xs)
            x = float(np.mean(coords[0]))
            y = float(np.mean(coords[1]))
            return x, y
    except Exception:
        return None


def _trajectory_midpoint(data_root: Path, drive_id: str, frame_ids: List[str]) -> Tuple[float, float]:
    mid = frame_ids[len(frame_ids) // 2]
    x, y, _ = load_kitti360_pose(data_root, drive_id, mid)
    return float(x), float(y)


def _intensity_map(raw: np.ndarray) -> Tuple[np.ndarray, str]:
    if raw.size == 0:
        return raw.astype(np.uint16), "empty"
    if raw.dtype.kind == "f":
        max_val = float(np.nanmax(raw))
        if max_val <= 1.5:
            return np.round(np.clip(raw, 0.0, 1.0) * 65535.0).astype(np.uint16), "float01_to_uint16"
        if max_val <= 255.0:
            return np.round(np.clip(raw, 0.0, 255.0) * 256.0).astype(np.uint16), "float255_to_uint16"
    if raw.dtype.kind in {"i", "u"}:
        max_val = float(np.max(raw))
        if max_val <= 255.0:
            return (raw.astype(np.uint16) * 256).astype(np.uint16), "uint8_to_uint16"
    return np.clip(raw, 0, 65535).astype(np.uint16), "uint16_direct"


def _stats(inten: np.ndarray) -> Dict[str, float]:
    if inten.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "nonzero_ratio": 0.0}
    vals = inten.astype(np.float64)
    nz = float(np.mean(vals > 0.0))
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "nonzero_ratio": nz,
    }


def _hist(inten: np.ndarray) -> np.ndarray:
    if inten.size == 0:
        return np.zeros((256,), dtype=np.int64)
    hist, _ = np.histogram(inten.astype(np.float64), bins=256, range=(0, 65535))
    return hist.astype(np.int64)


def _read_las_intensity(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    with path.open("rb") as f:
        header = f.read(227)
        if len(header) < 227:
            return None
        offset_to_points = struct.unpack_from("<I", header, 96)[0]
        point_format = struct.unpack_from("<B", header, 104)[0]
        record_len = struct.unpack_from("<H", header, 105)[0]
        point_count = struct.unpack_from("<I", header, 107)[0]
        if point_format != 0 or record_len < 20:
            return None
        f.seek(offset_to_points)
        raw = f.read(point_count * record_len)
        if len(raw) < point_count * record_len:
            return None
        intens = np.ndarray(
            shape=(point_count,),
            dtype="<u2",
            buffer=raw,
            offset=12,
            strides=(record_len,),
        )
        return intens.copy()


def _read_las_points(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        return None
    with path.open("rb") as f:
        header = f.read(227)
        if len(header) < 227:
            return None
        fmt = "<4sHH16sBB32s32sHHH I I B H I 5I ddd ddd dddddd"
        unpacked = struct.unpack(fmt, header)
        offset_to_points = unpacked[12]
        point_format = unpacked[14]
        record_len = unpacked[15]
        point_count = unpacked[16]
        sx, sy, sz = unpacked[22], unpacked[23], unpacked[24]
        ox, oy, oz = unpacked[25], unpacked[26], unpacked[27]
        if point_format != 0 or record_len < 20:
            return None
        f.seek(offset_to_points)
        raw = f.read(point_count * record_len)
        if len(raw) < point_count * record_len:
            return None
        xi = np.ndarray(shape=(point_count,), dtype="<i4", buffer=raw, offset=0, strides=(record_len,))
        yi = np.ndarray(shape=(point_count,), dtype="<i4", buffer=raw, offset=4, strides=(record_len,))
        zi = np.ndarray(shape=(point_count,), dtype="<i4", buffer=raw, offset=8, strides=(record_len,))
        intens = np.ndarray(shape=(point_count,), dtype="<u2", buffer=raw, offset=12, strides=(record_len,))
        x = xi.astype(np.float64) * float(sx) + float(ox)
        y = yi.astype(np.float64) * float(sy) + float(oy)
        z = zi.astype(np.float64) * float(sz) + float(oz)
        pts = np.stack([x, y, z], axis=1).astype(np.float32)
        return pts, intens.copy()


def main() -> int:
    from datetime import datetime

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"lidar_intensity_debug_0010_f250_300_{run_id}"
    ensure_overwrite(run_dir)
    ensure_dir(run_dir)
    setup_logging(run_dir / "run.log")

    data_root = _detect_kitti_root()
    drives = _load_drives(data_root)
    drive_id = _select_drive_0010(drives)
    frame_ids = [str(i) for i in range(250, 301)]

    report_lines = [
        "# LiDAR intensity debug 0010 f250-300",
        "",
        f"- drive_id: {drive_id}",
        f"- frames: 250-300",
        f"- data_root: {data_root}",
    ]

    try:
        velodyne_dir = _find_velodyne_dir(data_root, drive_id)
        report_lines.append(f"- velodyne_dir: {velodyne_dir}")
    except Exception as exc:
        report_lines.append(f"- velodyne_dir: missing ({exc})")
        write_text(run_dir / "report.md", "\n".join(report_lines))
        return 2

    available = []
    for path in velodyne_dir.glob("*.bin"):
        try:
            available.append(int(path.stem))
        except Exception:
            continue
    available_set = set(available)
    frame_ids_int = [i for i in range(250, 301) if i in available_set]
    frame_ids = [f"{i:010d}" for i in frame_ids_int]
    if not frame_ids:
        report_lines.append("- status: fail (no_frames_in_range)")
        write_text(run_dir / "report.md", "\n".join(report_lines))
        return 2

    # Fail-fast on current pipeline output
    latest_run = _latest_v2_run()
    pipeline_intensity_stats = {}
    pipeline_clip_path = None
    if latest_run:
        candidate = next(latest_run.rglob("road_surface_points_utm32.laz"), None)
        if candidate:
            inten = _read_las_intensity(candidate)
            if inten is not None:
                pipeline_intensity_stats = _stats(inten)
                flag = "missing" if pipeline_intensity_stats.get("max", 0.0) == 0.0 or pipeline_intensity_stats.get("nonzero_ratio", 0.0) < 0.001 else "ok"
                report_lines.append(f"- pipeline_intensity_check: {flag} ({relpath(latest_run, candidate)})")

    pts_list: List[np.ndarray] = []
    inten_list: List[np.ndarray] = []
    missing_frames: List[str] = []
    error_samples: List[str] = []
    for frame_id in frame_ids:
        try:
            raw = load_kitti360_lidar_points(data_root, drive_id, frame_id)
            world = load_kitti360_lidar_points_world_full(data_root, drive_id, frame_id, cam_id="image_00")
            if raw.size == 0 or world.size == 0:
                missing_frames.append(frame_id)
                continue
            pts_list.append(world.astype(np.float32))
            inten_list.append(raw[:, 3].astype(np.float32))
        except Exception as exc:
            if len(error_samples) < 3:
                error_samples.append(f"{frame_id}:{exc}")
            missing_frames.append(frame_id)

    if not pts_list:
        report_lines.append(f"- missing_frames: {len(missing_frames)}")
        if error_samples:
            report_lines.append(f"- error_samples: {', '.join(error_samples)}")
        report_lines.append("- status: fail (no_points)")
        write_text(run_dir / "report.md", "\n".join(report_lines))
        return 2

    points = np.vstack(pts_list)
    intensity_raw = np.concatenate(inten_list)

    center = None
    if latest_run:
        center = _read_marking_score_center(latest_run)
    if center is None:
        center = _trajectory_midpoint(data_root, drive_id, frame_ids)

    cx, cy = center
    half = 50.0
    clip_mask = (
        (points[:, 0] >= cx - half)
        & (points[:, 0] <= cx + half)
        & (points[:, 1] >= cy - half)
        & (points[:, 1] <= cy + half)
    )
    if not np.any(clip_mask):
        cx, cy = float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))
        clip_mask = (
            (points[:, 0] >= cx - half)
            & (points[:, 0] <= cx + half)
            & (points[:, 1] >= cy - half)
            & (points[:, 1] <= cy + half)
        )
    if not np.any(clip_mask):
        half = 100.0
        clip_mask = (
            (points[:, 0] >= cx - half)
            & (points[:, 0] <= cx + half)
            & (points[:, 1] >= cy - half)
            & (points[:, 1] <= cy + half)
        )
    points_clip = points[clip_mask]
    intensity_clip = intensity_raw[clip_mask]

    roi_poly = box(cx - half, cy - half, cx + half, cy + half)
    roi_gdf = gpd.GeoDataFrame([{"drive_id": drive_id, "geometry": roi_poly}], geometry="geometry", crs="EPSG:32632")
    roi_path = run_dir / "roi_bbox_utm32.gpkg"
    write_gpkg_layer(roi_path, "roi", roi_gdf, 32632, [], overwrite=True)

    intensity_mapped, mapping_rule = _intensity_map(intensity_clip)
    stats = _stats(intensity_mapped)
    hist = _hist(intensity_mapped)

    clip_dir = ensure_dir(run_dir / "clip")
    raw_las_path = clip_dir / "raw_clip_with_intensity_utm32.las"
    if points_clip.size:
        write_las(raw_las_path, points_clip, intensity_mapped, np.ones((points_clip.shape[0],), dtype=np.uint8), 32632)

    if latest_run and candidate and candidate.exists():
        pip = _read_las_points(candidate)
        if pip is not None:
            p_pts, p_int = pip
            pmask = (
                (p_pts[:, 0] >= cx - half)
                & (p_pts[:, 0] <= cx + half)
                & (p_pts[:, 1] >= cy - half)
                & (p_pts[:, 1] <= cy + half)
            )
            if np.any(pmask):
                pipeline_clip_path = clip_dir / "pipeline_clip_current_utm32.las"
                write_las(
                    pipeline_clip_path,
                    p_pts[pmask],
                    p_int[pmask],
                    np.ones((int(pmask.sum()),), dtype=np.uint8),
                    32632,
                )

    stats_payload = {
        "mapping_rule": mapping_rule,
        "stats": stats,
        "raw_intensity_min": float(np.min(intensity_raw)),
        "raw_intensity_max": float(np.max(intensity_raw)),
        "raw_intensity_mean": float(np.mean(intensity_raw)),
        "raw_intensity_dtype": str(intensity_raw.dtype),
        "missing_frames": missing_frames,
        "error_samples": error_samples,
    }
    write_json(run_dir / "intensity_stats.json", stats_payload)
    hist_rows = [{"bin": i, "count": int(c)} for i, c in enumerate(hist)]
    write_csv(run_dir / "intensity_hist.csv", hist_rows, ["bin", "count"])

    report_lines.extend(
        [
            "",
            "## Raw intensity field",
            f"- dtype: {intensity_raw.dtype}",
            f"- min: {float(np.min(intensity_raw)):.6f}",
            f"- max: {float(np.max(intensity_raw)):.6f}",
            f"- mean: {float(np.mean(intensity_raw)):.6f}",
            f"- mapping_rule: {mapping_rule}",
            "",
            "## Clip stats (mapped uint16)",
            f"- min: {stats['min']:.2f}",
            f"- max: {stats['max']:.2f}",
            f"- mean: {stats['mean']:.2f}",
            f"- p50: {stats['p50']:.2f}",
            f"- p90: {stats['p90']:.2f}",
            f"- p95: {stats['p95']:.2f}",
            f"- p99: {stats['p99']:.2f}",
            f"- nonzero_ratio: {stats['nonzero_ratio']:.4f}",
            "",
            "## Outputs",
            f"- {relpath(run_dir, roi_path)}",
            f"- {relpath(run_dir, raw_las_path)}",
            f"- {relpath(run_dir, pipeline_clip_path)}" if pipeline_clip_path else "- pipeline_clip_current: none",
            f"- {relpath(run_dir, run_dir / 'intensity_stats.json')}",
            f"- {relpath(run_dir, run_dir / 'intensity_hist.csv')}",
        ]
    )

    if stats["max"] <= 0.0 or stats["nonzero_ratio"] < 0.05:
        report_lines.append("- conclusion: raw intensity missing or mapping failed")
    else:
        report_lines.append("- conclusion: raw intensity present in frames 250-300")
        if pipeline_intensity_stats:
            if pipeline_intensity_stats.get("max", 0.0) == 0.0 or pipeline_intensity_stats.get("nonzero_ratio", 0.0) < 0.001:
                report_lines.append("- root_cause: pipeline export likely wrote float intensity without scaling (mapped to 0)")

    write_text(run_dir / "report.md", "\n".join(report_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

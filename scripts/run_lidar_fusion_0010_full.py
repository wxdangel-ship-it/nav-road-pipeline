from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import laspy
import numpy as np

from pipeline.lidar_fusion.fuse_frames import collect_input_fingerprints, fuse_frames_to_las
from scripts.pipeline_common import (
    ensure_dir,
    ensure_overwrite,
    now_ts,
    relpath,
    setup_logging,
    write_csv,
    write_json,
    write_text,
)

LOG = logging.getLogger("lidar_fusion_0010_full")

# =========================
# 参数区（按需修改）
# =========================
KITTI_ROOT = r"E:\KITTI360\KITTI-360"
DRIVE_ID = "2013_05_28_drive_0010_sync"
FRAME_START = 0
FRAME_END = 3835
STRIDE = 1
OUT_DIR = Path("runs") / "lidar_fusion_0010_full"
OVERWRITE = False
OUTPUT_MODE = "utm32"  # world | utm32 | auto
OUTPUT_FORMAT = "laz"
REQUIRE_LAZ = True
REQUIRE_CAM0_TO_WORLD = True
ALLOW_POSES_FALLBACK = False
USE_R_RECT_WITH_CAM0_TO_WORLD = True
TRANSFORM_JSON = r"runs\world_to_utm32_fit_0010_full_20260131_230902\report\world_to_utm32_report.json"

# 分块控制
ENABLE_CHUNKING = True
TARGET_LAZ_MB_PER_PART = 1200
MAX_PARTS = 8

# 可选：从 StepA 报告读取 frame 范围
DRIVE_SCAN_REPORT = r"runs\drive_scan_0010_20260131_212315\report\drive_frames_report.json"


def _load_frame_range() -> Tuple[int, int]:
    if DRIVE_SCAN_REPORT:
        report = json.loads(Path(DRIVE_SCAN_REPORT).read_text(encoding="utf-8"))
        return int(report["min_frame"]), int(report["max_frame"])
    return int(FRAME_START), int(FRAME_END)


def _parse_part_range(name: str) -> Tuple[str, str]:
    stem = Path(name).stem
    parts = stem.split("_part_")
    if len(parts) != 2:
        return "", ""
    rng = parts[1]
    if "_" not in rng:
        return "", ""
    start, end = rng.split("_", 1)
    return start, end


def _write_bbox_geojson(path: Path, bbox: Tuple[float, float, float, float, float, float]) -> None:
    xmin, ymin, _zmin, xmax, ymax, _zmax = bbox
    coords = [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
        [xmin, ymin],
    ]
    geo = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "bbox_utm32"},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(geo, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_head(path: Path, n_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()


def _banding_check(path: Path, max_points: int = 2_000_000) -> Dict[str, object]:
    with laspy.open(path) as reader:
        total = reader.header.point_count
        if total <= 0:
            return {"ok": False, "reason": "empty", "min_nonzero_dy": None, "unique_y_1mm": 0, "sample_n": 0}
        if total <= max_points:
            points = reader.read()
            y = np.asarray(points.y, dtype=np.float64)
        else:
            step = max(1, total // max_points)
            ys: List[float] = []
            count = 0
            for chunk in reader.chunk_iterator(1_000_000):
                y = np.asarray(chunk.y, dtype=np.float64)
                for j in range(0, y.size, step):
                    ys.append(float(y[j]))
                    count += 1
                    if count >= max_points:
                        break
                if count >= max_points:
                    break
            y = np.asarray(ys, dtype=np.float64)
    if y.size == 0:
        return {"ok": False, "reason": "empty", "min_nonzero_dy": None, "unique_y_1mm": 0, "sample_n": 0}
    y_round = np.round(y, 3)
    y_unique = np.unique(y_round)
    dy = np.diff(np.sort(y_unique))
    nonzero = dy[dy > 0]
    min_nonzero = float(np.min(nonzero)) if nonzero.size else None
    return {
        "ok": bool(min_nonzero is not None and min_nonzero <= 0.01),
        "min_nonzero_dy": min_nonzero,
        "unique_y_1mm": int(y_unique.size),
        "sample_n": int(y.size),
    }


def _select_parts(paths: List[str]) -> List[str]:
    if not paths:
        return []
    if len(paths) <= 2:
        return paths
    mid = len(paths) // 2
    return [paths[0], paths[mid], paths[-1]]


def main() -> int:
    run_id = now_ts()
    run_dir = Path(f"{OUT_DIR}_{run_id}")
    if OVERWRITE:
        ensure_overwrite(run_dir)
    ensure_dir(run_dir)
    setup_logging(run_dir / "logs" / "run.log")
    LOG.info("run_start")

    data_root = Path(KITTI_ROOT)
    if not data_root.exists():
        write_text(run_dir / "report" / "report.md", f"data_root_missing:{data_root}")
        write_json(run_dir / "report" / "metrics.json", {"status": "FAIL", "reason": "data_root_missing"})
        return 2

    frame_start, frame_end = _load_frame_range()
    coord_mode = str(OUTPUT_MODE or "auto").lower()
    out_dir = ensure_dir(run_dir / "outputs")
    out_path = out_dir / f"fused_points_{coord_mode}.laz"
    transform = json.loads(Path(TRANSFORM_JSON).read_text(encoding="utf-8")) if TRANSFORM_JSON else None

    result = fuse_frames_to_las(
        data_root=data_root,
        drive_id=DRIVE_ID,
        frame_start=frame_start,
        frame_end=frame_end,
        stride=STRIDE,
        out_path=out_path,
        coord=coord_mode,
        cam_id="image_00",
        overwrite=OVERWRITE,
        output_format=OUTPUT_FORMAT,
        require_laz=REQUIRE_LAZ,
        require_cam0_to_world=REQUIRE_CAM0_TO_WORLD,
        allow_poses_fallback=ALLOW_POSES_FALLBACK,
        use_r_rect_with_cam0_to_world=USE_R_RECT_WITH_CAM0_TO_WORLD,
        world_to_utm32_transform=transform,
        enable_chunking=ENABLE_CHUNKING,
        target_laz_mb_per_part=TARGET_LAZ_MB_PER_PART,
        max_parts=MAX_PARTS,
    )

    frame_ids = list(range(int(frame_start), int(frame_end) + 1, max(1, int(STRIDE))))
    missing_csv = run_dir / "outputs" / "missing_frames.csv"
    write_csv(missing_csv, result.missing_frames, ["frame_id", "reason"])
    write_json(run_dir / "outputs" / "missing_summary.json", result.missing_summary)
    _write_bbox_geojson(run_dir / "outputs" / "bbox_utm32.geojson", result.bbox)

    input_fingerprints = collect_input_fingerprints(data_root, DRIVE_ID, [f"{i:010d}" for i in frame_ids])
    output_list = []
    for p in result.output_paths:
        try:
            size = Path(p).stat().st_size
        except Exception:
            size = 0
        start_f, end_f = _parse_part_range(p)
        output_list.append(
            {
                "path": p,
                "size": size,
                "size_mb": size / (1024 * 1024) if size else 0.0,
                "frame_start": start_f,
                "frame_end": end_f,
                "sha256_head": _sha256_head(Path(p)) if size else "",
            }
        )

    if ENABLE_CHUNKING and output_list:
        write_json(run_dir / "outputs" / "fused_points_utm32_index.json", {"parts": output_list})

    intensity = result.intensity_stats
    intensity_error = intensity.get("max", 0.0) <= 0.0
    if intensity_error:
        result.errors.append("intensity_all_zero")

    meta = {
        "status": "PASS" if not result.errors else "FAIL",
        "drive_id": DRIVE_ID,
        "frames": [frame_start, frame_end],
        "stride": STRIDE,
        "frames_found_velodyne": result.frames_found_velodyne,
        "frames_processed": result.frames_processed,
        "points_read_total": result.points_read_total,
        "points_written_total": result.points_written_total,
        "coord": result.coord,
        "epsg": result.epsg,
        "pose_source": result.pose_source,
        "use_r_rect_with_cam0_to_world": result.use_r_rect_with_cam0_to_world,
        "total_frames": len(frame_ids),
        "success_frames": len(frame_ids) - len(result.missing_frames),
        "missing_frames": len(result.missing_frames),
        "missing_summary": result.missing_summary,
        "total_points": result.total_points,
        "written_points": result.written_points,
        "bbox": {
            "xmin": result.bbox[0],
            "ymin": result.bbox[1],
            "zmin": result.bbox[2],
            "xmax": result.bbox[3],
            "ymax": result.bbox[4],
            "zmax": result.bbox[5],
        },
        "bbox_check": result.bbox_check,
        "intensity": {
            "rule": result.intensity_rule,
            "min": intensity.get("min", 0.0),
            "mean": intensity.get("mean", 0.0),
            "max": intensity.get("max", 0.0),
            "nonzero_ratio": intensity.get("nonzero_ratio", 0.0),
            "error_all_zero": bool(intensity_error),
        },
        "output": {
            "path": str(result.output_path) if result.output_path else "",
            "format": result.output_format,
            "paths": result.output_paths,
        },
        "transform": transform or {},
        "gate_status": (transform or {}).get("gate_status", ""),
        "input_files": input_fingerprints,
        "warnings": result.warnings,
        "errors": result.errors,
    }
    write_json(run_dir / "outputs" / f"fused_points_{result.coord}.meta.json", meta)
    write_json(run_dir / "report" / "metrics.json", meta)
    if result.per_frame_points_sample:
        write_csv(
            run_dir / "report" / "per_frame_points_sample.csv",
            result.per_frame_points_sample,
            ["frame", "points_in", "points_written"],
        )

    banding_samples = _select_parts(result.output_paths or [])
    banding_report = {"parts_checked": [], "pass": True}
    for part in banding_samples:
        info = _banding_check(Path(part))
        entry = {"path": part, **info}
        banding_report["parts_checked"].append(entry)
        if not info.get("ok"):
            banding_report["pass"] = False
    write_json(run_dir / "report" / "banding_audit_full.json", banding_report)
    summary_line = "PASS" if banding_report["pass"] else "FAIL"
    write_text(run_dir / "report" / "banding_summary_full.md", f"banding_check_full: {summary_line}")
    if not banding_report["pass"]:
        raise RuntimeError("banding_check_full_failed")

    report_lines = [
        "# LiDAR fusion full (0010)",
        "",
        f"- run_dir: {run_dir}",
        f"- drive_id: {DRIVE_ID}",
        f"- frames: {frame_start}-{frame_end} (stride={STRIDE})",
        f"- coord: {result.coord} (epsg={result.epsg})",
        f"- pose_source: {result.pose_source}",
        f"- use_r_rect_with_cam0_to_world: {result.use_r_rect_with_cam0_to_world}",
        f"- output_paths: {len(result.output_paths)}",
        f"- total_points: {result.total_points}",
        f"- written_points: {result.written_points}",
        f"- bbox_check: {result.bbox_check.get('ok')} ({result.bbox_check.get('reason')})",
        f"- intensity_max: {intensity.get('max', 0.0):.1f}",
        f"- intensity_nonzero_ratio: {intensity.get('nonzero_ratio', 0.0):.4f}",
    ]
    if intensity_error:
        report_lines.append("- ERROR: intensity_max == 0")
    if result.warnings:
        report_lines.extend(["", "## Warnings"] + [f"- {w}" for w in result.warnings])
    if result.errors:
        report_lines.extend(["", "## Errors"] + [f"- {e}" for e in result.errors])
    write_text(run_dir / "report" / "report.md", "\n".join(report_lines))

    LOG.info("run_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

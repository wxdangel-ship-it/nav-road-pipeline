from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

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

LOG = logging.getLogger("lidar_fusion_0010_f000_300")

# =========================
# 参数区（按需修改）
# =========================
KITTI_ROOT = r"E:\KITTI360\KITTI-360"
DRIVE_ID = "2013_05_28_drive_0010_sync"
FRAME_START = 0
FRAME_END = 300
STRIDE = 1
OUT_DIR = Path("runs") / "lidar_fusion_0010_f000_300"
OVERWRITE = True
OUTPUT_MODE = "utm32"  # world | utm32 | auto
OUTPUT_FORMAT = "laz"
REQUIRE_LAZ = True
REQUIRE_CAM0_TO_WORLD = True
ALLOW_POSES_FALLBACK = False
USE_R_RECT_WITH_CAM0_TO_WORLD = True
WORLD_TO_UTM32_TRANSFORM_JSON = (
    r"runs\world_to_utm32_fit_0010_0_300_20260131_210301\report\world_to_utm32_report.json"
)


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

    coord_mode = str(OUTPUT_MODE or "auto").lower()
    base_coord = "utm32" if coord_mode == "auto" else coord_mode
    suffix = ".laz" if OUTPUT_FORMAT == "laz" else ".las" if OUTPUT_FORMAT == "las" else ".npz"
    out_path = ensure_dir(run_dir / "outputs") / f"fused_points_{base_coord}{suffix}"

    result = fuse_frames_to_las(
        data_root=data_root,
        drive_id=DRIVE_ID,
        frame_start=FRAME_START,
        frame_end=FRAME_END,
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
        world_to_utm32_transform=(
            json.loads(Path(WORLD_TO_UTM32_TRANSFORM_JSON).read_text(encoding="utf-8"))
            if WORLD_TO_UTM32_TRANSFORM_JSON
            else None
        ),
    )

    frame_ids = list(range(int(FRAME_START), int(FRAME_END) + 1, max(1, int(STRIDE))))
    missing_summary = result.missing_summary
    missing_csv = run_dir / "outputs" / "missing_frames.csv"
    write_csv(missing_csv, result.missing_frames, ["frame_id", "reason"])

    bbox = {
        "xmin": result.bbox[0],
        "ymin": result.bbox[1],
        "zmin": result.bbox[2],
        "xmax": result.bbox[3],
        "ymax": result.bbox[4],
        "zmax": result.bbox[5],
    }
    bbox_check = result.bbox_check

    intensity = result.intensity_stats
    intensity_error = intensity.get("max", 0.0) <= 0.0
    if intensity_error:
        result.errors.append("intensity_all_zero")

    utm32_evidence = result.utm32_evidence or {}
    utm32_failed_reason = result.utm32_failed_reason
    input_fingerprints = collect_input_fingerprints(data_root, DRIVE_ID, [f"{i:010d}" for i in frame_ids])

    meta = {
        "status": "PASS" if not result.errors else "FAIL",
        "drive_id": DRIVE_ID,
        "frames": [FRAME_START, FRAME_END],
        "stride": STRIDE,
        "coord": result.coord,
        "epsg": result.epsg,
        "pose_source": result.pose_source,
        "use_r_rect_with_cam0_to_world": result.use_r_rect_with_cam0_to_world,
        "world_to_utm32_transform_source": WORLD_TO_UTM32_TRANSFORM_JSON,
        "total_frames": len(frame_ids),
        "success_frames": len(frame_ids) - len(result.missing_frames),
        "missing_frames": len(result.missing_frames),
        "missing_summary": missing_summary,
        "total_points": result.total_points,
        "written_points": result.written_points,
        "bbox": bbox,
        "bbox_check": bbox_check,
        "utm32_failed_reason": utm32_failed_reason,
        "world_to_utm32_offset_median": utm32_evidence.get("world_to_utm32_offset_median"),
        "utm32_xy_residual_rms_m": utm32_evidence.get("utm32_xy_residual_rms_m"),
        "common_frames_used": utm32_evidence.get("common_frames_used"),
        "utm32_evidence_reason": utm32_evidence.get("reason"),
        "intensity": {
            "rule": result.intensity_rule,
            "min": intensity.get("min", 0.0),
            "mean": intensity.get("mean", 0.0),
            "max": intensity.get("max", 0.0),
            "nonzero_ratio": intensity.get("nonzero_ratio", 0.0),
            "error_all_zero": bool(intensity_error),
        },
        "banding_check": result.banding_check,
        "output": {
            "path": str(result.output_path) if result.output_path else "",
            "format": result.output_format,
        },
        "input_files": input_fingerprints,
        "warnings": result.warnings,
        "errors": result.errors,
        "coord_trace": {
            "velo_to_world": "pipeline.calib.kitti360_world.transform_points_V_to_W",
            "world_to_utm32": "pipeline.calib.kitti360_world.kitti_world_to_utm32" if result.coord == "utm32" else "n/a",
        },
        "transform_note": (
            "cam0_to_world: T_rectcam0_to_world @ T_rect @ T_velo_to_cam0"
            if result.pose_source == "cam0_to_world"
            else "poses_fallback: T_imu_to_world @ T_cam0_to_imu @ T_velo_to_cam0"
        ),
    }
    write_json(run_dir / "outputs" / f"fused_points_{result.coord}.meta.json", meta)

    report_lines = [
        "# LiDAR fusion v0 (0010 f000-300)",
        "",
        f"- run_dir: {run_dir}",
        f"- data_root: {data_root}",
        f"- drive_id: {DRIVE_ID}",
        f"- frames: {FRAME_START}-{FRAME_END} (stride={STRIDE})",
        f"- coord: {result.coord} (epsg={result.epsg})",
        f"- pose_source: {result.pose_source}",
        f"- use_r_rect_with_cam0_to_world: {result.use_r_rect_with_cam0_to_world}",
        f"- output: {relpath(run_dir, result.output_path) if result.output_path else 'n/a'}",
        f"- output_format: {result.output_format}",
        f"- total_points: {result.total_points}",
        f"- written_points: {result.written_points}",
        f"- bbox: xmin={bbox['xmin']:.3f}, xmax={bbox['xmax']:.3f}, ymin={bbox['ymin']:.3f}, ymax={bbox['ymax']:.3f}, zmin={bbox['zmin']:.3f}, zmax={bbox['zmax']:.3f}",
        f"- bbox_check: {bbox_check['ok']} ({bbox_check['reason']})",
        f"- utm32_failed_reason: {utm32_failed_reason}",
        f"- world_to_utm32_offset_median: {utm32_evidence.get('world_to_utm32_offset_median')}",
        f"- utm32_xy_residual_rms_m: {utm32_evidence.get('utm32_xy_residual_rms_m')}",
        f"- common_frames_used: {utm32_evidence.get('common_frames_used')}",
        f"- utm32_evidence_reason: {utm32_evidence.get('reason')}",
        f"- intensity_min: {intensity.get('min', 0.0):.1f}",
        f"- intensity_mean: {intensity.get('mean', 0.0):.1f}",
        f"- intensity_max: {intensity.get('max', 0.0):.1f}",
        f"- intensity_nonzero_ratio: {intensity.get('nonzero_ratio', 0.0):.4f}",
        f"- intensity_rule: {result.intensity_rule}",
        f"- missing_frames: {len(result.missing_frames)} / {len(frame_ids)}",
        "- residual: n/a",
    ]
    if intensity_error:
        report_lines.append("- ERROR: intensity_max == 0")
    if result.warnings:
        report_lines.extend(["", "## Warnings"] + [f"- {w}" for w in result.warnings])
    if result.errors:
        report_lines.extend(["", "## Errors"] + [f"- {e}" for e in result.errors])
    if missing_summary:
        report_lines.extend(["", "## Missing Frames Summary"] + [f"- {k}: {v}" for k, v in missing_summary.items()])
    report_lines.extend(
        [
            "",
            "## Outputs",
            f"- {relpath(run_dir, result.output_path) if result.output_path else 'n/a'}",
            f"- {relpath(run_dir, run_dir / 'outputs' / f'fused_points_{result.coord}.meta.json')}",
            f"- {relpath(run_dir, missing_csv)}",
        ]
    )
    write_text(run_dir / "report" / "report.md", "\n".join(report_lines))
    write_json(run_dir / "report" / "metrics.json", meta)

    LOG.info("run_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

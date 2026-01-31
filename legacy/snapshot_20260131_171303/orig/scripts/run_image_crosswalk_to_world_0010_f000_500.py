from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline._io import load_yaml
from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_json, write_text, write_csv


CFG_DEFAULT = Path("configs/image_crosswalk_to_world_0010_f000_500.yaml")


def _find_latest_run(pattern: str) -> Optional[Path]:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.glob(pattern) if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copytree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if header is None:
                header = parts
                continue
            row = {k: (parts[idx] if idx < len(parts) else "") for idx, k in enumerate(header)}
            rows.append(row)
    return rows


def _count_merged_candidates(gpkg_path: Path) -> int:
    try:
        import geopandas as gpd

        if not gpkg_path.exists():
            return 0
        gdf = gpd.read_file(gpkg_path, layer="crosswalk_candidates")
        return int(len(gdf))
    except Exception:
        return 0


def _qa_roundtrip_p90_max(roundtrip_csv: Path, qa_frames: List[int]) -> Optional[float]:
    rows = _load_csv_rows(roundtrip_csv)
    if not rows:
        return None
    qa_set = {f"{f:010d}" for f in qa_frames}
    vals = []
    for row in rows:
        fid = str(row.get("frame_id", "")).strip()
        if fid in qa_set:
            try:
                vals.append(float(row.get("p90", "nan")))
            except Exception:
                continue
    if not vals:
        return None
    return max(vals)


def _seed_ok(per_frame_csv: Path, seed_frame: int) -> bool:
    rows = _load_csv_rows(per_frame_csv)
    seed_id = f"{seed_frame:010d}"
    for row in rows:
        if str(row.get("frame_id", "")).strip() == seed_id:
            return str(row.get("status", "")).strip() == "ok"
    return False


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(CFG_DEFAULT))
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    drive_id = str(cfg.get("drive_id") or "2013_05_28_drive_0010_sync")
    frame_start = int(cfg.get("frame_start", 0))
    frame_end = int(cfg.get("frame_end", 500))
    image_cam = str(cfg.get("image_cam") or "image_00")
    overwrite = bool(cfg.get("overwrite", True))

    run_id = now_ts()
    run_dir = Path("runs") / f"image_crosswalk_to_world_0010_000_500_{run_id}"
    ensure_overwrite(run_dir if overwrite else run_dir)
    _ensure_dir(run_dir)
    setup_logging(run_dir / "run.log")

    qa_cfg = cfg.get("qa") or {}
    stage1_cfg = cfg.get("stage1") or {}
    stage2_cfg = cfg.get("stage2") or {}
    stage3_cfg = cfg.get("stage3") or {}
    merge_cfg = cfg.get("merge") or {}

    stage12_cfg = {
        "drive_id": drive_id,
        "kitti_root": str(cfg.get("kitti_root") or ""),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "image_cam": image_cam,
        "seed_frame": int(stage2_cfg.get("seed_frame", 290)),
        "qa_random_seed": int(qa_cfg.get("random_seed", 20260130)),
        "qa_sample_n": int(qa_cfg.get("sample_n", 18)),
        "qa_force_include": list(qa_cfg.get("force_include") or [0, 100, 250, 290, 400, 500]),
        "image_model_id": str(cfg.get("image_model_id") or "gdino_sam2_v1"),
        "image_model_zoo": str(cfg.get("image_model_zoo") or "configs/image_model_zoo.yaml"),
        "stage1": stage1_cfg,
        "stage2": {
            "sam2_video_propagate": stage2_cfg.get("sam2_video_propagate", "both"),
            "sam2_video_max_frames": "all",
            "output_mask_format": stage2_cfg.get("output_mask_format", "png"),
            "output_overlay_only_qa": True,
        },
    }

    stage12_cfg_path = run_dir / "stage12_config.yaml"
    stage12_cfg_path.write_text(yaml.safe_dump(stage12_cfg, sort_keys=False), encoding="utf-8")

    py = sys.executable
    cmd_stage12 = [py, "scripts/run_image_stage12_crosswalk_0010_f000_500.py", "--config", str(stage12_cfg_path)]
    subprocess.run(cmd_stage12, check=True)

    stage12_run = _find_latest_run("image_stage12_crosswalk_0010_000_500_*")
    if stage12_run is None:
        stage12_run = _find_latest_run("image_stage12_crosswalk_0010_250_500_*")
    if stage12_run is None:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "stage12_run_not_found"})
        raise SystemExit("stage12 run not found")

    stage12_decision_path = stage12_run / "decision.json"
    if stage12_decision_path.exists():
        stage12_decision = json.loads(stage12_decision_path.read_text(encoding="utf-8"))
        if str(stage12_decision.get("status", "")).upper() == "FAIL":
            write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "stage12_failed", "stage12": stage12_decision})
            raise SystemExit("stage12 failed")

    world_cfg = {
        "drive_id": drive_id,
        "kitti_root": str(cfg.get("kitti_root") or ""),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "image_cam": image_cam,
        "input_stage12_run": str(stage12_run),
        "input_mask_dir": "stage2/masks",
        "qa_frames_path": "qa/qa_frames.json",
        "output_epsg": int(stage3_cfg.get("output_epsg", 32632)),
        "backproject": {
            "use_dtm": bool(stage3_cfg.get("use_dtm", True)),
            "dtm_path": str(stage3_cfg.get("dtm_path", "auto_latest_clean_dtm")),
            "fixed_plane_z0_mode": str(stage3_cfg.get("fixed_plane_z0_mode", "dtm_median")),
            "pixel_use_bottom_crop": float(stage3_cfg.get("pixel_filter_strict_v_min", 0.45)),
            "pixel_use_side_crop": list(stage3_cfg.get("pixel_side_crop") or [0.05, 0.95]),
            "pixel_use_bottom_crop_fallback": float(stage3_cfg.get("pixel_filter_fallback_v_min", 0.30)),
            "pixel_use_side_crop_fallback": list(stage3_cfg.get("pixel_side_crop") or [0.05, 0.95]),
            "contour_sample_step_px": int(stage3_cfg.get("contour_sample_step_px", 2)),
            "min_valid_world_pts": int(stage3_cfg.get("min_valid_world_pts", 60)),
            "dtm_iterations": int(stage3_cfg.get("dtm_iterations", 2)),
            "max_ray_t_m": 200.0,
            "min_area_px": int(stage3_cfg.get("min_area_px_present", 200)),
            "camera_height_m": float(stage3_cfg.get("camera_height_m", 1.65)),
        },
        "simplify": {
            "simplify_tol_m": 0.20,
            "smooth_buffer_m": 0.30,
            "min_poly_area_m2": 6.0,
        },
        "merge": {
            "merge_dist_m": float(merge_cfg.get("merge_dist_m", 2.0)),
            "merge_iou_min": float(merge_cfg.get("merge_iou_min", 0.20)),
            "min_support_frames": int(merge_cfg.get("min_support_frames", 3)),
        },
        "roundtrip": {
            "roundtrip_sample_pts": 200,
            "roundtrip_valid_ratio_min": 0.30,
            "roundtrip_p90_pass_px": 8,
            "roundtrip_p90_warn_px": 15,
        },
    }
    world_cfg_path = run_dir / "world_candidates_config.yaml"
    world_cfg_path.write_text(yaml.safe_dump(world_cfg, sort_keys=False), encoding="utf-8")

    cmd_world = [py, "scripts/run_world_crosswalk_candidates_0010_f250_500.py", "--config", str(world_cfg_path)]
    subprocess.run(cmd_world, check=True)

    world_run = _find_latest_run("world_crosswalk_candidates_0010_250_500_*")
    if world_run is None:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "world_run_not_found"})
        raise SystemExit("world run not found")

    _ensure_dir(run_dir / "qa")
    _ensure_dir(run_dir / "stage1")
    _ensure_dir(run_dir / "stage2")
    _ensure_dir(run_dir / "frames")
    _ensure_dir(run_dir / "merged")
    _ensure_dir(run_dir / "tables")
    _ensure_dir(run_dir / "images")

    _copytree(stage12_run / "qa", run_dir / "qa")
    _copytree(stage12_run / "stage1", run_dir / "stage1")
    _copytree(stage12_run / "stage2", run_dir / "stage2")
    _copytree(stage12_run / "tables" / "per_frame_stage1_counts.csv", run_dir / "tables" / "per_frame_stage1_counts.csv")
    _copytree(stage12_run / "tables" / "per_frame_mask_area.csv", run_dir / "tables" / "per_frame_mask_area.csv")
    _copytree(stage12_run / "tables" / "missing_frames.csv", run_dir / "tables" / "missing_frames.csv")

    _copytree(world_run / "frames", run_dir / "frames")
    _copytree(world_run / "merged", run_dir / "merged")
    _copytree(world_run / "tables", run_dir / "tables")
    _copytree(world_run / "images", run_dir / "images")
    _copytree(world_run / "images" / "qa_montage_roundtrip.png", run_dir / "qa" / "qa_montage_roundtrip.png")

    qa_frames = []
    qa_json_path = run_dir / "qa" / "qa_frames.json"
    if qa_json_path.exists():
        qa_json = json.loads(qa_json_path.read_text(encoding="utf-8"))
        qa_frames = [int(v) for v in qa_json.get("frames") or []]

    per_frame_landing_csv = run_dir / "tables" / "per_frame_landing.csv"
    roundtrip_csv = run_dir / "tables" / "roundtrip_px_errors.csv"
    merged_gpkg = run_dir / "merged" / "crosswalk_candidates_canonical_utm32.gpkg"

    world_decision_path = world_run / "decision.json"
    world_decision = {}
    if world_decision_path.exists():
        world_decision = json.loads(world_decision_path.read_text(encoding="utf-8"))

    n_pixel_present = int(world_decision.get("n_pixel_present", 0))
    n_ok = int(world_decision.get("n_world_ok", 0))
    ok_rate = float(world_decision.get("ok_rate", 0.0)) if n_pixel_present else 0.0
    n_not_crosswalk = int(world_decision.get("frames_not_crosswalk", 0))
    n_backproject_failed = int(world_decision.get("n_backproject_failed", 0))
    merged_count = _count_merged_candidates(merged_gpkg)

    seed_frame = int(stage2_cfg.get("seed_frame", 290))
    seed_ok = _seed_ok(per_frame_landing_csv, seed_frame)
    qa_p90_max = _qa_roundtrip_p90_max(roundtrip_csv, qa_frames)

    status = "PASS"
    if not seed_ok:
        status = "FAIL"
    elif n_pixel_present >= 5 and n_ok == 0:
        status = "FAIL"
    elif qa_p90_max is not None and qa_p90_max > 15:
        status = "FAIL"
    elif ok_rate < 0.3 or merged_count == 0:
        status = "WARN"
    elif qa_p90_max is not None and qa_p90_max > 8:
        status = "WARN"

    decision = {
        "status": status,
        "n_pixel_present": n_pixel_present,
        "n_ok": n_ok,
        "ok_rate": round(ok_rate, 4) if n_pixel_present else 0.0,
        "not_crosswalk": n_not_crosswalk,
        "backproject_failed": n_backproject_failed,
        "merged_candidate_count": merged_count,
        "seed_frame_ok": seed_ok,
        "qa_roundtrip_p90_max": qa_p90_max,
        "stage12_run": str(stage12_run),
        "world_run": str(world_run),
    }
    write_json(run_dir / "decision.json", decision)

    resolved_cfg = dict(cfg)
    resolved_cfg.update(
        {
            "resolved": {
                "run_id": run_id,
                "stage12_run": str(stage12_run),
                "world_run": str(world_run),
            }
        }
    )
    resolved_path = run_dir / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved_cfg, sort_keys=False), encoding="utf-8")
    (run_dir / "params_hash.txt").write_text(_hash_file(resolved_path), encoding="utf-8")

    report_lines = [
        "# Image Crosswalk to World (0010 f0-500)",
        "",
        f"- run_id: {run_id}",
        f"- stage12_run: {stage12_run}",
        f"- world_run: {world_run}",
        f"- status: {status}",
        f"- n_pixel_present: {n_pixel_present}",
        f"- n_ok: {n_ok}",
        f"- ok_rate: {ok_rate:.3f}" if n_pixel_present else "- ok_rate: 0.000",
        f"- merged_candidate_count: {merged_count}",
        "",
        "## outputs",
        "- qa frames: qa/qa_frames.json",
        "- qa montage stage1 before: qa/qa_montage_stage1_before.png",
        "- qa montage stage1 after: qa/qa_montage_stage1_after.png",
        "- qa montage stage2 before: qa/qa_montage_stage2_before.png",
        "- qa montage stage2 after: qa/qa_montage_stage2_after.png",
        "- qa montage roundtrip: qa/qa_montage_roundtrip.png",
        "- stage1: stage1/",
        "- stage2 masks: stage2/masks/frame_*.png",
        "- frames: frames/",
        "- merged canonical: merged/crosswalk_candidates_canonical_utm32.gpkg",
        "- per-frame landing: tables/per_frame_landing.csv",
        "- roundtrip: tables/roundtrip_px_errors.csv",
        "- decision: decision.json",
    ]
    write_text(run_dir / "report.md", "\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

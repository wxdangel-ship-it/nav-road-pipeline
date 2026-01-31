from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline._io import load_yaml
from scripts.pipeline_common import ensure_overwrite, now_ts, setup_logging, write_csv, write_json, write_text


CFG_DEFAULT = Path("configs/image_stage12_stride5_to_world_0010_f000_500.yaml")


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


def _qa_roundtrip_p90_for_frame(roundtrip_csv: Path, frame_id: int) -> Optional[float]:
    rows = _load_csv_rows(roundtrip_csv)
    target = f"{frame_id:010d}"
    for row in rows:
        fid = str(row.get("frame_id", "")).strip()
        if fid == target:
            try:
                return float(row.get("p90", "nan"))
            except Exception:
                return None
    return None


def _seed_ok(per_frame_csv: Path, seed_frame: int) -> bool:
    rows = _load_csv_rows(per_frame_csv)
    seed_id = f"{seed_frame:010d}"
    for row in rows:
        if str(row.get("frame_id", "")).strip() == seed_id:
            return str(row.get("status", "")).strip() == "ok"
    return False


def _top_backproject_reasons(per_frame_csv: Path, topn: int = 3) -> List[Tuple[str, int]]:
    rows = _load_csv_rows(per_frame_csv)
    counter = Counter()
    for row in rows:
        if str(row.get("status", "")).strip() != "backproject_failed":
            continue
        reason = str(row.get("reason", "")).strip() or "unknown"
        counter[reason] += 1
    return counter.most_common(topn)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(CFG_DEFAULT))
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))

    drive_id = str(cfg.get("DRIVE_ID") or "2013_05_28_drive_0010_sync")
    frame_start = int(cfg.get("FRAME_START", 0))
    frame_end = int(cfg.get("FRAME_END", 500))
    image_cam = str(cfg.get("IMAGE_CAM") or "image_00")
    overwrite = bool(cfg.get("OVERWRITE", True))

    if drive_id != "2013_05_28_drive_0010_sync":
        raise SystemExit(f"drive_id must be 2013_05_28_drive_0010_sync, got {drive_id}")
    if frame_start != 0 or frame_end != 500:
        raise SystemExit(f"frame range must be 0-500, got {frame_start}-{frame_end}")

    run_id = now_ts()
    run_dir = Path("runs") / f"image_stage12_stride5_to_world_0010_000_500_{run_id}"
    if overwrite:
        ensure_overwrite(run_dir)
    _ensure_dir(run_dir)
    setup_logging(run_dir / "run.log")

    qa_seed = int(cfg.get("QA_RANDOM_SEED", 20260130))
    qa_sample_n = int(cfg.get("QA_SAMPLE_N", 18))
    qa_force = list(cfg.get("QA_FORCE_INCLUDE") or [0, 100, 250, 290, 400, 500])

    stage12_cfg = {
        "DRIVE_ID": drive_id,
        "KITTI_ROOT": str(cfg.get("KITTI_ROOT") or ""),
        "FRAME_START": frame_start,
        "FRAME_END": frame_end,
        "IMAGE_CAM": image_cam,
        "IMAGE_MODEL_ID": str(cfg.get("IMAGE_MODEL_ID") or "gdino_sam2_v1"),
        "IMAGE_MODEL_ZOO": str(cfg.get("IMAGE_MODEL_ZOO") or "configs/image_model_zoo.yaml"),
        "STAGE1_STRIDE": int(cfg.get("STAGE1_STRIDE", 5)),
        "STAGE1_FORCE_FRAMES": list(cfg.get("STAGE1_FORCE_FRAMES") or [290]),
        "STAGE1_QA_RANDOM_SEED": qa_seed,
        "STAGE1_QA_SAMPLE_N": qa_sample_n,
        "STAGE1_QA_FORCE_INCLUDE": qa_force,
        "STAGE2_WINDOW_PRE": int(cfg.get("STAGE2_WINDOW_PRE", 30)),
        "STAGE2_WINDOW_POST": int(cfg.get("STAGE2_WINDOW_POST", 30)),
        "STAGE2_MAX_SEEDS_TOTAL": int(cfg.get("STAGE2_MAX_SEEDS_TOTAL", 30)),
        "STAGE2_PROPAGATE": str(cfg.get("SAM2_VIDEO_PROPAGATE") or "both"),
        "OUTPUT_MASK_FORMAT": str(cfg.get("OUTPUT_MASK_FORMAT") or "png"),
        "OUTPUT_OVERLAY_QA_ONLY": True,
        "OVERWRITE": True,
        "ROI_BOTTOM_CROP": float(cfg.get("ROI_BOTTOM_CROP", 0.50)),
        "ROI_SIDE_CROP": list(cfg.get("ROI_SIDE_CROP") or [0.05, 0.95]),
        "GDINO_TEXT_PROMPT": list(cfg.get("GDINO_TEXT_PROMPT") or []),
        "GDINO_BOX_TH": float(cfg.get("GDINO_BOX_TH", 0.23)),
        "GDINO_TEXT_TH": float(cfg.get("GDINO_TEXT_TH", 0.23)),
        "GDINO_TOPK": int(cfg.get("GDINO_TOPK", 12)),
        "NMS_IOU_TH": float(cfg.get("NMS_IOU_TH", 0.50)),
        "MAX_SEEDS_PER_FRAME": int(cfg.get("MAX_SEEDS_PER_FRAME", 4)),
        "SAM2_IMAGE_MAX_MASKS": int(cfg.get("SAM2_IMAGE_MAX_MASKS", 2)),
        "SAM2_MASK_MIN_AREA_PX": int(cfg.get("SAM2_MASK_MIN_AREA_PX", 600)),
        "BOX_ASPECT_MIN": float(cfg.get("BOX_ASPECT_MIN", 2.0)),
        "BOX_H_MAX_RATIO": float(cfg.get("BOX_H_MAX_RATIO", 0.35)),
        "BOX_CENTER_V_MIN_RATIO": float(cfg.get("BOX_CENTER_V_MIN_RATIO", 0.50)),
        "MASK_MRR_ASPECT_MIN": float(cfg.get("MASK_MRR_ASPECT_MIN", 2.0)),
        "STRIPE_MIN_COUNT": int(cfg.get("STRIPE_MIN_COUNT", 6)),
        "STRIPE_ASPECT_RATIO_MIN": float(cfg.get("STRIPE_ASPECT_RATIO_MIN", 2.0)),
    }
    stage12_cfg_path = run_dir / "stage12_config.yaml"
    stage12_cfg_path.write_text(yaml.safe_dump(stage12_cfg, sort_keys=False), encoding="utf-8")

    py = sys.executable
    cmd_stage12 = [py, "scripts/run_image_stage12_sparse_seed_0010_f000_500.py", "--config", str(stage12_cfg_path)]
    subprocess.run(cmd_stage12, check=True)

    stage12_run = _find_latest_run("image_stage12_sparse_seed_0010_000_500_*")
    if stage12_run is None:
        write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "stage12_run_not_found"})
        raise SystemExit("stage12 run not found")

    stage12_decision = {}
    stage12_decision_path = stage12_run / "decision.json"
    if stage12_decision_path.exists():
        stage12_decision = json.loads(stage12_decision_path.read_text(encoding="utf-8"))
        if str(stage12_decision.get("status", "")).upper() == "FAIL":
            write_json(run_dir / "decision.json", {"status": "FAIL", "reason": "stage12_failed", "stage12": stage12_decision})
            raise SystemExit("stage12 failed")

    world_cfg = {
        "drive_id": drive_id,
        "kitti_root": str(cfg.get("KITTI_ROOT") or ""),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "image_cam": image_cam,
        "input_stage12_run": str(stage12_run),
        "input_mask_dir": "stage2/merged_masks",
        "qa_frames_path": "qa/qa_frames.json",
        "output_epsg": int(cfg.get("OUTPUT_EPSG", 32632)),
        "backproject": {
            "use_dtm": bool(cfg.get("USE_DTM", True)),
            "dtm_path": str(cfg.get("DTM_PATH", "auto_latest_clean_dtm")),
            "fixed_plane_z0_mode": str(cfg.get("FIXED_PLANE_Z0_MODE", "dtm_median")),
            "pixel_use_bottom_crop": float(cfg.get("PIXEL_FILTER_STRICT_V_MIN", 0.45)),
            "pixel_use_side_crop": list(cfg.get("PIXEL_SIDE_CROP") or [0.05, 0.95]),
            "pixel_use_bottom_crop_fallback": float(cfg.get("PIXEL_FILTER_FALLBACK_V_MIN", 0.30)),
            "pixel_use_side_crop_fallback": list(cfg.get("PIXEL_SIDE_CROP") or [0.05, 0.95]),
            "contour_sample_step_px": int(cfg.get("CONTOUR_SAMPLE_STEP_PX", 2)),
            "min_valid_world_pts": int(cfg.get("MIN_VALID_WORLD_PTS", 60)),
            "dtm_iterations": int(cfg.get("DTM_ITERATIONS", 2)),
            "max_ray_t_m": 200.0,
            "min_area_px": int(cfg.get("MIN_AREA_PX_PRESENT", 200)),
            "camera_height_m": float(cfg.get("CAMERA_HEIGHT_M", 1.65)),
        },
        "merge": {
            "merge_dist_m": float(cfg.get("MERGE_DIST_M", 2.0)),
            "merge_iou_min": float(cfg.get("MERGE_IOU_MIN", 0.20)),
            "min_support_frames": int(cfg.get("MIN_SUPPORT_FRAMES", 3)),
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
    roundtrip_src = world_run / "images" / "qa_montage_roundtrip.png"
    if roundtrip_src.exists():
        _copytree(roundtrip_src, run_dir / "qa" / "montage_roundtrip_qa.png")
    else:
        canonical_src = world_run / "images" / "qa_montage_canonical.png"
        _copytree(canonical_src, run_dir / "qa" / "montage_roundtrip_qa.png")

    qa_frames = []
    qa_json_path = run_dir / "qa" / "qa_frames.json"
    if qa_json_path.exists():
        qa_json = json.loads(qa_json_path.read_text(encoding="utf-8"))
        qa_frames = [int(v) for v in qa_json.get("frames") or []]

    # normalize stage2 overlay names for QA frames
    stage2_overlays = run_dir / "stage2" / "overlays"
    for frame in qa_frames:
        frame_id = f"{frame:010d}"
        src = stage2_overlays / f"frame_{frame_id}_overlay.png"
        dst = stage2_overlays / f"frame_{frame_id}_overlay_stage2.png"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # copy merged masks into frames/*/mask.png for QA frames (include 290)
    merged_masks_dir = run_dir / "stage2" / "merged_masks"
    overlays_dir = world_run / "overlays"
    for frame in sorted(set(qa_frames + [290])):
        frame_id = f"{frame:010d}"
        frame_dir = run_dir / "frames" / frame_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        mask_src = merged_masks_dir / f"frame_{frame_id}.png"
        if mask_src.exists():
            shutil.copy2(mask_src, frame_dir / "mask.png")
        overlay_src = overlays_dir / f"frame_{frame_id}_overlay_canonical.png"
        if overlay_src.exists():
            shutil.copy2(overlay_src, frame_dir / "overlay_roundtrip.png")

    per_frame_landing_csv = run_dir / "tables" / "per_frame_landing.csv"
    roundtrip_csv = run_dir / "tables" / "roundtrip_px_errors.csv"
    merged_gpkg = run_dir / "merged" / "crosswalk_candidates_canonical_utm32.gpkg"

    world_decision = {}
    world_decision_path = world_run / "decision.json"
    if world_decision_path.exists():
        world_decision = json.loads(world_decision_path.read_text(encoding="utf-8"))

    n_pixel_present = int(world_decision.get("n_pixel_present", 0))
    n_ok = int(world_decision.get("n_world_ok", 0))
    ok_rate = float(world_decision.get("ok_rate", 0.0)) if n_pixel_present else 0.0
    merged_count = _count_merged_candidates(merged_gpkg)

    seed_total = int(stage12_decision.get("seed_total", 0))
    seeds_used = int(stage12_decision.get("seeds_used", 0))

    seed_ok = _seed_ok(per_frame_landing_csv, 290)
    qa_p90_max = _qa_roundtrip_p90_max(roundtrip_csv, qa_frames)
    qa_p90_290 = _qa_roundtrip_p90_for_frame(roundtrip_csv, 290)
    top_reasons = _top_backproject_reasons(per_frame_landing_csv, 3)

    status = "PASS"
    fail_reasons = []

    if str(world_decision.get("status", "")).upper() == "FAIL":
        status = "FAIL"
        fail_reasons.append("world_fail")
    if not seed_ok:
        status = "FAIL"
        fail_reasons.append("seed_290_not_ok")
    if n_ok == 0:
        status = "FAIL"
        fail_reasons.append("n_ok_zero")
    if qa_p90_290 is None or qa_p90_290 > 8:
        status = "FAIL"
        fail_reasons.append("roundtrip_290_p90")
    if qa_p90_max is not None and qa_p90_max > 8:
        status = "FAIL"
        fail_reasons.append("roundtrip_qa_p90")

    if status != "FAIL":
        if ok_rate < 0.6 or merged_count == 0:
            status = "WARN"

    decision = {
        "status": status,
        "fail_reasons": fail_reasons,
        "seed_total": seed_total,
        "seeds_used": seeds_used,
        "n_pixel_present": n_pixel_present,
        "n_ok": n_ok,
        "ok_rate": round(ok_rate, 4) if n_pixel_present else 0.0,
        "merged_candidate_count": merged_count,
        "qa_roundtrip_p90_max": qa_p90_max,
        "qa_roundtrip_p90_290": qa_p90_290,
        "seed_290_ok": seed_ok,
        "top_backproject_failed_reasons": [{"reason": r, "count": c} for r, c in top_reasons],
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
        "# Image Stage12 Stride5 to World (0010 f0-500)",
        "",
        f"- run_id: {run_id}",
        f"- stage12_run: {stage12_run}",
        f"- world_run: {world_run}",
        f"- status: {status}",
        f"- seed_total: {seed_total}",
        f"- seeds_used: {seeds_used}",
        f"- n_pixel_present: {n_pixel_present}",
        f"- n_ok: {n_ok}",
        f"- ok_rate: {ok_rate:.3f}" if n_pixel_present else "- ok_rate: 0.000",
        f"- merged_candidate_count: {merged_count}",
        "",
        "## outputs",
        "- qa/montage_stage1_qa.png",
        "- qa/montage_stage2_qa.png",
        "- qa/montage_roundtrip_qa.png",
        "- merged/crosswalk_candidates_canonical_utm32.gpkg",
        "- tables/per_frame_stage1_counts.csv",
        "- tables/per_frame_mask_area.csv",
        "- tables/per_frame_landing.csv",
        "- tables/roundtrip_px_errors.csv",
        "- decision: decision.json",
    ]
    write_text(run_dir / "report.md", "\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

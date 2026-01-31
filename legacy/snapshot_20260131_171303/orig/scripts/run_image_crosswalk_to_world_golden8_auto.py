from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
from scripts.pipeline_common import now_ts, setup_logging, write_csv, write_json, write_text


CFG_DEFAULT = Path("configs/image_crosswalk_to_world_golden8_auto.yaml")
GOLDEN_LIST = Path("configs/golden_drives.txt")


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _find_data_root(cfg_root: str) -> Path:
    if cfg_root:
        path = Path(cfg_root)
        if path.exists():
            return path
    env_root = os.environ.get("POC_DATA_ROOT", "").strip()
    if env_root:
        path = Path(env_root)
        if path.exists():
            return path
    default_root = Path(r"E:\\KITTI360\\KITTI-360")
    if default_root.exists():
        return default_root
    raise SystemExit("missing data root: set POC_DATA_ROOT or config.KITTI_ROOT")


def _find_image_dir(data_root: Path, drive: str, camera: str) -> Path:
    candidates = [
        data_root / "data_2d_raw" / drive / camera / "data_rect",
        data_root / "data_2d_raw" / drive / camera / "data",
        data_root / drive / camera / "data_rect",
        data_root / drive / camera / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"image data not found for drive={drive} camera={camera}")


def _scan_frames(image_dir: Path) -> List[int]:
    frames = set()
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in image_dir.glob(ext):
            m = re.match(r"^(\d+)", p.stem)
            if not m:
                continue
            try:
                frames.add(int(m.group(1)))
            except Exception:
                continue
    return sorted(frames)


def _nearest_frame(target: int, frames: List[int]) -> int:
    return min(frames, key=lambda x: (abs(x - target), x))


def _qa_force_frames(
    frames: List[int],
    frame_start: int,
    frame_end: int,
    seed_frame: int,
    tokens: List[str],
) -> List[int]:
    out = set()
    span = max(1, frame_end - frame_start)
    for t in tokens:
        token = str(t).strip().lower()
        if token == "start":
            out.add(frame_start)
        elif token == "end":
            out.add(frame_end)
        elif token == "seed":
            out.add(seed_frame)
        elif token.endswith("%"):
            try:
                pct = float(token.replace("%", "")) / 100.0
            except Exception:
                continue
            raw = int(round(frame_start + pct * span))
            out.add(raw)
    snapped = set()
    for f in out:
        snapped.add(_nearest_frame(f, frames))
    return sorted(snapped)


def _detect_new_run(pattern: str, before: set) -> Optional[Path]:
    runs_dir = Path("runs")
    after = {p for p in runs_dir.glob(pattern) if p.is_dir()}
    new = list(after - before)
    if new:
        new.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return new[0]
    if after:
        items = sorted(after, key=lambda p: p.stat().st_mtime, reverse=True)
        return items[0]
    return None


def _find_latest_dtm_for_drive(drive_id: str) -> Optional[Path]:
    runs_dir = Path("runs")
    candidates = []
    for p in runs_dir.glob(f"lidar_ground_{drive_id}_*"):
        if not p.is_dir():
            continue
        cand = p / "rasters" / "dtm_median_clean_utm32.tif"
        if cand.exists():
            candidates.append(cand)
        else:
            cand2 = p / "rasters" / "dtm_median_utm32.tif"
            if cand2.exists():
                candidates.append(cand2)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _count_merged_candidates(gpkg_path: Path) -> int:
    try:
        import geopandas as gpd

        if not gpkg_path.exists():
            return 0
        gdf = gpd.read_file(gpkg_path, layer="crosswalk_candidates")
        return int(len(gdf))
    except Exception:
        return 0


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


def _seed_has_ok(per_frame_csv: Path, seed_frame: int) -> bool:
    rows = _load_csv_rows(per_frame_csv)
    target = f"{seed_frame:010d}"
    for row in rows:
        if str(row.get("frame_id", "")).strip() == target:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CFG_DEFAULT))
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    if not cfg:
        raise SystemExit(f"missing config: {args.config}")

    if not GOLDEN_LIST.exists():
        raise SystemExit(f"missing golden list: {GOLDEN_LIST}")

    drives = [line.strip() for line in GOLDEN_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not drives:
        raise SystemExit("golden list empty")

    run_id = now_ts()
    run_dir = Path("runs") / f"crosswalk_golden8_auto_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")

    (run_dir / "merged").mkdir(parents=True, exist_ok=True)
    (run_dir / "drives").mkdir(parents=True, exist_ok=True)

    data_root = _find_data_root(str(cfg.get("KITTI_ROOT") or ""))
    image_cam = str(cfg.get("IMAGE_CAM") or "image_00")

    summary_rows: List[Dict[str, object]] = []
    qa_index_rows: List[Dict[str, object]] = []
    merged_items = []

    for drive_id in drives:
        drive_dir = run_dir / "drives" / drive_id
        drive_dir.mkdir(parents=True, exist_ok=True)

        drive_status = "FAIL"
        fail_reason = ""

        try:
            image_dir = _find_image_dir(data_root, drive_id, image_cam)
            frames = _scan_frames(image_dir)
            if not frames:
                raise SystemExit("no frames found")
            frame_start = int(frames[0])
            frame_end = int(frames[-1])

            seed_pref = int(cfg.get("SEED_FRAME_PREFERRED", 290))
            seed_frame = seed_pref if seed_pref in frames else _nearest_frame(seed_pref, frames)

            qa_force_tokens = list(cfg.get("QA_FORCE_INCLUDE_RELATIVE") or [])
            qa_force = _qa_force_frames(frames, frame_start, frame_end, seed_frame, qa_force_tokens)

            stage12_cfg = {
                "DRIVE_ID": drive_id,
                "KITTI_ROOT": str(cfg.get("KITTI_ROOT") or ""),
                "FRAME_START": frame_start,
                "FRAME_END": frame_end,
                "IMAGE_CAM": image_cam,
                "IMAGE_MODEL_ID": str(cfg.get("IMAGE_MODEL_ID") or "gdino_sam2_v1"),
                "IMAGE_MODEL_ZOO": str(cfg.get("IMAGE_MODEL_ZOO") or "configs/image_model_zoo.yaml"),
                "SEED_FRAME": seed_frame,
                "STAGE1_STRIDE": int(cfg.get("STAGE1_STRIDE", 5)),
                "STAGE1_FORCE_FRAMES": [seed_frame],
                "STAGE1_QA_RANDOM_SEED": int(cfg.get("QA_RANDOM_SEED", 20260130)),
                "STAGE1_QA_SAMPLE_N": int(cfg.get("QA_SAMPLE_N", 18)),
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
            stage12_cfg_path = drive_dir / "stage12_config.yaml"
            stage12_cfg_path.write_text(yaml.safe_dump(stage12_cfg, sort_keys=False), encoding="utf-8")

            runs_dir = Path("runs")
            before_stage12 = {p for p in runs_dir.glob("image_stage12_sparse_seed_*") if p.is_dir()}
            py = sys.executable
            cmd_stage12 = [py, "scripts/run_image_stage12_sparse_seed_0010_f000_500.py", "--config", str(stage12_cfg_path)]
            subprocess.run(cmd_stage12, check=True)
            stage12_run = _detect_new_run("image_stage12_sparse_seed_*", before_stage12)
            if stage12_run is None:
                raise SystemExit("stage12 run not found")

            stage12_decision = {}
            stage12_decision_path = stage12_run / "decision.json"
            if stage12_decision_path.exists():
                stage12_decision = json.loads(stage12_decision_path.read_text(encoding="utf-8"))
                if str(stage12_decision.get("status", "")).upper() == "FAIL":
                    raise SystemExit("stage12 failed")

            dtm_cfg = str(cfg.get("DTM_PATH", "auto_latest_clean_dtm_per_drive"))
            dtm_path = None
            use_dtm = bool(cfg.get("USE_DTM", True))
            if dtm_cfg == "auto_latest_clean_dtm_per_drive":
                dtm_path = _find_latest_dtm_for_drive(drive_id)
                if dtm_path is None:
                    use_dtm = False
            elif dtm_cfg:
                dtm_path = Path(dtm_cfg)
                if not dtm_path.exists():
                    dtm_path = None
                    use_dtm = False

            world_cfg = {
                "drive_id": drive_id,
                "kitti_root": str(cfg.get("KITTI_ROOT") or ""),
                "frame_start": frame_start,
                "frame_end": frame_end,
                "image_cam": image_cam,
                "cycle_gate_enable": bool(cfg.get("CYCLE_GATE_ENABLE", False)),
                "input_stage12_run": str(stage12_run),
                "input_mask_dir": "stage2/merged_masks",
                "qa_frames_path": "qa/qa_frames.json",
                "output_epsg": int(cfg.get("OUTPUT_EPSG", 32632)),
                "backproject": {
                    "use_dtm": bool(use_dtm),
                    "dtm_path": str(dtm_path) if dtm_path else "",
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
            world_cfg_path = drive_dir / "world_candidates_config.yaml"
            world_cfg_path.write_text(yaml.safe_dump(world_cfg, sort_keys=False), encoding="utf-8")

            before_world = {p for p in runs_dir.glob("world_crosswalk_candidates_0010_250_500_*") if p.is_dir()}
            cmd_world = [py, "scripts/run_world_crosswalk_candidates_0010_f250_500.py", "--config", str(world_cfg_path)]
            subprocess.run(cmd_world, check=True)
            world_run = _detect_new_run("world_crosswalk_candidates_0010_250_500_*", before_world)
            if world_run is None:
                raise SystemExit("world run not found")

            # copy outputs
            (drive_dir / "qa").mkdir(parents=True, exist_ok=True)
            (drive_dir / "stage1").mkdir(parents=True, exist_ok=True)
            (drive_dir / "stage2").mkdir(parents=True, exist_ok=True)
            (drive_dir / "frames").mkdir(parents=True, exist_ok=True)
            (drive_dir / "merged").mkdir(parents=True, exist_ok=True)
            (drive_dir / "tables").mkdir(parents=True, exist_ok=True)

            def _copy(src: Path, dst: Path) -> None:
                if not src.exists():
                    return
                if src.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                else:
                    shutil.copytree(src, dst, dirs_exist_ok=True)

            _copy(stage12_run / "qa", drive_dir / "qa")
            _copy(stage12_run / "stage1" / "seeds_index.csv", drive_dir / "stage1" / "seeds_index.csv")
            _copy(stage12_run / "stage2" / "merged_masks", drive_dir / "stage2" / "merged_masks")
            _copy(stage12_run / "tables" / "per_frame_stage1_counts.csv", drive_dir / "tables" / "per_frame_stage1_counts.csv")
            _copy(stage12_run / "tables" / "per_frame_mask_area.csv", drive_dir / "tables" / "per_frame_mask_area.csv")
            _copy(stage12_run / "tables" / "missing_frames.csv", drive_dir / "tables" / "missing_frames.csv")

            _copy(world_run / "frames", drive_dir / "frames")
            _copy(world_run / "merged", drive_dir / "merged")
            _copy(world_run / "tables" / "per_frame_landing.csv", drive_dir / "tables" / "per_frame_landing.csv")
            _copy(world_run / "tables" / "roundtrip_px_errors.csv", drive_dir / "tables" / "roundtrip_px_errors.csv")

            roundtrip_src = world_run / "images" / "qa_montage_roundtrip.png"
            if roundtrip_src.exists():
                _copy(roundtrip_src, drive_dir / "qa" / "montage_roundtrip_qa.png")
            else:
                _copy(world_run / "images" / "qa_montage_canonical.png", drive_dir / "qa" / "montage_roundtrip_qa.png")

            # copy overlay_roundtrip for QA frames
            qa_frames = []
            qa_json = drive_dir / "qa" / "qa_frames.json"
            if qa_json.exists():
                qa_frames = [int(v) for v in json.loads(qa_json.read_text(encoding="utf-8")).get("frames", [])]
            overlays_dir = world_run / "overlays"
            for frame in sorted(set(qa_frames + [seed_frame])):
                frame_id = f"{frame:010d}"
                src = overlays_dir / f"frame_{frame_id}_overlay_canonical.png"
                dst = drive_dir / "frames" / frame_id / "overlay_roundtrip.png"
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

            # decision
            world_decision = {}
            world_decision_path = world_run / "decision.json"
            if world_decision_path.exists():
                world_decision = json.loads(world_decision_path.read_text(encoding="utf-8"))

            n_pixel_present = int(world_decision.get("n_pixel_present", 0))
            n_ok = int(world_decision.get("n_world_ok", world_decision.get("frames_ok", 0)))
            ok_rate = float(world_decision.get("ok_rate", 0.0)) if n_pixel_present else 0.0
            merged_count = _count_merged_candidates(drive_dir / "merged" / "crosswalk_candidates_canonical_utm32.gpkg")

            seed_total = int(stage12_decision.get("seed_total", 0))
            seeds_used = int(stage12_decision.get("seeds_used", 0))

            if n_pixel_present > 0 and n_ok == 0:
                drive_status = "FAIL"
                fail_reason = "n_ok_zero"
            elif str(stage12_decision.get("status", "")).upper() == "FAIL":
                drive_status = "FAIL"
                fail_reason = "stage12_failed"
            elif ok_rate < 0.6 or merged_count == 0:
                drive_status = "WARN"
            else:
                drive_status = "PASS"

            decision = {
                "status": drive_status,
                "fail_reason": fail_reason,
                "drive_id": drive_id,
                "frame_range_detected": [frame_start, frame_end],
                "seed_frame": seed_frame,
                "seed_total": seed_total,
                "seeds_used": seeds_used,
                "n_pixel_present": n_pixel_present,
                "n_ok": n_ok,
                "ok_rate": round(ok_rate, 4) if n_pixel_present else 0.0,
                "merged_candidate_count": merged_count,
                "stage12_run": str(stage12_run),
                "world_run": str(world_run),
                "top_backproject_failed_reasons": [
                    {"reason": r, "count": c}
                    for r, c in _top_backproject_reasons(drive_dir / "tables" / "per_frame_landing.csv")
                ],
            }
            write_json(drive_dir / "decision.json", decision)

            resolved = dict(cfg)
            resolved.update(
                {
                    "resolved": {
                        "drive_id": drive_id,
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "seed_frame": seed_frame,
                        "stage12_run": str(stage12_run),
                        "world_run": str(world_run),
                    }
                }
            )
            resolved_path = drive_dir / "resolved_config.yaml"
            resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
            (drive_dir / "params_hash.txt").write_text(_hash_file(resolved_path), encoding="utf-8")

            report_lines = [
                f"# Golden8 Crosswalk Report ({drive_id})",
                "",
                f"- frame_range_detected: {frame_start}-{frame_end}",
                f"- seed_frame: {seed_frame}",
                f"- seed_total: {seed_total}",
                f"- seeds_used: {seeds_used}",
                f"- n_pixel_present: {n_pixel_present}",
                f"- n_ok: {n_ok}",
                f"- ok_rate: {ok_rate:.3f}" if n_pixel_present else "- ok_rate: 0.000",
                f"- merged_candidate_count: {merged_count}",
                f"- status: {drive_status}",
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
            ]
            write_text(drive_dir / "report.md", "\n".join(report_lines) + "\n")

            # collect for summary
            summary_rows.append(
                {
                    "drive_id": drive_id,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "seed_frame": seed_frame,
                    "seed_total": seed_total,
                    "seeds_used": seeds_used,
                    "n_pixel_present": n_pixel_present,
                    "n_ok": n_ok,
                    "ok_rate": round(ok_rate, 4) if n_pixel_present else 0.0,
                    "merged_count": merged_count,
                    "status": drive_status,
                }
            )
            qa_index_rows.append(
                {
                    "drive_id": drive_id,
                    "frame_range": f"{frame_start}-{frame_end}",
                    "seed_frame": seed_frame,
                    "seed_count": seed_total,
                    "ok_rate": round(ok_rate, 4) if n_pixel_present else 0.0,
                    "merged_count": merged_count,
                    "status": drive_status,
                }
            )

            # merged candidates
            gpkg = drive_dir / "merged" / "crosswalk_candidates_canonical_utm32.gpkg"
            try:
                import geopandas as gpd

                if gpkg.exists():
                    gdf = gpd.read_file(gpkg, layer="crosswalk_candidates")
                    if not gdf.empty:
                        gdf["drive_id"] = drive_id
                        merged_items.append(gdf)
            except Exception:
                pass

        except Exception as exc:
            write_json(drive_dir / "decision.json", {"status": "FAIL", "reason": str(exc)})
            summary_rows.append(
                {
                    "drive_id": drive_id,
                    "frame_start": "",
                    "frame_end": "",
                    "seed_frame": "",
                    "seed_total": 0,
                    "seeds_used": 0,
                    "n_pixel_present": 0,
                    "n_ok": 0,
                    "ok_rate": 0.0,
                    "merged_count": 0,
                    "status": "FAIL",
                }
            )
            qa_index_rows.append(
                {
                    "drive_id": drive_id,
                    "frame_range": "",
                    "seed_frame": "",
                    "seed_count": 0,
                    "ok_rate": 0.0,
                    "merged_count": 0,
                    "status": "FAIL",
                }
            )

    # merge all canonical candidates
    merged_out = run_dir / "merged" / "crosswalk_candidates_canonical_utm32.gpkg"
    try:
        import geopandas as gpd
        import pandas as pd

        if merged_items:
            merged_gdf = gpd.GeoDataFrame(pd.concat(merged_items, ignore_index=True), crs="EPSG:32632")
        else:
            merged_gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:32632")
        merged_gdf.to_file(merged_out, layer="crosswalk_candidates", driver="GPKG")
    except Exception:
        pass

    # qa_index.csv
    write_csv(
        run_dir / "merged" / "qa_index.csv",
        qa_index_rows,
        ["drive_id", "frame_range", "seed_frame", "seed_count", "ok_rate", "merged_count", "status"],
    )

    # summary
    fail_count = sum(1 for r in summary_rows if r.get("status") == "FAIL")
    if fail_count <= 1:
        overall_status = "PASS"
    elif fail_count <= 3:
        overall_status = "WARN"
    else:
        overall_status = "FAIL"

    summary = {
        "status": overall_status,
        "fail_count": fail_count,
        "drive_total": len(drives),
        "drives": summary_rows,
        "merged_gpkg": str(merged_out),
    }
    write_json(run_dir / "run_summary.json", summary)

    md_lines = [
        "# Crosswalk Golden8 Auto Summary",
        "",
        f"- status: {overall_status}",
        f"- drive_total: {len(drives)}",
        f"- fail_count: {fail_count}",
        "",
        "## drives",
    ]
    for row in summary_rows:
        md_lines.append(
            f"- {row['drive_id']}: range={row['frame_start']}-{row['frame_end']} seed={row['seed_frame']} "
            f"ok_rate={row['ok_rate']} merged={row['merged_count']} status={row['status']}"
        )
    md_lines += [
        "",
        "## outputs",
        f"- merged canonical: merged/crosswalk_candidates_canonical_utm32.gpkg",
        f"- qa_index: merged/qa_index.csv",
        f"- run_summary: run_summary.json",
    ]
    write_text(run_dir / "run_summary.md", "\n".join(md_lines) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

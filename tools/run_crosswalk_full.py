from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import geopandas as gpd
import numpy as np
import yaml
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_image_sample_index import _find_image_dir, _list_images, _extract_frame_id


LOG = logging.getLogger("run_crosswalk_full")


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("run_crosswalk_full")


def _safe_unlink(path: Path) -> None:
    if path.exists():
        path.unlink()


def _load_index_drives(path: Path) -> List[str]:
    drives = []
    if not path.exists():
        return drives
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        drive = row.get("drive_id") or row.get("drive") or row.get("tile_id")
        if drive:
            drives.append(str(drive))
    return sorted(set(drives))


def _write_index(records: Iterable[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _safe_unlink(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _build_stride_index(
    data_root: Path,
    drives: List[str],
    stride: int,
    out_path: Path,
    camera: str,
) -> None:
    records: List[dict] = []
    for drive in drives:
        img_dir = _find_image_dir(data_root, drive, camera)
        if not img_dir:
            LOG.warning("drive=%s image dir not found", drive)
            continue
        images = _list_images(img_dir)
        if not images:
            continue
        sampled = images[::max(1, stride)]
        for path in sampled:
            records.append(
                {
                    "drive_id": drive,
                    "camera": camera,
                    "frame_id": _extract_frame_id(path),
                    "image_path": str(path),
                    "scene_profile": "car",
                }
            )
    _write_index(records, out_path)
    LOG.info("stage1 index=%s total=%d", out_path, len(records))


def _frame_id_int(frame_id: str) -> int:
    digits = "".join(ch for ch in str(frame_id) if ch.isdigit())
    return int(digits) if digits else -1


def _score_candidate(row: gpd.GeoSeries) -> float:
    inside = float(row.get("inside_road_ratio", 0.0) or 0.0)
    rect = float(row.get("rectangularity", 0.0) or 0.0)
    heading = row.get("heading_diff_to_perp_deg", "")
    heading_score = 0.5
    if heading != "" and heading is not None:
        heading_score = max(0.0, 1.0 - min(float(heading) / 45.0, 1.0))
    return float(0.6 * inside + 0.25 * rect + 0.15 * heading_score)


def _cluster_by_centroid(gdf: gpd.GeoDataFrame, eps_m: float) -> List[List[int]]:
    if gdf.empty:
        return []
    centroids = [geom.centroid for geom in gdf.geometry]
    try:
        from shapely.strtree import STRtree

        tree = STRtree(centroids)
        geom_id = {id(g): idx for idx, g in enumerate(centroids)}
    except Exception:
        tree = None
        geom_id = {}
    parent = list(range(len(centroids)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri = _find(i)
        rj = _find(j)
        if ri != rj:
            parent[rj] = ri

    for i, center in enumerate(centroids):
        candidates = []
        if tree is not None:
            for cand in tree.query(center.buffer(eps_m)):
                if isinstance(cand, (int, np.integer)):
                    idx = int(cand)
                else:
                    idx = geom_id.get(id(cand))
                if idx is not None:
                    candidates.append(idx)
        else:
            candidates = list(range(len(centroids)))
        for j in candidates:
            if j <= i:
                continue
            if center.distance(centroids[j]) <= eps_m:
                _union(i, j)

    clusters: Dict[int, List[int]] = {}
    for idx in range(len(centroids)):
        root = _find(idx)
        clusters.setdefault(root, []).append(idx)
    return list(clusters.values())


def _write_stage1_candidates(cand_gdf: gpd.GeoDataFrame, out_csv: Path) -> gpd.GeoDataFrame:
    if cand_gdf.empty:
        _safe_unlink(out_csv)
        return cand_gdf
    cand = cand_gdf.copy()
    cand["score_total"] = cand.apply(_score_candidate, axis=1)
    cand = cand.sort_values(["drive_id", "score_total"], ascending=[True, False])
    cols = [
        "drive_id",
        "frame_id",
        "candidate_id",
        "score_total",
        "inside_road_ratio",
        "rectangularity",
        "rect_w_m",
        "rect_l_m",
        "heading_diff_to_perp_deg",
        "reject_reasons",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _safe_unlink(out_csv)
    cand[cols].to_csv(out_csv, index=False)
    return cand


def _build_stage2_windows(
    cand_gdf: gpd.GeoDataFrame,
    out_jsonl: Path,
    eps_m: float,
    window_half: int,
) -> List[dict]:
    windows: List[dict] = []
    if cand_gdf.empty:
        _safe_unlink(out_jsonl)
        return windows
    for drive_id, group in cand_gdf.groupby("drive_id"):
        clusters = _cluster_by_centroid(group, eps_m)
        for cluster in clusters:
            hits = group.iloc[cluster].copy()
            hits["score_total"] = hits.apply(_score_candidate, axis=1)
            best = hits.sort_values("score_total", ascending=False).iloc[0]
            center_frame = _frame_id_int(best.get("frame_id"))
            if center_frame < 0:
                continue
            window = {
                "drive_id": drive_id,
                "frame_start": max(0, center_frame - window_half),
                "frame_end": center_frame + window_half,
                "reason": "stage1_candidate",
                "center_frame": center_frame,
                "score_total": float(best.get("score_total", 0.0)),
            }
            windows.append(window)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _safe_unlink(out_jsonl)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in windows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return windows


def _build_stage2_index(
    data_root: Path,
    windows: List[dict],
    out_path: Path,
    camera: str,
) -> None:
    records: List[dict] = []
    for window in windows:
        drive_id = str(window["drive_id"])
        frame_start = int(window["frame_start"])
        frame_end = int(window["frame_end"])
        img_dir = _find_image_dir(data_root, drive_id, camera)
        if not img_dir:
            LOG.warning("stage2 drive=%s image dir missing", drive_id)
            continue
        images = _list_images(img_dir)
        for path in images:
            frame_id = _extract_frame_id(path)
            frame_int = _frame_id_int(frame_id)
            if frame_int < frame_start or frame_int > frame_end:
                continue
            records.append(
                {
                    "drive_id": drive_id,
                    "camera": camera,
                    "frame_id": frame_id,
                    "image_path": str(path),
                    "scene_profile": "car",
                }
            )
    # dedup
    seen = set()
    unique = []
    for row in records:
        key = (row["drive_id"], row["frame_id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    _write_index(unique, out_path)
    LOG.info("stage2 index=%s total=%d", out_path, len(unique))


def _update_config(
    base_cfg: Path,
    overrides: Dict[str, dict],
    out_path: Path,
) -> None:
    cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8")) or {}
    for key, value in overrides.items():
        if key not in cfg or not isinstance(cfg[key], dict):
            cfg[key] = {}
        cfg[key].update(value)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _read_candidates(gpkg_path: Path) -> gpd.GeoDataFrame:
    if not gpkg_path.exists():
        return gpd.GeoDataFrame()
    try:
        return gpd.read_file(gpkg_path, layer="crosswalk_candidate_poly")
    except Exception:
        return gpd.GeoDataFrame()


def _read_final(gpkg_path: Path) -> gpd.GeoDataFrame:
    if not gpkg_path.exists():
        return gpd.GeoDataFrame()
    try:
        return gpd.read_file(gpkg_path, layer="crosswalk_poly")
    except Exception:
        return gpd.GeoDataFrame()


def _write_crosswalk_gpkg(src_gpkg: Path, out_gpkg: Path) -> None:
    layers = {}
    for layer in ("crosswalk_candidate_poly", "crosswalk_poly"):
        try:
            layers[layer] = gpd.read_file(src_gpkg, layer=layer)
        except Exception:
            layers[layer] = gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs="EPSG:32632")
    if out_gpkg.exists():
        out_gpkg.unlink()
    for layer, gdf in layers.items():
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:32632")
        gdf.to_file(out_gpkg, layer=layer, driver="GPKG")


def _select_qa_frames(
    final_gdf: gpd.GeoDataFrame,
    cand_gdf: gpd.GeoDataFrame,
    qa_frames_per_entity: int,
    topn_reject: int,
) -> Tuple[Dict[str, List[str]], List[dict]]:
    per_drive_frames: Dict[str, List[str]] = {}
    if not final_gdf.empty:
        for _, row in final_gdf.iterrows():
            drive_id = str(row.get("drive_id") or "")
            if not drive_id:
                continue
            frame_ids = str(row.get("frame_ids") or "")
            if ".." in frame_ids:
                parts = frame_ids.split("..")
                start = _frame_id_int(parts[0])
                end = _frame_id_int(parts[1].split()[0]) if parts[1] else start
                if start >= 0 and end >= start:
                    frames = [f"{i:010d}" for i in range(start, end + 1)]
                else:
                    frames = []
            else:
                frames = [frame_ids] if frame_ids else []
            if not frames:
                continue
            if qa_frames_per_entity > 0:
                step = max(1, len(frames) // qa_frames_per_entity)
                frames = frames[::step][:qa_frames_per_entity]
            per_drive_frames.setdefault(drive_id, []).extend(frames)

    reject_samples = []
    if not cand_gdf.empty:
        cand = cand_gdf.copy()
        cand["score_total"] = cand.apply(_score_candidate, axis=1)
        cand = cand[cand["reject_reasons"].fillna("") != ""]
        cand = cand.sort_values("score_total", ascending=False).head(topn_reject)
        for _, row in cand.iterrows():
            drive_id = str(row.get("drive_id") or "")
            frame_id = str(row.get("frame_id") or "")
            if drive_id and frame_id:
                per_drive_frames.setdefault(drive_id, []).append(frame_id)
            reject_samples.append(
                {
                    "drive_id": drive_id,
                    "frame_id": frame_id,
                    "reject_reasons": row.get("reject_reasons", ""),
                    "score_total": float(row.get("score_total", 0.0)),
                }
            )

    # de-dup frames
    for drive_id in list(per_drive_frames.keys()):
        frames = sorted(set(per_drive_frames[drive_id]))
        per_drive_frames[drive_id] = frames
    return per_drive_frames, reject_samples


def _filter_qa_assets(
    qa_index_path: Path,
    outputs_dir: Path,
    per_drive_frames: Dict[str, List[str]],
) -> Path:
    qa_gdf = gpd.read_file(qa_index_path)
    if qa_gdf.empty:
        return qa_index_path
    keep = []
    for _, row in qa_gdf.iterrows():
        drive_id = str(row.get("drive_id") or "")
        frame_id = str(row.get("frame_id") or "")
        if frame_id in set(per_drive_frames.get(drive_id, [])):
            keep.append(True)
        else:
            keep.append(False)
    qa_gdf = qa_gdf[keep]
    new_path = outputs_dir / "qa_index_wgs84.geojson"
    if new_path.exists():
        new_path.unlink()
    for idx, row in qa_gdf.iterrows():
        drive_id = str(row.get("drive_id") or "")
        for key in ("overlay_raw_path", "overlay_gated_path", "overlay_entities_path"):
            path = row.get(key, "")
            if not path:
                continue
            name = Path(path).name
            if drive_id:
                qa_gdf.at[idx, key] = str(outputs_dir / "qa_images" / drive_id / name)
    qa_gdf.to_file(new_path, driver="GeoJSON")

    # prune images
    qa_dir = outputs_dir / "qa_images"
    if qa_dir.exists():
        keep_files = set()
        for _, row in qa_gdf.iterrows():
            for key in ("overlay_raw_path", "overlay_gated_path", "overlay_entities_path"):
                path = row.get(key, "")
                if path:
                    keep_files.add(Path(path).resolve())
        for path in qa_dir.rglob("*.png"):
            if path.resolve() not in keep_files:
                path.unlink()
    return new_path


def _write_report(
    out_path: Path,
    stage1_csv: Path,
    final_gdf: gpd.GeoDataFrame,
    reject_samples: List[dict],
) -> None:
    lines = []
    lines.append("# Crosswalk Full Report\n")
    lines.append(f"- report_time: {dt.datetime.now():%Y-%m-%d %H:%M:%S}\n")
    if stage1_csv.exists():
        import pandas as pd

        df = pd.read_csv(stage1_csv)
        lines.append(f"- stage1_candidate_count: {len(df)}\n")
        for drive_id, group in df.groupby("drive_id"):
            lines.append(f"- stage1_{drive_id}_count: {len(group)}")
        lines.append("")
        if "reject_reasons" in df.columns:
            counts = df["reject_reasons"].fillna("").str.split(",", expand=False)
            flat = {}
            for reasons in counts:
                for reason in [r for r in reasons if r]:
                    flat[reason] = flat.get(reason, 0) + 1
            if flat:
                lines.append("## Stage1 Reject Reasons")
                for reason, count in sorted(flat.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"- {reason}: {count}")
                lines.append("")

    lines.append("## Stage2 Final Summary")
    if final_gdf.empty:
        lines.append("- final_count: 0\n")
    else:
        lines.append(f"- final_count: {len(final_gdf)}")
        for drive_id, group in final_gdf.groupby("drive_id"):
            lines.append(f"- final_{drive_id}_count: {len(group)}")
        lines.append("")
        frames = [int(v) for v in final_gdf["frames_hit"].tolist() if v]
        if frames:
            lines.append(f"- frames_hit_p50: {np.percentile(frames, 50):.1f}")
            lines.append(f"- frames_hit_p90: {np.percentile(frames, 90):.1f}")
        if "rect_w_m" in final_gdf.columns:
            widths = [float(v) for v in final_gdf["rect_w_m"].tolist() if v]
            lengths = [float(v) for v in final_gdf["rect_l_m"].tolist() if v]
            if widths and lengths:
                lines.append(f"- rect_w_p50: {np.percentile(widths, 50):.2f}")
                lines.append(f"- rect_l_p50: {np.percentile(lengths, 50):.2f}")
        lines.append("")

    lines.append("## Top Reject Samples")
    if not reject_samples:
        lines.append("- none\n")
    else:
        for row in reject_samples[:10]:
            lines.append(
                f"- {row.get('drive_id')}:{row.get('frame_id')} score={row.get('score_total'):.3f} reject={row.get('reject_reasons')}"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--image-run", required=True)
    ap.add_argument("--image-provider", default="grounded_sam2_v1")
    ap.add_argument("--road-root", required=True)
    ap.add_argument("--config", default="configs/road_entities.yaml")
    ap.add_argument("--index", default="")
    ap.add_argument("--full-stride", type=int, default=5)
    ap.add_argument("--stage2", type=int, default=1)
    ap.add_argument("--min-frames-hit-final", type=int, default=3)
    ap.add_argument("--min-frames-hit-coarse", type=int, default=2)
    ap.add_argument("--qa-frames-per-entity", type=int, default=20)
    ap.add_argument("--qa-topn-reject", type=int, default=30)
    args = ap.parse_args()

    log = _setup_logger()
    data_root = Path(os.environ.get("POC_DATA_ROOT", ""))
    if not data_root.exists():
        log.error("POC_DATA_ROOT not set or invalid.")
        return 2

    run_dir = Path(args.out_dir) if args.out_dir else Path("runs") / f"crosswalk_full_{dt.datetime.now():%Y%m%d_%H%M%S}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    outputs_dir = run_dir / "outputs"
    debug_dir = run_dir / "debug"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    if args.index:
        index_path = Path(args.index)
    else:
        golden8_index = Path("runs") / "image_samples_golden8_full" / "sample_index.jsonl"
        index_path = golden8_index if golden8_index.exists() else (run_dir / "stage1_index.jsonl")
    if not index_path.exists():
        drives = _load_index_drives(Path(args.index)) if args.index else []
        if not drives:
            candidate = data_root / "data_2d_raw"
            if candidate.exists():
                drives = sorted([p.name for p in candidate.iterdir() if p.is_dir()])
        _build_stride_index(data_root, drives, args.full_stride, index_path, "image_00")

    stage1_cfg = debug_dir / "crosswalk_stage1.yaml"
    _update_config(
        Path(args.config),
        {
            "crosswalk_final": {
                "min_frames_hit": int(args.min_frames_hit_coarse),
                "max_heading_diff_deg": 35.0,
                "min_rectangularity": 0.35,
                "min_rect_w_m": 1.0,
                "max_rect_w_m": 12.0,
                "min_rect_l_m": 2.0,
                "max_rect_l_m": 30.0,
            }
        },
        stage1_cfg,
    )

    stage1_dir = run_dir / "stage1"
    if stage1_dir.exists():
        shutil.rmtree(stage1_dir)
    cmd = [
        sys.executable,
        "tools/build_road_entities.py",
        "--index",
        str(index_path),
        "--image-run",
        str(args.image_run),
        "--image-provider",
        str(args.image_provider),
        "--config",
        str(stage1_cfg),
        "--road-root",
        str(args.road_root),
        "--out-dir",
        str(stage1_dir),
        "--emit-qa-images",
        "0",
    ]
    log.info("stage1: %s", " ".join(cmd))
    if subprocess.run(cmd, check=False).returncode != 0:
        log.error("stage1 failed")
        return 3

    stage1_gpkg = stage1_dir / "outputs" / "road_entities_utm32.gpkg"
    cand_gdf = _read_candidates(stage1_gpkg)
    stage1_csv = debug_dir / "stage1_candidates.csv"
    cand_scored = _write_stage1_candidates(cand_gdf, stage1_csv)

    windows = _build_stage2_windows(
        cand_scored,
        debug_dir / "stage2_windows.jsonl",
        eps_m=6.0,
        window_half=40,
    )
    if not windows:
        log.warning("no stage2 windows, skipping stage2")
        return 0

    stage2_index = run_dir / "stage2_index.jsonl"
    _build_stage2_index(data_root, windows, stage2_index, "image_00")

    stage2_cfg = debug_dir / "crosswalk_stage2.yaml"
    _update_config(Path(args.config), {"crosswalk_final": {"min_frames_hit": int(args.min_frames_hit_final)}}, stage2_cfg)

    stage2_dir = run_dir / "stage2"
    if stage2_dir.exists():
        shutil.rmtree(stage2_dir)
    cmd = [
        sys.executable,
        "tools/build_road_entities.py",
        "--index",
        str(stage2_index),
        "--image-run",
        str(args.image_run),
        "--image-provider",
        str(args.image_provider),
        "--config",
        str(stage2_cfg),
        "--road-root",
        str(args.road_root),
        "--out-dir",
        str(stage2_dir),
        "--emit-qa-images",
        "1",
    ]
    log.info("stage2: %s", " ".join(cmd))
    if subprocess.run(cmd, check=False).returncode != 0:
        log.error("stage2 failed")
        return 4

    stage2_gpkg = stage2_dir / "outputs" / "road_entities_utm32.gpkg"
    final_gdf = _read_final(stage2_gpkg)
    cand_gdf = _read_candidates(stage2_gpkg)

    out_gpkg = outputs_dir / "crosswalk_entities_utm32.gpkg"
    _write_crosswalk_gpkg(stage2_gpkg, out_gpkg)

    out_wgs84 = outputs_dir / "crosswalk_entities_wgs84.gpkg"
    if out_wgs84.exists():
        out_wgs84.unlink()
    gdf_wgs84 = gpd.read_file(stage2_dir / "outputs" / "road_entities_wgs84.gpkg", layer="crosswalk_candidate_poly")
    gdf_final = gpd.read_file(stage2_dir / "outputs" / "road_entities_wgs84.gpkg", layer="crosswalk_poly")
    gdf_wgs84.to_file(out_wgs84, layer="crosswalk_candidate_poly", driver="GPKG")
    gdf_final.to_file(out_wgs84, layer="crosswalk_poly", driver="GPKG")

    qa_index_path = stage2_dir / "outputs" / "qa_index_wgs84.geojson"
    per_drive_frames, reject_samples = _select_qa_frames(
        final_gdf,
        cand_gdf,
        args.qa_frames_per_entity,
        args.qa_topn_reject,
    )
    if (stage2_dir / "outputs" / "qa_images").exists():
        if (outputs_dir / "qa_images").exists():
            shutil.rmtree(outputs_dir / "qa_images")
        shutil.copytree(stage2_dir / "outputs" / "qa_images", outputs_dir / "qa_images")
    if qa_index_path.exists():
        _filter_qa_assets(qa_index_path, outputs_dir, per_drive_frames)

    report_path = outputs_dir / "crosswalk_full_report.md"
    _write_report(report_path, stage1_csv, final_gdf, reject_samples)

    log.info("done: %s", outputs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

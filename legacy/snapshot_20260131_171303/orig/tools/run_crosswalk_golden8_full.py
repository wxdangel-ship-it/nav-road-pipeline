from __future__ import annotations

import argparse
import csv
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
import pandas as pd
import yaml


LOG = logging.getLogger("run_crosswalk_golden8_full")


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("run_crosswalk_golden8_full")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_golden_drives(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _expand_drives(drives: Iterable[str]) -> List[str]:
    expanded: List[str] = []
    for token in drives:
        key = str(token).strip().lower()
        if key in {"golden8", "golden"}:
            expanded.extend(_load_golden_drives(Path("configs") / "golden_drives.txt"))
        elif token:
            expanded.append(str(token))
    return sorted(set(expanded))


def _run_cmd(cmd: List[str]) -> int:
    LOG.info("run: %s", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def _parse_support_frames(value: object) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [p for p in text.split("|") if p]
    return parts


def _select_qa_frames(
    outputs_dir: Path,
    drive_id: str,
    topk_support: int = 10,
    topk_low_iou: int = 10,
    topk_near_final: int = 20,
) -> List[str]:
    cluster_path = outputs_dir / "cluster_summary.csv"
    roundtrip_path = outputs_dir / "roundtrip_metrics.csv"
    if not cluster_path.exists():
        return []
    cluster = pd.read_csv(cluster_path)
    final_rows = cluster[cluster.get("final_pass", 0).astype(int) == 1]
    support_frames: List[str] = []
    for _, row in final_rows.iterrows():
        support_frames.extend(_parse_support_frames(row.get("support_frames")))
    support_frames = support_frames[:topk_support] if support_frames else []

    low_iou_frames: List[str] = []
    if support_frames and roundtrip_path.exists():
        roundtrip = pd.read_csv(roundtrip_path)
        roundtrip["frame_id"] = roundtrip["frame_id"].astype(str).str.zfill(10)
        subset = roundtrip[roundtrip["frame_id"].isin(support_frames)]
        subset = subset.sort_values(by="reproj_iou_bbox", ascending=True)
        low_iou_frames = subset["frame_id"].head(topk_low_iou).tolist()

    near_final_frames: List[str] = []
    near_rows = cluster[
        (cluster.get("final_pass", 0).astype(int) == 0)
        & (cluster.get("frames_hit_support", 0).astype(int) >= 2)
    ]
    for _, row in near_rows.iterrows():
        near_final_frames.extend(_parse_support_frames(row.get("support_frames")))
        if len(near_final_frames) >= topk_near_final:
            break
    near_final_frames = near_final_frames[:topk_near_final]

    selected = []
    for frame_id in support_frames + low_iou_frames + near_final_frames:
        if frame_id not in selected:
            selected.append(frame_id)
    return selected


def _prune_qa(outputs_dir: Path, drive_id: str, keep_frames: List[str]) -> None:
    qa_dir = outputs_dir / "qa_images" / drive_id
    qa_index_path = outputs_dir / "qa_index_wgs84.geojson"
    if keep_frames and qa_index_path.exists():
        try:
            qa = gpd.read_file(qa_index_path)
        except Exception:
            qa = None
        if qa is not None and not qa.empty:
            qa["frame_id"] = qa["frame_id"].astype(str).str.zfill(10)
            qa = qa[qa["frame_id"].isin(keep_frames)]
            qa.to_file(qa_index_path, driver="GeoJSON")
    if qa_dir.exists() and keep_frames:
        keep_set = {str(fid).zfill(10) for fid in keep_frames}
        for path in qa_dir.glob("*.png"):
            token = path.name.split("_")[0]
            if token not in keep_set:
                path.unlink()


def _read_report_counts(report_path: Path) -> Tuple[int, int, int]:
    if not report_path.exists():
        return 0, 0, 0
    text = report_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    candidate = review = final = 0
    for line in text:
        if line.strip().startswith("- candidate_count_total:"):
            candidate = int(line.split(":")[-1].strip())
        if line.strip().startswith("- review_count:"):
            review = int(line.split(":")[-1].strip())
        if line.strip().startswith("- final_count:"):
            final = int(line.split(":")[-1].strip())
    return candidate, review, final


def _final_bounds(utm_path: Path) -> Tuple[str, str, str, str]:
    if not utm_path.exists():
        return "", "", "", ""
    try:
        final = gpd.read_file(utm_path, layer="crosswalk_poly")
    except Exception:
        return "", "", "", ""
    if final.empty:
        return "", "", "", ""
    minx, miny, maxx, maxy = final.total_bounds
    return f"{minx:.3f}", f"{miny:.3f}", f"{maxx:.3f}", f"{maxy:.3f}"


def _roundtrip_stats(roundtrip_path: Path) -> Tuple[str, str]:
    if not roundtrip_path.exists():
        return "", ""
    df = pd.read_csv(roundtrip_path)
    if df.empty:
        return "", ""
    subset = df[
        (df.get("raw_has_crosswalk", 0).astype(int) == 1)
        & (df.get("geom_ok", 0).astype(int) == 1)
        & (df.get("proj_method", "").astype(str) == "lidar")
    ]
    if subset.empty:
        return "", ""
    vals = subset["reproj_iou_bbox"].to_numpy(dtype=float)
    return f"{np.percentile(vals, 50):.6f}", f"{np.percentile(vals, 90):.6f}"


def _accum_ratio(trace_path: Path) -> str:
    if not trace_path.exists():
        return ""
    df = pd.read_csv(trace_path)
    if df.empty:
        return ""
    subset = df[(df.get("raw_has_crosswalk", 0).astype(int) == 1) & (df.get("proj_method", "") == "lidar")]
    if subset.empty:
        return "0.0"
    accum = subset.get("lidar_fit_source", "").astype(str) == "accum"
    ratio = float(accum.sum()) / float(len(subset))
    return f"{ratio:.3f}"


def _drift_ratio(trace_path: Path) -> str:
    if not trace_path.exists():
        return ""
    df = pd.read_csv(trace_path)
    if df.empty:
        return ""
    if "prop_drift_flag" not in df.columns:
        return "0.0"
    ratio = float((df["prop_drift_flag"].fillna(0).astype(int) == 1).sum()) / float(len(df))
    return f"{ratio:.3f}"


def _top_reject_reasons(trace_path: Path, topk: int = 5) -> str:
    if not trace_path.exists():
        return ""
    df = pd.read_csv(trace_path)
    if df.empty or "reject_reasons" not in df.columns:
        return ""
    counts: Dict[str, int] = {}
    for raw in df["reject_reasons"].fillna("").astype(str):
        for token in [t for t in raw.split(",") if t]:
            counts[token] = counts.get(token, 0) + 1
    if not counts:
        return ""
    sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:topk]
    return ";".join([f"{k}:{v}" for k, v in sorted_items])


def _write_drive_report(outputs_dir: Path, drive_id: str) -> None:
    src = outputs_dir / "crosswalk_stage2_report.md"
    dst = outputs_dir / "report.md"
    if src.exists():
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crosswalk_golden8_full.yaml")
    ap.add_argument("--out-run", default="")
    ap.add_argument("--drives", default="")
    ap.add_argument("--skip-existing", type=int, default=1)
    args = ap.parse_args()

    log = _setup_logger()
    cfg = _load_yaml(Path(args.config))
    drives_cfg = cfg.get("drives", [])
    drives_arg = [d for d in str(args.drives).split(",") if d.strip()]
    drives = _expand_drives(drives_arg if drives_arg else drives_cfg)
    if not drives:
        log.error("no drives configured")
        return 2

    out_run = Path(args.out_run) if args.out_run else Path("runs") / f"crosswalk_golden8_full_{dt.datetime.now():%Y%m%d_%H%M%S}"
    outputs_root = out_run / "outputs"
    summary_dir = out_run / "summary"
    drive_root = out_run / "drive_runs"
    outputs_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    drive_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    skip_existing = bool(args.skip_existing)
    for drive_id in drives:
        drive_run = drive_root / drive_id
        drive_out = outputs_root / drive_id
        if skip_existing and drive_out.exists():
            log.info("skip existing drive: %s", drive_id)
            continue
        cmd = [
            str(Path(".venv") / "Scripts" / "python.exe"),
            "tools/run_crosswalk_monitor_range.py",
            "--config",
            str(Path(args.config)),
            "--drive",
            drive_id,
            "--auto-frame-range",
            "1",
            "--out-run",
            str(drive_run),
        ]
        if cfg.get("kitti_root"):
            cmd.extend(["--kitti-root", str(cfg.get("kitti_root"))])
        if _run_cmd(cmd) != 0:
            log.error("drive failed: %s", drive_id)
            continue

        if drive_out.exists():
            shutil.rmtree(drive_out)
        shutil.copytree(drive_run / "outputs", drive_out)
        _write_drive_report(drive_out, drive_id)

        keep_frames = _select_qa_frames(drive_out, drive_id)
        _prune_qa(drive_out, drive_id, keep_frames)

        # summary rebuilt after loop

    for drive_dir in sorted([p for p in outputs_root.iterdir() if p.is_dir()]):
        report_path = drive_dir / "crosswalk_stage2_report.md"
        candidate_count, review_count, final_count = _read_report_counts(report_path)
        final_bbox = _final_bounds(drive_dir / "crosswalk_entities_utm32.gpkg")
        iou_p50, iou_p90 = _roundtrip_stats(drive_dir / "roundtrip_metrics.csv")
        accum_ratio = _accum_ratio(drive_dir / "crosswalk_trace.csv")
        drift_ratio = _drift_ratio(drive_dir / "crosswalk_trace.csv")
        reject_top = _top_reject_reasons(drive_dir / "crosswalk_trace.csv")

        summary_rows.append(
            {
                "drive_id": drive_dir.name,
                "final_count": final_count,
                "review_count": review_count,
                "candidate_count": candidate_count,
                "final_centroid_bbox": ",".join(final_bbox),
                "reproj_iou_bbox_p50": iou_p50,
                "reproj_iou_bbox_p90": iou_p90,
                "lidar_fit_source_ratio": accum_ratio,
                "drift_ratio": drift_ratio,
                "top_reject_reasons": reject_top,
            }
        )

    overview_path = summary_dir / "golden8_overview.csv"
    with overview_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "drive_id",
                "final_count",
                "review_count",
                "candidate_count",
                "final_centroid_bbox",
                "reproj_iou_bbox_p50",
                "reproj_iou_bbox_p90",
                "lidar_fit_source_ratio",
                "drift_ratio",
                "top_reject_reasons",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    report_lines = ["# Golden8 Crosswalk Report", f"- run_dir: {out_run}", ""]
    for row in summary_rows:
        report_lines.append(f"## {row['drive_id']}")
        report_lines.append(f"- final_count: {row['final_count']}")
        report_lines.append(f"- review_count: {row['review_count']}")
        report_lines.append(f"- candidate_count: {row['candidate_count']}")
        report_lines.append(f"- final_centroid_bbox: {row['final_centroid_bbox']}")
        report_lines.append(f"- reproj_iou_bbox_p50/p90: {row['reproj_iou_bbox_p50']} / {row['reproj_iou_bbox_p90']}")
        report_lines.append(f"- lidar_fit_source_ratio(accum): {row['lidar_fit_source_ratio']}")
        report_lines.append(f"- drift_ratio: {row['drift_ratio']}")
        report_lines.append(f"- top_reject_reasons: {row['top_reject_reasons']}")
        report_lines.append("")
    (summary_dir / "golden8_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    log.info("done: %s", out_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

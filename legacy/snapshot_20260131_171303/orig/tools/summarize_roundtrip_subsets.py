from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _normalize_frame_id(value: Any) -> str:
    try:
        return f"{int(str(value)):010d}"
    except Exception:
        return ""


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _metric_stats(values: pd.Series) -> Dict[str, Optional[float]]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return {"p10": None, "p50": None, "p90": None, "mean": None}
    return {
        "p10": float(vals.quantile(0.10)),
        "p50": float(vals.quantile(0.50)),
        "p90": float(vals.quantile(0.90)),
        "mean": float(vals.mean()),
    }


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    run_root = run_dir.parent
    candidates = [
        run_root / "debug" / "run_config.yaml",
        run_root / "debug" / "run_config.json",
        Path("configs/crosswalk_range_250_500_strict.yaml"),
    ]
    for cfg_path in candidates:
        if not cfg_path.exists():
            continue
        try:
            if cfg_path.suffix.lower() == ".json":
                return json.loads(cfg_path.read_text(encoding="utf-8")) or {}
            import yaml

            return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
    return {}


def _resolve_overlay_path(run_dir: Path, drive_id: str, frame_id: str, suffix: str) -> str:
    path = run_dir / "qa_images" / drive_id / f"{frame_id}_{suffix}.png"
    return str(path) if path.exists() else ""


def _resolve_proj_debug_dir(run_dir: Path, frame_id: str) -> str:
    path = run_dir / "proj_debug" / f"{frame_id}_lidar_on_image.png"
    if path.exists():
        return str(path.parent)
    return ""


def _build_subsets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    raw_has = pd.to_numeric(df.get("raw_has_crosswalk", 0), errors="coerce").fillna(0).astype(int)
    geom_ok = pd.to_numeric(df.get("geom_ok", 0), errors="coerce").fillna(0).astype(int)
    proj_method = df.get("proj_method_rt", df.get("proj_method", "")).astype(str)
    stage2_added = pd.to_numeric(df.get("stage2_added", 0), errors="coerce").fillna(0).astype(int)
    subsets = {
        "S0_ALL_FRAMES": df,
        "S1_VALID_ANY_GEOM": df[(raw_has == 1) & (geom_ok == 1)],
        "S2_VALID_LIDAR": df[(raw_has == 1) & (geom_ok == 1) & (proj_method == "lidar")],
    }
    if "stage2_added" in df.columns:
        subsets["S3_VALID_STAGE2_ADDED"] = df[(stage2_added == 1) & (geom_ok == 1)]
    return subsets


def _stats_rows(subsets: Dict[str, pd.DataFrame], total: int) -> List[Dict[str, Any]]:
    rows = []
    for name, subset in subsets.items():
        count = int(len(subset))
        ratio = float(count / total) if total else 0.0
        bbox_stats = _metric_stats(subset.get("reproj_iou_bbox", pd.Series(dtype=float)))
        dilated_stats = _metric_stats(subset.get("reproj_iou_dilated", pd.Series(dtype=float)))
        center_stats = _metric_stats(subset.get("reproj_center_err_px", pd.Series(dtype=float)))
        rows.append(
            {
                "subset": name,
                "count": count,
                "total": total,
                "ratio": ratio,
                "reproj_iou_bbox_p10": bbox_stats["p10"],
                "reproj_iou_bbox_p50": bbox_stats["p50"],
                "reproj_iou_bbox_p90": bbox_stats["p90"],
                "reproj_iou_bbox_mean": bbox_stats["mean"],
                "reproj_iou_dilated_p10": dilated_stats["p10"],
                "reproj_iou_dilated_p50": dilated_stats["p50"],
                "reproj_iou_dilated_p90": dilated_stats["p90"],
                "reproj_iou_dilated_mean": dilated_stats["mean"],
                "reproj_center_err_px_p50": center_stats["p50"],
                "reproj_center_err_px_p90": center_stats["p90"],
                "reproj_center_err_px_mean": center_stats["mean"],
            }
        )
    return rows


def _heuristic_fail_reason(row: pd.Series, iou_thr: float) -> str:
    reasons = []
    points_support = int(pd.to_numeric(row.get("points_support", 0), errors="coerce") or 0)
    proj_method = str(row.get("proj_method_rt", row.get("proj_method", "")) or "")
    iou_bbox = _as_float(row.get("reproj_iou_bbox")) or 0.0
    iou_dilated = _as_float(row.get("reproj_iou_dilated")) or 0.0
    if points_support == 0:
        reasons.append("SUPPORT_EMPTY")
    if proj_method != "lidar":
        reasons.append("NON_LIDAR")
    if iou_bbox == 0 and iou_dilated > 0:
        reasons.append("THIN_MASK_EFFECT")
    if iou_bbox == 0 and iou_dilated == 0:
        reasons.append("PROJ_OR_ROI_MISMATCH")
    if iou_bbox < iou_thr and not reasons:
        reasons.append("LOW_IOU")
    return "|".join(reasons)


def _make_failpack(
    run_dir: Path,
    frames: List[str],
    run_cfg: Dict[str, Any],
) -> Dict[str, str]:
    kitti_root = run_cfg.get("kitti_root")
    drive_id = run_cfg.get("drive_id")
    camera = run_cfg.get("camera", "image_00")
    lidar_world_mode = run_cfg.get("lidar_world_mode", "fullpose")
    image_run = run_cfg.get("image_run")
    image_provider = run_cfg.get("image_provider")
    if not (kitti_root and drive_id and image_run and image_provider):
        return {}
    out_root = run_dir / "proj_debug_failpack"
    out_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for frame_id in frames:
        out_dir = out_root / frame_id
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "tools/debug_projection_chain.py",
            "--kitti-root",
            str(kitti_root),
            "--drive",
            str(drive_id),
            "--frame",
            str(int(frame_id)),
            "--camera",
            str(camera),
            "--lidar-world-mode",
            str(lidar_world_mode),
            "--candidate-source",
            str(run_dir),
            "--image-run",
            str(image_run),
            "--image-provider",
            str(image_provider),
            "--out-dir",
            str(out_dir),
        ]
        subprocess.run(cmd, check=False)
        results[frame_id] = str(out_dir)
    return results


def _write_report(
    out_path: Path,
    run_dir: Path,
    stats_rows: List[Dict[str, Any]],
    fail_rows: List[Dict[str, Any]],
    iou_thr: float,
) -> None:
    lines = []
    lines.append("# projection_alignment_report_v2")
    lines.append(f"- run_dir: {run_dir}")
    lines.append("")
    lines.append("## Roundtrip Subset Stats")
    lines.append(
        "| subset | count | ratio | iou_bbox_p50 | iou_bbox_p90 | iou_dilated_p50 | iou_dilated_p90 | center_err_p50 | center_err_p90 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    stats_map = {row["subset"]: row for row in stats_rows}
    for row in stats_rows:
        lines.append(
            f"| {row['subset']} | {row['count']} | {row['ratio']:.3f} | {row.get('reproj_iou_bbox_p50')} | {row.get('reproj_iou_bbox_p90')} | {row.get('reproj_iou_dilated_p50')} | {row.get('reproj_iou_dilated_p90')} | {row.get('reproj_center_err_px_p50')} | {row.get('reproj_center_err_px_p90')} |"
        )
    lines.append("")
    lines.append("## Conclusion")
    s2 = stats_map.get("S2_VALID_LIDAR", {})
    s2_ratio = _as_float(s2.get("ratio")) or 0.0
    s2_p50 = _as_float(s2.get("reproj_iou_bbox_p50"))
    s2_p90 = _as_float(s2.get("reproj_iou_bbox_p90"))
    lines.append(
        f"- S2_VALID_LIDAR: count_ratio={s2_ratio:.3f} iou_bbox_p50={s2_p50} iou_bbox_p90={s2_p90}"
    )
    if s2_p50 is not None and s2_ratio >= 0.10 and s2_p50 >= iou_thr:
        lines.append("- Verdict: alignment likely OK; issue leans toward low valid-frame ratio or candidate quality.")
    else:
        lines.append("- Verdict: S2 p50 or ratio is low; likely projection/ROI/backfill chain issue.")
    lines.append("")
    lines.append("## Top Fail Frames (S2_VALID_LIDAR)")
    if not fail_rows:
        lines.append("- (none)")
    else:
        lines.append("| frame_id | iou_bbox | iou_dilated | overlay_gated | proj_debug | failpack |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in fail_rows:
            lines.append(
                f"| {row.get('frame_id','')} | {row.get('reproj_iou_bbox','')} | {row.get('reproj_iou_dilated','')} | {row.get('overlay_gated_path','')} | {row.get('proj_debug_dir','')} | {row.get('proj_debug_failpack_dir','')} |"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-md", default="projection_alignment_report_v2.md")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--iou-thr", type=float, default=0.05)
    ap.add_argument("--make-failpack", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run-dir not found: {run_dir}")

    trace_path = run_dir / "crosswalk_trace.csv"
    roundtrip_path = run_dir / "roundtrip_metrics.csv"
    if not trace_path.exists() or not roundtrip_path.exists():
        raise SystemExit("missing crosswalk_trace.csv or roundtrip_metrics.csv")

    trace_df = pd.read_csv(trace_path)
    roundtrip_df = pd.read_csv(roundtrip_path)
    trace_df["frame_id_norm"] = trace_df["frame_id"].apply(_normalize_frame_id)
    roundtrip_df["frame_id_norm"] = roundtrip_df["frame_id"].apply(_normalize_frame_id)
    merged = roundtrip_df.merge(trace_df, on="frame_id_norm", how="left", suffixes=("_rt", ""))
    total = len(merged)

    subsets = _build_subsets(merged)
    stats_rows = _stats_rows(subsets, total)
    stats_path = run_dir / "roundtrip_stats_subsets.csv"
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)

    fail_subset = subsets.get("S2_VALID_LIDAR", merged)
    fail_subset = fail_subset.copy()
    fail_subset["reproj_iou_bbox_val"] = pd.to_numeric(
        fail_subset.get("reproj_iou_bbox"), errors="coerce"
    ).fillna(0.0)
    fail_subset = fail_subset[fail_subset["reproj_iou_bbox_val"] < float(args.iou_thr)]
    fail_subset = fail_subset.sort_values("reproj_iou_bbox_val", ascending=True).head(int(args.topn))

    run_cfg = _load_run_config(run_dir)
    drive_id = str(run_cfg.get("drive_id") or (trace_df.get("drive_id").iloc[0] if not trace_df.empty else ""))
    if not drive_id:
        drive_id = "unknown_drive"

    fail_rows = []
    for _, row in fail_subset.iterrows():
        frame_id = _normalize_frame_id(row.get("frame_id_norm") or row.get("frame_id") or "")
        fail_rows.append(
            {
                "frame_id": frame_id,
                "proj_method": row.get("proj_method_rt", row.get("proj_method", "")),
                "raw_has_crosswalk": row.get("raw_has_crosswalk", 0),
                "raw_status": row.get("raw_status", ""),
                "geom_ok": row.get("geom_ok", 0),
                "points_in_bbox": row.get("points_in_bbox", 0),
                "points_in_mask": row.get("points_in_mask", 0),
                "points_support": row.get("points_support", 0),
                "points_support_accum": row.get("points_support_accum", 0),
                "reproj_iou_bbox": row.get("reproj_iou_bbox", 0),
                "reproj_iou_dilated": row.get("reproj_iou_dilated", 0),
                "overlay_raw_path": _resolve_overlay_path(run_dir, drive_id, frame_id, "overlay_raw"),
                "overlay_gated_path": _resolve_overlay_path(run_dir, drive_id, frame_id, "overlay_gated"),
                "overlay_entities_path": _resolve_overlay_path(run_dir, drive_id, frame_id, "overlay_entities"),
                "proj_debug_dir": _resolve_proj_debug_dir(run_dir, frame_id),
                "fail_reason_hint": _heuristic_fail_reason(row, float(args.iou_thr)),
            }
        )

    failpack_dirs = {}
    if args.make_failpack and fail_rows:
        frames = [row["frame_id"] for row in fail_rows]
        failpack_dirs = _make_failpack(run_dir, frames, run_cfg)

    for row in fail_rows:
        row["proj_debug_failpack_dir"] = failpack_dirs.get(row["frame_id"], "")

    fail_path = run_dir / "roundtrip_fail_frames_top20.csv"
    pd.DataFrame(fail_rows).to_csv(fail_path, index=False)

    out_md_path = run_dir / args.out_md
    _write_report(out_md_path, run_dir, stats_rows, fail_rows, float(args.iou_thr))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

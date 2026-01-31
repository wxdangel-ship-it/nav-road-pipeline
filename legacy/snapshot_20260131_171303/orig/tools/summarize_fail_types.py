from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


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
        return {"p10": None, "p50": None, "p90": None}
    return {
        "p10": float(vals.quantile(0.10)),
        "p50": float(vals.quantile(0.50)),
        "p90": float(vals.quantile(0.90)),
    }


def _pick_representatives(group: pd.DataFrame, topn: int) -> pd.DataFrame:
    if group.empty:
        return group
    group = group.copy()
    group["reproj_iou_bbox_val"] = pd.to_numeric(
        group.get("reproj_iou_bbox"), errors="coerce"
    ).fillna(0.0)
    group["points_support_val"] = pd.to_numeric(
        group.get("points_support"), errors="coerce"
    ).fillna(0.0)
    group = group.sort_values(
        ["reproj_iou_bbox_val", "points_support_val"],
        ascending=[True, False],
    )
    return group.head(topn)


def _render_overview_table(rows: List[Dict[str, Any]]) -> List[str]:
    lines = []
    lines.append("| fail_type | count | ratio | iou_bbox_p50 | iou_bbox_p90 | points_support_p50 | points_support_p90 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row['fail_type']} | {row['count']} | {row['ratio']:.3f} |"
            f" {row.get('iou_bbox_p50')} | {row.get('iou_bbox_p90')} |"
            f" {row.get('points_support_p50')} | {row.get('points_support_p90')} |"
        )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--topn-per-type", type=int, default=5)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    fail_path = run_dir / "roundtrip_fail_frames_top20.csv"
    if not fail_path.exists():
        raise SystemExit(f"missing {fail_path}")

    df = pd.read_csv(fail_path)
    if "fail_reason_hint" not in df.columns:
        df["fail_reason_hint"] = "UNKNOWN"
    df["fail_reason_hint"] = df["fail_reason_hint"].fillna("UNKNOWN").astype(str)

    total = len(df)
    rows = []
    detail_blocks = []
    for fail_type, group in df.groupby("fail_reason_hint", dropna=False):
        count = int(len(group))
        ratio = float(count / total) if total else 0.0
        iou_stats = _metric_stats(group.get("reproj_iou_bbox", pd.Series(dtype=float)))
        iou_d_stats = _metric_stats(group.get("reproj_iou_dilated", pd.Series(dtype=float)))
        support_stats = _metric_stats(group.get("points_support", pd.Series(dtype=float)))
        support_accum_stats = _metric_stats(
            group.get("points_support_accum", pd.Series(dtype=float))
        )
        center_stats = _metric_stats(group.get("reproj_center_err_px", pd.Series(dtype=float)))
        reps = _pick_representatives(group, int(args.topn_per_type))
        rep_frames = ";".join(reps.get("frame_id", "").astype(str).tolist())
        rows.append(
            {
                "fail_type": fail_type,
                "count": count,
                "ratio": ratio,
                "iou_bbox_p10": iou_stats["p10"],
                "iou_bbox_p50": iou_stats["p50"],
                "iou_bbox_p90": iou_stats["p90"],
                "iou_dilated_p10": iou_d_stats["p10"],
                "iou_dilated_p50": iou_d_stats["p50"],
                "iou_dilated_p90": iou_d_stats["p90"],
                "points_support_p50": support_stats["p50"],
                "points_support_p90": support_stats["p90"],
                "points_support_accum_p50": support_accum_stats["p50"],
                "points_support_accum_p90": support_accum_stats["p90"],
                "center_err_p50": center_stats["p50"],
                "center_err_p90": center_stats["p90"],
                "rep_frames": rep_frames,
            }
        )
        detail_blocks.append((fail_type, reps))

    rows.sort(key=lambda r: r["count"], reverse=True)
    out_csv = run_dir / "fail_type_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    md_lines = []
    md_lines.append("# fail_type_summary")
    md_lines.append(f"- run_dir: {run_dir}")
    md_lines.append("")
    md_lines.append("## Overview")
    md_lines.extend(_render_overview_table(rows))
    md_lines.append("")

    for fail_type, reps in detail_blocks:
        md_lines.append(f"## {fail_type}")
        md_lines.append("- Notes: grouped by fail_reason_hint; metrics from top20 fail frames.")
        if reps.empty:
            md_lines.append("- No representative frames.")
            md_lines.append("")
            continue
        md_lines.append("| frame_id | iou_bbox | iou_dilated | points_support | points_support_accum | gated_overlay |")
        md_lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, row in reps.iterrows():
            frame_id = str(row.get("frame_id", ""))
            gated = row.get("overlay_gated_path", "")
            md_lines.append(
                f"| {frame_id} | {row.get('reproj_iou_bbox', '')} | {row.get('reproj_iou_dilated', '')} |"
                f" {row.get('points_support', '')} | {row.get('points_support_accum', '')} | {gated} |"
            )
        md_lines.append("")

    out_md = run_dir / "fail_type_summary.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

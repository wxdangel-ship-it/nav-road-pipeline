from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def _as_str(value: Any) -> str:
    return "" if value is None else str(value)


def _normalize_frame_id(value: Any) -> str:
    try:
        return f"{int(str(value)):010d}"
    except Exception:
        return _as_str(value)


def _rel_or_not_found(base: Path, path: Path) -> str:
    if path.exists():
        try:
            return str(path.relative_to(base))
        except Exception:
            return str(path)
    return "NOT_FOUND"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--format", default="md")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    fail_path = run_dir / "roundtrip_fail_frames_top20.csv"
    if not fail_path.exists():
        raise SystemExit(f"missing {fail_path}")

    df = pd.read_csv(fail_path)
    out_dir = run_dir / "proj_debug_failpack"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in df.iterrows():
        frame_id = _normalize_frame_id(row.get("frame_id"))
        frame_dir = out_dir / frame_id
        rows.append(
            {
                "frame_id": frame_id,
                "fail_reason_hint": _as_str(row.get("fail_reason_hint")),
                "reproj_iou_bbox": row.get("reproj_iou_bbox", ""),
                "reproj_iou_dilated": row.get("reproj_iou_dilated", ""),
                "points_support": row.get("points_support", ""),
                "points_support_accum": row.get("points_support_accum", ""),
                "proj_method": _as_str(row.get("proj_method")),
                "lidar_on_image": _rel_or_not_found(
                    out_dir, frame_dir / f"{frame_id}_lidar_on_image.png"
                ),
                "mask_and_lidar": _rel_or_not_found(
                    out_dir, frame_dir / f"{frame_id}_mask_and_lidar.png"
                ),
                "reproj_vs_mask": _rel_or_not_found(
                    out_dir, frame_dir / f"{frame_id}_reproj_vs_mask.png"
                ),
            }
        )

    out_path = out_dir / ("index.md" if args.format == "md" else "index.html")
    if args.format == "html":
        lines = []
        lines.append("<html><body>")
        lines.append("<h1>proj_debug_failpack index</h1>")
        lines.append("<table border='1'>")
        lines.append(
            "<tr><th>frame_id</th><th>fail_reason_hint</th><th>iou_bbox</th><th>iou_dilated</th>"
            "<th>points_support</th><th>points_support_accum</th><th>proj_method</th>"
            "<th>lidar_on_image</th><th>mask_and_lidar</th><th>reproj_vs_mask</th></tr>"
        )
        for row in rows:
            lines.append(
                "<tr>"
                f"<td>{row['frame_id']}</td>"
                f"<td>{row['fail_reason_hint']}</td>"
                f"<td>{row['reproj_iou_bbox']}</td>"
                f"<td>{row['reproj_iou_dilated']}</td>"
                f"<td>{row['points_support']}</td>"
                f"<td>{row['points_support_accum']}</td>"
                f"<td>{row['proj_method']}</td>"
                f"<td>{row['lidar_on_image']}</td>"
                f"<td>{row['mask_and_lidar']}</td>"
                f"<td>{row['reproj_vs_mask']}</td>"
                "</tr>"
            )
        lines.append("</table></body></html>")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 0

    lines = []
    lines.append("# proj_debug_failpack index")
    lines.append("")
    lines.append(
        "| frame_id | fail_reason_hint | iou_bbox | iou_dilated | points_support | points_support_accum | proj_method | lidar_on_image | mask_and_lidar | reproj_vs_mask |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for row in rows:
        lines.append(
            f"| {row['frame_id']} | {row['fail_reason_hint']} | {row['reproj_iou_bbox']} |"
            f" {row['reproj_iou_dilated']} | {row['points_support']} | {row['points_support_accum']} |"
            f" {row['proj_method']} | {row['lidar_on_image']} | {row['mask_and_lidar']} | {row['reproj_vs_mask']} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

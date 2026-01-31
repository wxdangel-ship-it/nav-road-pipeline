from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from scripts.pipeline_common import write_csv, write_json, write_text


def write_drive_report(path: Path, drive_id: str, stats: Dict[str, object], warnings: Iterable[str]) -> None:
    lines: List[str] = [
        "# LiDAR Semantic Report",
        "",
        f"- drive_id: {drive_id}",
        f"- status: {stats.get('status', 'ok')}",
    ]
    for k in [
        "roi_source",
        "frame_count",
        "point_count",
        "road_cover",
        "marking_cover_on_road",
        "marking_points_ratio",
        "crosswalk_count",
    ]:
        if k in stats:
            lines.append(f"- {k}: {stats.get(k)}")
    lines.append("")
    lines.append("## Outputs")
    for k in [
        "roi_path",
        "road_surface_path",
        "markings_path",
        "crosswalk_path",
        "semantic_points_path",
        "road_points_path",
        "markings_points_path",
    ]:
        v = stats.get(k)
        if v:
            lines.append(f"- {v}")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.extend([f"- {w}" for w in warnings])
    write_text(path, "\n".join(lines) + "\n")


def write_run_summary(
    run_dir: Path,
    summary: Dict[str, object],
    per_drive_rows: List[Dict[str, object]],
) -> None:
    write_json(run_dir / "run_summary.json", summary)
    md_lines = [
        "# LiDAR Semantic Golden8 Summary",
        "",
        f"- run_id: {summary.get('run_id')}",
        f"- data_root: {summary.get('data_root')}",
        f"- drives_total: {summary.get('drives_total')}",
        f"- drives_ok: {summary.get('drives_ok')}",
        f"- drives_fail: {summary.get('drives_fail')}",
        "",
        "## Drive Status",
    ]
    for row in per_drive_rows:
        md_lines.append(
            f"- {row.get('drive_id')}: status={row.get('status')} road_cover={row.get('road_cover')} markings_ratio={row.get('marking_points_ratio')} crosswalks={row.get('crosswalk_count')}"
        )
    write_text(run_dir / "run_summary.md", "\n".join(md_lines) + "\n")


def write_qa_index(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        write_csv(path, [], ["drive_id"])
        return
    write_csv(path, rows, list(rows[0].keys()))


__all__ = ["write_drive_report", "write_run_summary", "write_qa_index"]


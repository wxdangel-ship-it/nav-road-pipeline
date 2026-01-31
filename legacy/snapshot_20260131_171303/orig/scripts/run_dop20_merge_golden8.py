from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd

from scripts.pipeline_common import (
    ensure_dir,
    ensure_overwrite,
    now_ts,
    setup_logging,
    validate_output_crs,
    write_csv,
    write_json,
    write_text,
    write_gpkg_layer,
)


@dataclass
class Dop20Run:
    run_dir: Path
    run_id: str
    drives: Dict[str, Dict]


def _load_golden8() -> List[str]:
    path = Path("configs/golden_drives.txt")
    if not path.exists():
        return []
    drives = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return drives


def _read_summary(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _is_run_ok(run_dir: Path, drives: List[str]) -> Optional[Dop20Run]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return None
    summary = _read_summary(summary_path)
    if not summary:
        return None
    run_drives = summary.get("drives")
    if not isinstance(run_drives, dict):
        return None
    for d in drives:
        info = run_drives.get(d)
        if not isinstance(info, dict):
            return None
        if str(info.get("status") or "") != "ok":
            return None
        cand_path = run_dir / "drives" / d / "candidates" / "dop20_candidates_utm32.gpkg"
        if not cand_path.exists():
            return None
    return Dop20Run(run_dir=run_dir, run_id=str(summary.get("run_id") or run_dir.name), drives=run_drives)


def _pick_latest_ok(drives: List[str]) -> Optional[Dop20Run]:
    runs_root = Path("runs")
    candidates: List[Dop20Run] = []
    for run_dir in sorted(runs_root.glob("dop20*")):
        if not run_dir.is_dir():
            continue
        run = _is_run_ok(run_dir, drives)
        if run:
            candidates.append(run)
    if not candidates:
        return None
    # Run names are timestamped; lexical order is a stable proxy for recency.
    candidates.sort(key=lambda r: r.run_dir.name)
    return candidates[-1]


def main() -> int:
    drives = _load_golden8()
    run = _pick_latest_ok(drives) if drives else None
    if run is None:
        out_dir = ensure_dir(Path("runs") / f"dop20_merge_golden8_{now_ts()}")
        setup_logging(out_dir / "run.log")
        write_text(out_dir / "run_summary.md", "# DOP20 Merge Golden8\n\n- status: fail\n- reason: no_complete_dop20_run_found\n")
        write_json(out_dir / "run_summary.json", {"status": "fail", "reason": "no_complete_dop20_run_found"})
        return 2

    merged_dir = ensure_dir(run.run_dir / "merged")
    setup_logging(run.run_dir / "run.log")

    warnings: List[str] = []
    index_rows: List[Dict[str, object]] = []
    cand_frames: List[gpd.GeoDataFrame] = []
    roi_frames: List[gpd.GeoDataFrame] = []

    for d in drives:
        cand_path = run.run_dir / "drives" / d / "candidates" / "dop20_candidates_utm32.gpkg"
        roi_path = run.run_dir / "drives" / d / "roi" / "roi_buffer100_utm32.gpkg"
        qa_path = run.run_dir / "drives" / d / "qa" / "qa_index.csv"
        cand_exists = cand_path.exists()
        roi_exists = roi_path.exists()
        qa_exists = qa_path.exists()

        if cand_exists:
            gdf = gpd.read_file(cand_path, layer="world_candidates")
            if gdf.crs is None or gdf.crs.to_epsg() != 32632:
                gdf = gdf.set_crs(32632, allow_override=True)
            cand_frames.append(gdf)
        if roi_exists:
            rgdf = gpd.read_file(roi_path)
            if rgdf.crs is None or rgdf.crs.to_epsg() != 32632:
                rgdf = rgdf.set_crs(32632, allow_override=True)
            roi_frames.append(rgdf)

        index_rows.append(
            {
                "drive_id": d,
                "candidates_path": str(cand_path) if cand_exists else "",
                "roi_path": str(roi_path) if roi_exists else "",
                "qa_index_path": str(qa_path) if qa_exists else "",
                "status": run.drives.get(d, {}).get("status", ""),
            }
        )

    candidates_out = merged_dir / "dop20_candidates_utm32.gpkg"
    if cand_frames:
        merged = gpd.GeoDataFrame(pd.concat(cand_frames, ignore_index=True), crs="EPSG:32632")
        validate_output_crs(candidates_out, 32632, merged, warnings)
        write_gpkg_layer(candidates_out, "world_candidates", merged, 32632, warnings, overwrite=True)

    roi_out = merged_dir / "dop20_roi_utm32.gpkg"
    if roi_frames:
        merged_roi = gpd.GeoDataFrame(pd.concat(roi_frames, ignore_index=True), crs="EPSG:32632")
        validate_output_crs(roi_out, 32632, merged_roi, warnings)
        write_gpkg_layer(roi_out, "roi", merged_roi, 32632, warnings, overwrite=True)

    index_out = merged_dir / "dop20_index.csv"
    write_csv(index_out, index_rows, list(index_rows[0].keys()) if index_rows else ["drive_id"])

    summary = {
        "status": "ok",
        "run_dir": str(run.run_dir),
        "merged_dir": str(merged_dir),
        "drives": drives,
        "warnings": warnings,
    }
    write_json(merged_dir / "dop20_merge_summary.json", summary)
    lines = [
        "# DOP20 Merge Golden8",
        "",
        f"- source_run: {run.run_dir}",
        f"- merged_candidates: {candidates_out}",
        f"- merged_roi: {roi_out if roi_frames else 'n/a'}",
        f"- index_csv: {index_out}",
    ]
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.extend([f"- {w}" for w in warnings])
    write_text(merged_dir / "dop20_merge_summary.md", "\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


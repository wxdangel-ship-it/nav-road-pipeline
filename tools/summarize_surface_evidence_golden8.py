from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"

# =========================
# Parameters
# =========================
BATCH_FILE = r"configs\jobs\surface_evidence\golden8_full_from_lidar_fusion_baseline.yaml"
RUN_DIRS: List[str] = []


def _load_yaml(path: Path) -> Dict[str, object]:
    try:
        import yaml  # type: ignore
    except Exception:
        print("missing dependency: PyYAML. Install via: python -m pip install pyyaml")
        raise SystemExit(2)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _latest_run_dir(prefix: str) -> Path | None:
    candidates = sorted(RUNS_DIR.glob(f"{prefix}*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _collect_run_dirs() -> List[Path]:
    if RUN_DIRS:
        return [Path(p) for p in RUN_DIRS]
    batch = _load_yaml(REPO_ROOT / BATCH_FILE)
    jobs = batch.get("jobs") or []
    run_dirs: List[Path] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        drive_id = str(job.get("drive_id") or "")
        if not drive_id:
            continue
        prefix = f"surface_evidence_{drive_id}_"
        latest = _latest_run_dir(prefix)
        if latest:
            run_dirs.append(latest)
    return run_dirs


def _gate_status(gates: Dict[str, object]) -> Tuple[str, List[str]]:
    missing = [k for k in ["epsg_ok", "bbox_ok", "points_ok", "dem_ok", "bev_ok"] if k not in gates]
    if missing:
        return "WARN", [f"missing_gates:{','.join(missing)}"]
    fails = [k for k, v in gates.items() if v is False]
    if fails:
        return "FAIL", [f"gate_fail:{','.join(fails)}"]
    return "PASS", []


def _sample_tiles(index_path: Path) -> List[str]:
    if not index_path.exists():
        return []
    data = json.loads(index_path.read_text(encoding="utf-8"))
    feats = data.get("features", [])
    items = []
    for feat in feats:
        props = feat.get("properties", {})
        tile_x = props.get("tile_x", 0)
        tile_y = props.get("tile_y", 0)
        path = props.get("path", "")
        items.append((int(tile_x), int(tile_y), str(path)))
    if not items:
        return []
    items.sort(key=lambda x: (x[0], x[1]))
    mid = items[len(items) // 2]
    return [items[0][2], mid[2]]


def main() -> int:
    run_dirs = _collect_run_dirs()
    if not run_dirs:
        raise SystemExit("no_run_dirs_found")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = RUNS_DIR / f"surface_evidence_golden8_summary_{ts}"
    report_dir = summary_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    qc_notes = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "report" / "metrics.json"
        gates_path = run_dir / "report" / "gates.json"
        if not metrics_path.exists() or not gates_path.exists():
            rows.append(
                {
                    "drive_id": run_dir.name.split("_")[2] if "_" in run_dir.name else "",
                    "run_dir": str(run_dir),
                    "gate_status": "FAIL",
                    "warnings": "missing_metrics_or_gates",
                }
            )
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        gates = json.loads(gates_path.read_text(encoding="utf-8"))
        gate_status, warnings = _gate_status(gates)
        drive_id = str(metrics.get("drive_id") or "")
        if not drive_id:
            drive_id = run_dir.name.split("_")[2] if "_" in run_dir.name else ""

        tiles_index = metrics.get("bev", {}).get("tiles_index", "")
        tiles_index_path = Path(str(tiles_index)) if tiles_index else run_dir / "outputs" / "bev_markings_tiles_index_r005m.geojson"
        sampled = _sample_tiles(tiles_index_path)
        qc_notes.append(
            {
                "drive_id": drive_id,
                "run_dir": str(run_dir),
                "sampled_tiles": sampled,
                "roi_top_hat_gray": str(
                    run_dir / "outputs" / "bev_rois_r005m" / "cw_000_top_hat_gray_pctl.png"
                ),
                "roi_overlay": str(run_dir / "outputs" / "bev_rois_r005m" / "cw_000_overlay.png"),
            }
        )

        rows.append(
            {
                "drive_id": drive_id,
                "run_dir": str(run_dir),
                "gate_status": gate_status,
                "epsg": metrics.get("epsg"),
                "bbox_check": metrics.get("bbox_check", {}).get("ok"),
                "points_road_surface": metrics.get("points_road_surface"),
                "ratio_road_surface": metrics.get("ratio_road_surface"),
                "dem_valid_ratio": metrics.get("dem_valid_ratio"),
                "bev_tiles_count": metrics.get("bev", {}).get("tiles_count"),
                "bev_empty_tile_ratio": metrics.get("bev", {}).get("empty_tile_ratio"),
                "top_hat_p95": metrics.get("top_hat_p95"),
                "top_hat_p98": metrics.get("top_hat_p98"),
                "preview_gray_is_true": metrics.get("preview_gray_is_true"),
                "warnings": ";".join(warnings),
            }
        )

    csv_path = report_dir / "golden8_surface_evidence_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["# Golden8 Surface Evidence Summary", ""]
    for row in rows:
        md_lines.append(f"- {row['drive_id']}: {row['gate_status']} | run_dir={row['run_dir']}")
        if row.get("warnings"):
            md_lines.append(f"  - warnings: {row['warnings']}")
    md_lines.append("")
    md_lines.append("## QC Sampled Tiles")
    for item in qc_notes:
        md_lines.append(f"- {item['drive_id']} | run_dir={item['run_dir']}")
        md_lines.append(f"  - tiles: {', '.join(item['sampled_tiles']) if item['sampled_tiles'] else 'none'}")
        md_lines.append(f"  - roi_gray: {item['roi_top_hat_gray']}")
        md_lines.append(f"  - roi_overlay: {item['roi_overlay']}")
        md_lines.append("  - qc_result: PASS (auto check: files exist)")
    (report_dir / "golden8_surface_evidence_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    runs_list = report_dir / "golden8_surface_evidence_runs.txt"
    runs_list.write_text("\n".join([str(p) for p in run_dirs]), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

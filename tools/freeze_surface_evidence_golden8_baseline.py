from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]

# =========================
# Parameters
# =========================
RUNS_LIST_FILE = r""
DST_BASE = r"E:\KITTI360\KITTI-360"
OVERWRITE = False


def _load_runs() -> List[Path]:
    if not RUNS_LIST_FILE:
        raise SystemExit("RUNS_LIST_FILE not set")
    path = Path(RUNS_LIST_FILE)
    if not path.exists():
        raise SystemExit(f"runs_list_missing:{path}")
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _collect_large_files(run_dir: Path) -> List[Path]:
    outputs = run_dir / "outputs"
    large = []
    for ext in ("*.laz", "*.tif", "*.gpkg"):
        large.extend(list(outputs.rglob(ext)))
    return large


def _build_manifest(paths: List[Path], base: Path) -> List[Dict[str, object]]:
    items = []
    from tools.manifest_hash_head import hash_head

    for p in paths:
        stat = p.stat()
        rel = p.relative_to(base)
        items.append(
            {
                "rel_path": str(rel),
                "size": stat.st_size,
                "sha256_head": hash_head(p),
            }
        )
    return items


def _copy_small_files(run_dir: Path, baseline_dir: Path) -> None:
    picks = [
        run_dir / "report" / "metrics.json",
        run_dir / "report" / "gates.json",
        run_dir / "report" / "params.json",
        run_dir / "outputs" / "road_surface_polygon_preview.geojson",
        run_dir / "outputs" / "surface_dem_preview.png",
        run_dir / "outputs" / "large_files_manifest.json",
        run_dir / "outputs" / "bev_markings_tiles_index_r005m.geojson",
        run_dir / "logs" / "run_tail.log",
    ]
    roi_dir = run_dir / "outputs" / "bev_rois_r005m"
    if roi_dir.exists():
        for p in sorted(roi_dir.glob("cw_000_*png"))[:5]:
            picks.append(p)
    for p in picks:
        if not p.exists():
            continue
        rel = p.relative_to(run_dir)
        _copy_file(p, baseline_dir / run_dir.name / rel)


def main() -> int:
    runs = _load_runs()
    if not runs:
        raise SystemExit("no_runs")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_name = f"surface_evidence_golden8_utm32_{ts}"
    baseline_dir = REPO_ROOT / "baselines" / baseline_name
    artifacts_base = Path(DST_BASE) / baseline_name

    if baseline_dir.exists() and OVERWRITE:
        shutil.rmtree(baseline_dir, ignore_errors=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    artifacts_base.mkdir(parents=True, exist_ok=True)

    all_large_copied: List[Path] = []
    for run_dir in runs:
        if not run_dir.exists():
            raise SystemExit(f"run_dir_missing:{run_dir}")
        _copy_small_files(run_dir, baseline_dir)
        large_files = _collect_large_files(run_dir)
        for src in large_files:
            rel = Path(run_dir.name) / src.relative_to(run_dir)
            dst = artifacts_base / rel
            _copy_file(src, dst)
            all_large_copied.append(dst)

    manifest = _build_manifest(all_large_copied, artifacts_base)
    outputs_dir = baseline_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "large_files_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    artifact_location = {
        "skill_id": "surface_evidence_utm32",
        "artifacts_base_abs": str(artifacts_base),
        "layout": "manifest_rel_path_under_base",
        "migrated_from_abs": str(REPO_ROOT / "runs"),
        "migrated_at": datetime.now().isoformat(timespec="seconds"),
        "notes": "Golden8 surface evidence baseline frozen",
    }
    (baseline_dir / "artifact_location.json").write_text(
        json.dumps(artifact_location, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (baseline_dir / "DONE.ok").write_text("OK", encoding="utf-8")

    active = REPO_ROOT / "baselines" / "ACTIVE_SURFACE_EVIDENCE_BASELINE.txt"
    active.write_text(str(baseline_dir), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from tools.manifest_hash_head import hash_head

REPO_ROOT = Path(__file__).resolve().parents[1]

# =========================
# Parameters
# =========================
RUN_DIR = r""
BASELINE_NAME = r""
DST_ROOT = r"E:\KITTI360\KITTI-360"
OVERWRITE = False


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _collect_large_files(run_dir: Path) -> List[Path]:
    outputs = run_dir / "outputs"
    large = []
    for ext in ("*.laz", "*.tif", "*.gpkg"):
        large.extend(list(outputs.rglob(ext)))
    return large


def _write_manifest(paths: List[Path], base: Path, out_path: Path) -> None:
    items = []
    for p in paths:
        stat = p.stat()
        items.append(
            {
                "rel_path": str(p.relative_to(base)),
                "size": stat.st_size,
                "sha256_head": hash_head(p),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_small_files(run_dir: Path, baseline_dir: Path) -> None:
    picks = [
        run_dir / "report" / "metrics.json",
        run_dir / "report" / "gates.json",
        run_dir / "report" / "params.json",
        run_dir / "report" / "acceptance.md",
        run_dir / "outputs" / "road_surface_polygon_preview.geojson",
        run_dir / "outputs" / "surface_dem_preview.png",
        run_dir / "outputs" / "bev_markings_tiles_index_r005m.geojson",
        run_dir / "outputs" / "large_files_manifest.json",
        run_dir / "logs" / "run_tail.log",
    ]
    roi_dir = run_dir / "outputs" / "bev_rois_r005m"
    if roi_dir.exists():
        for p in sorted(roi_dir.glob("cw_000_*png"))[:6]:
            picks.append(p)
    tiles_dir = run_dir / "outputs" / "bev_markings_utm32_tiles_r005m"
    if tiles_dir.exists():
        for p in sorted(tiles_dir.glob("*_preview.png"))[:2]:
            picks.append(p)

    for p in picks:
        if not p.exists():
            continue
        rel = p.relative_to(run_dir)
        _copy_file(p, baseline_dir / rel)


def main() -> int:
    if not RUN_DIR:
        raise SystemExit("RUN_DIR not set")
    run_dir = Path(RUN_DIR)
    if not run_dir.exists():
        raise SystemExit(f"run_dir_missing:{run_dir}")

    ts = run_dir.name.split("_")[-1]
    name = BASELINE_NAME or f"surface_evidence_0010_part0_utm32_{ts}"
    baseline_dir = REPO_ROOT / "baselines" / name
    artifacts_base = Path(DST_ROOT) / name

    if baseline_dir.exists() and OVERWRITE:
        shutil.rmtree(baseline_dir, ignore_errors=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    artifacts_base.mkdir(parents=True, exist_ok=True)

    _copy_small_files(run_dir, baseline_dir)

    large_files = _collect_large_files(run_dir)
    copied = []
    for src in large_files:
        rel = src.relative_to(run_dir)
        dst = artifacts_base / rel
        _copy_file(src, dst)
        copied.append(dst)

    _write_manifest(copied, artifacts_base, baseline_dir / "outputs" / "large_files_manifest.json")

    artifact_location = {
        "skill_id": "surface_evidence_utm32",
        "artifacts_base_abs": str(artifacts_base),
        "layout": "manifest_rel_path_under_base",
        "migrated_from_abs": str(REPO_ROOT / "runs"),
        "migrated_at": datetime.now().isoformat(timespec="seconds"),
        "notes": "Surface evidence baseline (0010 part0) frozen",
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

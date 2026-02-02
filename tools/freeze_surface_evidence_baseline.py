from __future__ import annotations

"""
冻结 surface_evidence 基线：仅小证据入库，大文件仅记录 manifest/hash_head。
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List

from tools.manifest_hash_head import build_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]

# =========================
# 参数区（按需修改）
# =========================
RUN_DIR = r""
OVERWRITE = True


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def _collect_small_files(run_dir: Path, baseline_dir: Path) -> List[Path]:
    outputs = run_dir / "outputs"
    report = run_dir / "report"
    logs = run_dir / "logs"
    picks = [
        report / "metrics.json",
        report / "gates.json",
        report / "params.json",
        outputs / "road_surface_polygon_preview.geojson",
        outputs / "surface_dem_preview.png",
        outputs / "bev_markings_tiles_index_r010m.geojson",
        outputs / "large_files_manifest.json",
        logs / "run_tail.log",
    ]
    # optional ROI
    rois_index = outputs / "bev_rois_index.geojson"
    if rois_index.exists():
        picks.append(rois_index)
    roi_dir = outputs / "bev_rois_r005m"
    if roi_dir.exists():
        for p in sorted(roi_dir.glob("*_overlay.png"))[:3]:
            picks.append(p)

    copied = []
    for p in picks:
        if not p.exists():
            continue
        rel = p.relative_to(run_dir)
        dst = baseline_dir / rel
        _copy_file(p, dst)
        copied.append(dst)
    return copied


def main() -> int:
    if not RUN_DIR:
        raise SystemExit("RUN_DIR not set")
    run_dir = Path(RUN_DIR)
    if not run_dir.exists():
        raise SystemExit(f"run_dir_missing:{run_dir}")

    ts = run_dir.name.split("_")[-1]
    baseline_dir = REPO_ROOT / "baselines" / f"surface_evidence_0010_f000_300_{ts}"
    if baseline_dir.exists() and OVERWRITE:
        shutil.rmtree(baseline_dir, ignore_errors=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    _collect_small_files(run_dir, baseline_dir)

    # manifest for large outputs (from run_dir outputs)
    large_files = []
    outputs = run_dir / "outputs"
    for ext in ("*.laz", "*.tif", "*.gpkg"):
        large_files.extend(list(outputs.rglob(ext)))
    manifest = build_manifest(large_files)
    manifest_dir = baseline_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "large_files_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    active = REPO_ROOT / "baselines" / "ACTIVE_SURFACE_EVIDENCE_BASELINE.txt"
    active.write_text(str(baseline_dir), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

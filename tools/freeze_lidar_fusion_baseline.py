from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


# =========================
# 参数区（按需修改）
# =========================
REPO_ROOT = r"E:\Work\nav-road-pipeline"
RUN_DIR_FUSION_FULL_UTM32 = r"runs\lidar_fusion_0010_full_20260131_230942"
RUN_DIR_WORLD_TO_UTM32_FIT = r"runs\world_to_utm32_fit_0010_full_20260131_230902"
RUN_DIR_BANDING = r""  # optional if separate
OVERWRITE = True


def _run(cmd: List[str], cwd: Path) -> str:
    return subprocess.check_output(cmd, cwd=str(cwd), text=True, encoding="utf-8", errors="ignore")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_head(path: Path, n_bytes: int = 2 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    repo_root = Path(REPO_ROOT)
    run_dir = repo_root / RUN_DIR_FUSION_FULL_UTM32
    fit_dir = repo_root / RUN_DIR_WORLD_TO_UTM32_FIT
    band_dir = repo_root / RUN_DIR_BANDING if RUN_DIR_BANDING else run_dir

    if not run_dir.exists():
        raise SystemExit(f"missing run_dir: {run_dir}")
    if not fit_dir.exists():
        raise SystemExit(f"missing fit_dir: {fit_dir}")

    ts = Path(run_dir).name.split("_")[-1]
    baseline_dir = repo_root / "baselines" / f"lidar_fusion_0010_full_utm32_{ts}"
    if baseline_dir.exists() and OVERWRITE:
        shutil.rmtree(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # repo snapshot
    repo_dir = baseline_dir / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "branch.txt").write_text(_run(["git", "branch", "--show-current"], repo_root), encoding="utf-8")
    (repo_dir / "commit.txt").write_text(_run(["git", "rev-parse", "HEAD"], repo_root), encoding="utf-8")
    (repo_dir / "status.txt").write_text(_run(["git", "status", "--porcelain"], repo_root), encoding="utf-8")
    (repo_dir / "diff.patch").write_text(_run(["git", "diff"], repo_root), encoding="utf-8")
    (repo_dir / "log_20.txt").write_text(_run(["git", "log", "-n", "20", "--oneline"], repo_root), encoding="utf-8")

    # env snapshot
    env_dir = baseline_dir / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "python.txt").write_text(_run([sys.executable, "--version"], repo_root), encoding="utf-8")
    (env_dir / "pip_freeze.txt").write_text(_run([sys.executable, "-m", "pip", "freeze"], repo_root), encoding="utf-8")

    # evidence files
    evidence_dir = baseline_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_files = [
        fit_dir / "report" / "world_to_utm32_report.json",
        fit_dir / "report" / "world_to_utm32_summary.md",
        fit_dir / "report" / "world_to_utm32_pairs.csv",
        run_dir / "outputs" / "fused_points_utm32.meta.json",
        run_dir / "report" / "metrics.json",
        run_dir / "outputs" / "missing_frames.csv",
        run_dir / "outputs" / "missing_summary.json",
        run_dir / "outputs" / "bbox_utm32.geojson",
        run_dir / "outputs" / "fused_points_utm32_index.json",
        band_dir / "report" / "banding_audit_full.json",
        band_dir / "report" / "banding_summary_full.md",
        run_dir / "logs" / "run_tail.log",
    ]
    for src in evidence_files:
        if not src.exists():
            raise SystemExit(f"missing evidence file: {src}")
        _copy(src, evidence_dir / src.name)

    # manifests
    manifest_dir = baseline_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    small_manifest = []
    for f in sorted(evidence_dir.glob("*")):
        small_manifest.append(
            {"path": str(f), "size": f.stat().st_size, "sha256": _sha256(f)}
        )
    (manifest_dir / "small_files_manifest.json").write_text(
        json.dumps(small_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # laz parts manifest
    index_path = run_dir / "outputs" / "fused_points_utm32_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    laz_manifest = []
    for part in index.get("parts", []):
        p = Path(part["path"])
        size = p.stat().st_size if p.exists() else 0
        laz_manifest.append(
            {
                "path": str(p.resolve()),
                "size": size,
                "sha256_head2mb": _sha256_head(p) if p.exists() else "",
            }
        )
    (manifest_dir / "laz_parts_manifest.json").write_text(
        json.dumps(laz_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # report
    report_dir = baseline_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((run_dir / "report" / "metrics.json").read_text(encoding="utf-8"))
    banding = json.loads((band_dir / "report" / "banding_audit_full.json").read_text(encoding="utf-8"))
    fit = json.loads((fit_dir / "report" / "world_to_utm32_report.json").read_text(encoding="utf-8"))

    total_laz_size = sum(p["size"] for p in laz_manifest)
    report_lines = [
        "# Lidar Fusion Baseline (0010 full utm32)",
        "",
        f"- run_dir: {run_dir}",
        f"- gate_status: {fit.get('gate_status')}",
        f"- rms/yaw/scale: {fit.get('rms_m'):.3f} / {fit.get('yaw_deg'):.3f} / {fit.get('scale'):.6f}",
        f"- frames: {metrics.get('frames')}",
        f"- success/missing: {metrics.get('success_frames')} / {metrics.get('missing_frames')}",
        f"- points: {metrics.get('written_points')}",
        f"- intensity_max/nonzero: {metrics.get('intensity', {}).get('max')} / {metrics.get('intensity', {}).get('nonzero_ratio')}",
        f"- epsg: {metrics.get('epsg')}",
        f"- bbox_check: {metrics.get('bbox_check', {}).get('ok')}",
        f"- banding_check_pass: {banding.get('pass')}",
        f"- parts: {len(laz_manifest)} total_size_gb={total_laz_size / (1024**3):.3f}",
        "- human_qc: PASS",
    ]
    (report_dir / "baseline_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    # ACTIVE pointer
    active_path = repo_root / "baselines" / "ACTIVE_LIDAR_FUSION_BASELINE.txt"
    active_path.write_text(str(baseline_dir.resolve()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

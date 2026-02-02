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
RUN_DIRS = [
    r"runs\lidar_fusion_2013_05_28_drive_0000_sync_0_11517_20260201_193340",
    r"runs\lidar_fusion_2013_05_28_drive_0003_sync_0_1030_20260201_194445",
    r"runs\lidar_fusion_2013_05_28_drive_0004_sync_0_11586_20260201_194519",
    r"runs\lidar_fusion_2013_05_28_drive_0005_sync_0_6742_20260201_195629",
    r"runs\lidar_fusion_2013_05_28_drive_0006_sync_0_9698_20260201_200238",
    r"runs\lidar_fusion_2013_05_28_drive_0007_sync_0_3395_20260201_200750",
    r"runs\lidar_fusion_2013_05_28_drive_0009_sync_0_14055_20260201_200945",
    r"runs\lidar_fusion_2013_05_28_drive_0010_sync_0_3835_20260201_192348",
]
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


def _drive_id_from_run(run_dir: Path) -> str:
    # lidar_fusion_2013_05_28_drive_0007_sync_0_3395_20260201_200750
    name = run_dir.name
    parts = name.split("_")
    if "drive" in parts:
        i = parts.index("drive")
        if i + 2 < len(parts):
            return f"{parts[i+1]}_{parts[i+2]}"
    return name


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    repo_root = Path(REPO_ROOT)
    run_dirs = [repo_root / p for p in RUN_DIRS]
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise SystemExit(f"missing run_dir: {run_dir}")

    ts = run_dirs[0].name.split("_")[-1]
    baseline_dir = repo_root / "baselines" / f"lidar_fusion_golden8_utm32_{ts}"
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

    evidence_dir = baseline_dir / "evidence"
    manifest_dir = baseline_dir / "manifests"
    report_dir = baseline_dir / "report"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    small_manifest: List[Dict] = []
    laz_manifest: List[Dict] = []
    summary_rows: List[Dict] = []

    for run_dir in run_dirs:
        drive_id = _drive_id_from_run(run_dir)
        drive_evidence_dir = evidence_dir / drive_id
        drive_evidence_dir.mkdir(parents=True, exist_ok=True)

        required_files = [
            run_dir / "report" / "world_to_utm32_report.json",
            run_dir / "report" / "world_to_utm32_summary.md",
            run_dir / "report" / "world_to_utm32_pairs.csv",
            run_dir / "report" / "metrics.json",
            run_dir / "report" / "params.json",
            run_dir / "outputs" / "fused_points_utm32.meta.json",
            run_dir / "outputs" / "fused_points_utm32_index.json",
            run_dir / "outputs" / "missing_frames.csv",
            run_dir / "outputs" / "missing_summary.json",
            run_dir / "outputs" / "bbox_utm32.geojson",
            run_dir / "logs" / "run.log",
        ]
        optional_files = [
            run_dir / "report" / "gates.json",
            run_dir / "report" / "report.md",
            run_dir / "report" / "banding_audit_full.json",
            run_dir / "report" / "banding_summary_full.md",
            run_dir / "report" / "per_frame_points_sample.csv",
            run_dir / "logs" / "run_tail.log",
        ]
        for src in required_files:
            if not src.exists():
                raise SystemExit(f"missing evidence file: {src}")
            dst = drive_evidence_dir / src.name
            _copy(src, dst)
            small_manifest.append(
                {"path": str(dst), "size": dst.stat().st_size, "sha256": _sha256(dst)}
            )
        for src in optional_files:
            if not src.exists():
                continue
            dst = drive_evidence_dir / src.name
            _copy(src, dst)
            small_manifest.append(
                {"path": str(dst), "size": dst.stat().st_size, "sha256": _sha256(dst)}
            )

        # laz parts manifest
        index_path = run_dir / "outputs" / "fused_points_utm32_index.json"
        index = _load_json(index_path)
        for part in index.get("parts", []):
            p = Path(part["path"])
            laz_manifest.append(
                {
                    "drive_id": drive_id,
                    "path": str(p.resolve()),
                    "size": p.stat().st_size if p.exists() else 0,
                    "sha256_head2mb": _sha256_head(p) if p.exists() else "",
                }
            )

        metrics = _load_json(run_dir / "report" / "metrics.json")
        transform = metrics.get("transform", {})
        summary_rows.append(
            {
                "drive_id": drive_id,
                "run_dir": str(run_dir),
                "gate_status": transform.get("gate_status"),
                "rms_inlier_m": transform.get("rms_inlier_m"),
                "rms_raw_m": transform.get("rms_raw_m"),
                "p95_inlier_m": transform.get("p95_inlier_m"),
                "inlier_ratio": transform.get("inlier_ratio"),
                "crs_write_ok": metrics.get("crs_write_ok"),
            }
        )

    (manifest_dir / "small_files_manifest.json").write_text(
        json.dumps(small_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (manifest_dir / "laz_parts_manifest.json").write_text(
        json.dumps(laz_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # baseline report
    report_lines = [
        "# Lidar Fusion Baseline (Golden8 utm32)",
        "",
        f"- drives: {len(summary_rows)}",
        f"- baseline_dir: {baseline_dir}",
        "- gate_status: PASS/WARN only (no FAIL)",
        "",
        "## Per-drive",
    ]
    for row in summary_rows:
        report_lines.append(
            f"- {row['drive_id']}: gate={row['gate_status']} rms_inlier={row['rms_inlier_m']:.4f} "
            f"p95={row['p95_inlier_m']:.4f} inlier_ratio={row['inlier_ratio']:.4f} "
            f"crs_write_ok={row['crs_write_ok']}"
        )
    (report_dir / "baseline_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    # summary table to repo report/
    repo_report_dir = repo_root / "report"
    repo_report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = repo_report_dir / f"golden8_fusion_summary_{ts}.csv"
    md_path = repo_report_dir / f"golden8_fusion_summary_{ts}.md"
    header = [
        "drive_id",
        "run_dir",
        "gate_status",
        "rms_inlier_m",
        "rms_raw_m",
        "p95_inlier_m",
        "inlier_ratio",
        "crs_write_ok",
    ]
    lines = [",".join(header)]
    for row in summary_rows:
        lines.append(
            ",".join(
                [
                    row["drive_id"],
                    row["run_dir"],
                    str(row["gate_status"]),
                    f"{row['rms_inlier_m']:.6f}",
                    f"{row['rms_raw_m']:.6f}",
                    f"{row['p95_inlier_m']:.6f}",
                    f"{row['inlier_ratio']:.6f}",
                    str(row["crs_write_ok"]),
                ]
            )
        )
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    md_lines = ["# Golden8 Fusion Summary", "", "|drive_id|gate|rms_inlier|rms_raw|p95|inlier_ratio|crs_write_ok|", "|---|---|---:|---:|---:|---:|---|"]
    for row in summary_rows:
        md_lines.append(
            f"|{row['drive_id']}|{row['gate_status']}|{row['rms_inlier_m']:.4f}|{row['rms_raw_m']:.4f}|"
            f"{row['p95_inlier_m']:.4f}|{row['inlier_ratio']:.4f}|{row['crs_write_ok']}|"
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # ACTIVE pointer
    active_path = repo_root / "baselines" / "ACTIVE_LIDAR_FUSION_BASELINE.txt"
    active_path.write_text(str(baseline_dir.resolve()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

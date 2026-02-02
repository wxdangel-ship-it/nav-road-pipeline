from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


# =========================
# 参数区（按需修改）
# =========================
BASELINE_DIR = ""  # 若为空，自动读取 ACTIVE 指针
DST_BASE = r"E:\KITTI360\KITTI-360\lidar_fusion_golden8_utm32_193340"
MODE = "copy"  # copy | move
VERIFY = True
HEAD_BYTES = 1024 * 1024


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_head(path: Path, n_bytes: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()


def _common_base(paths: List[Path]) -> Path:
    common = os.path.commonpath([str(p) for p in paths])
    return Path(common)


def _read_manifest(baseline_dir: Path) -> Tuple[List[Dict], Path]:
    primary = baseline_dir / "outputs" / "large_files_manifest.json"
    if primary.exists():
        return _load_json(primary), primary
    legacy = baseline_dir / "manifests" / "laz_parts_manifest.json"
    if legacy.exists():
        return _load_json(legacy), legacy
    raise FileNotFoundError("baseline_large_files_manifest_missing")


def _normalize_entries(entries: List[Dict], src_base: Path) -> List[Dict]:
    out = []
    for item in entries:
        rel_path = item.get("rel_path")
        abs_path = item.get("path")
        sha256_head = item.get("sha256_head")
        if rel_path:
            src = src_base / rel_path
        elif abs_path:
            src = Path(str(abs_path))
            rel_path = os.path.relpath(str(src), str(src_base))
        else:
            raise ValueError("manifest_entry_missing_path")
        size_val = item.get("size")
        size = int(size_val) if size_val is not None else (src.stat().st_size if src.exists() else 0)
        out.append(
            {
                "rel_path": rel_path.replace("/", "\\"),
                "path": str(src),
                "size": size,
                "sha256_head": sha256_head,
            }
        )
    return out


def main() -> int:
    baseline_dir = Path(BASELINE_DIR) if BASELINE_DIR else None
    if not baseline_dir:
        pointer = REPO_ROOT / "baselines" / "ACTIVE_LIDAR_FUSION_BASELINE.txt"
        baseline_dir = Path(pointer.read_text(encoding="utf-8").strip())
    if not baseline_dir.exists():
        raise SystemExit(f"baseline_dir_missing:{baseline_dir}")

    entries, manifest_path = _read_manifest(baseline_dir)
    artifact_location = {}
    artifact_path = baseline_dir / "artifact_location.json"
    if artifact_path.exists():
        artifact_location = _load_json(artifact_path)
    src_paths = []
    for item in entries:
        p = item.get("path")
        if p:
            src_paths.append(Path(str(p)))
    if src_paths:
        src_base = _common_base(src_paths)
    else:
        migrated_from = artifact_location.get("migrated_from_abs", "")
        src_base = Path(migrated_from) if migrated_from else Path(DST_BASE)
    normalized = _normalize_entries(entries, src_base)

    dst_base = Path(DST_BASE)
    dst_base.mkdir(parents=True, exist_ok=True)

    total_size = sum(e["size"] for e in normalized)
    plan = {
        "baseline_dir": str(baseline_dir),
        "src_base": str(src_base),
        "dst_base": str(dst_base),
        "files_count": len(normalized),
        "total_size_est": total_size,
        "sample": normalized[:10],
    }
    report_dir = baseline_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "migration_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    results = {"success": [], "failed": []}
    for item in normalized:
        rel_path = item["rel_path"]
        src = Path(item["path"])
        dst = dst_base / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            results["failed"].append(
                {
                    "rel_path": rel_path,
                    "reason": "src_missing",
                    "src": str(src),
                    "dst": str(dst),
                }
            )
            continue

        if MODE == "move":
            shutil.move(str(src), str(dst))
        else:
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                pass
            else:
                shutil.copy2(str(src), str(dst))

        if VERIFY:
            expected_size = int(item["size"]) if item.get("size") else 0
            dst_size = dst.stat().st_size if dst.exists() else 0
            if expected_size:
                size_ok = dst_size == expected_size
            else:
                size_ok = dst_size == (src.stat().st_size if MODE == "copy" else dst_size)
            head_src = _sha256_head(src, HEAD_BYTES) if MODE == "copy" else None
            head_dst = _sha256_head(dst, HEAD_BYTES)
            manifest_head = item.get("sha256_head")
            if manifest_head:
                head_ok = manifest_head == head_dst
            else:
                head_ok = head_src == head_dst
            if size_ok and head_ok:
                results["success"].append(rel_path)
            else:
                results["failed"].append(
                    {
                        "rel_path": rel_path,
                        "size_ok": size_ok,
                        "head_ok": head_ok,
                        "expected_size": expected_size,
                        "dst_size": dst_size,
                        "expected_head": manifest_head,
                        "actual_head": head_dst,
                        "dst": str(dst),
                    }
                )
        else:
            results["success"].append(rel_path)

    result = {
        "baseline_dir": str(baseline_dir),
        "dst_base": str(dst_base),
        "success_count": len(results["success"]),
        "fail_count": len(results["failed"]),
        "failed": results["failed"],
    }
    (report_dir / "migration_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 写 large_files_manifest.json（rel_path 版本）
    out_dir = baseline_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = []
    for item in normalized:
        sha256_head = item.get("sha256_head")
        if not sha256_head:
            sha256_head = _sha256_head(Path(item["path"]), HEAD_BYTES)
        out_manifest.append(
            {
                "rel_path": item["rel_path"],
                "size": item["size"],
                "sha256_head": sha256_head,
            }
        )
    (out_dir / "large_files_manifest.json").write_text(
        json.dumps(out_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    artifact_info = {
        "skill_id": "lidar_fusion_full_utm32",
        "artifacts_base_abs": str(dst_base),
        "layout": "manifest_rel_path_under_base",
        "migrated_from_abs": str(src_base),
        "migrated_at": datetime.now().isoformat(timespec="seconds"),
        "notes": "Golden8 baseline migrated to KITTI360 root",
    }
    (baseline_dir / "artifact_location.json").write_text(
        json.dumps(artifact_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    migration_verified = len(results["failed"]) == 0
    (report_dir / "manifest_baseline.json").write_text(
        json.dumps(
            {
                "artifacts_base_abs": str(dst_base),
                "migration_plan": str(report_dir / "migration_plan.json"),
                "migration_result": str(report_dir / "migration_result.json"),
                "migration_verified": migration_verified,
                "large_files_manifest": str(out_dir / "large_files_manifest.json"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (baseline_dir / "DONE.ok").write_text("OK", encoding="utf-8")

    if MODE == "copy" and src_base.exists() and src_base != dst_base:
        if str(src_base).lower().startswith(str(REPO_ROOT).lower()):
            return 0
        suffix = datetime.now().strftime("%Y%m%d%H%M")
        backup = Path(str(src_base) + f"_bak_{suffix}")
        if not backup.exists():
            src_base.rename(backup)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

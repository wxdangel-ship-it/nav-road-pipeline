from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    __package__ = "tools"

import json
import re
from pathlib import Path
from typing import Dict, List

import laspy

from tools.resolve_baseline import resolve_laz_paths


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_epsg(header: laspy.header.LasHeader) -> int | None:
    try:
        crs = header.parse_crs()
        if crs is None:
            return None
        return int(crs.to_epsg() or 0) or None
    except Exception:
        return None


def main() -> int:
    pointer = REPO_ROOT / "baselines" / "ACTIVE_LIDAR_FUSION_BASELINE.txt"
    baseline_dir = Path(pointer.read_text(encoding="utf-8").strip())
    report_dir = baseline_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    evidence_dir = baseline_dir / "evidence"
    index_checks: List[Dict] = []
    bbox_checks: List[Dict] = []
    if evidence_dir.exists():
        for drive_dir in sorted([p for p in evidence_dir.iterdir() if p.is_dir()]):
            index_path = drive_dir / "fused_points_utm32_index.json"
            bbox_path = drive_dir / "bbox_utm32.geojson"
            index_checks.append({"path": str(index_path), "exists": index_path.exists()})
            bbox_checks.append({"path": str(bbox_path), "exists": bbox_path.exists()})

    laz_paths = resolve_laz_paths(baseline_dir)
    by_drive: Dict[str, List[Path]] = {}
    drive_pattern = re.compile(r"drive_(\d{4})_sync")
    for p in laz_paths:
        m = drive_pattern.search(str(p))
        drive_key = m.group(1) if m else "unknown"
        by_drive.setdefault(drive_key, []).append(p)
    sampled: List[Path] = []
    for drive_key in sorted(by_drive.keys()):
        sampled.append(by_drive[drive_key][0])
    if not sampled:
        sampled = laz_paths[: min(8, len(laz_paths))]

    results: List[Dict] = []
    ok = True
    if any(not item["exists"] for item in index_checks):
        ok = False
    if any(not item["exists"] for item in bbox_checks):
        ok = False
    for p in sampled:
        entry = {"path": str(p), "exists": p.exists()}
        if not p.exists():
            ok = False
            results.append(entry)
            continue
        try:
            with laspy.open(p) as reader:
                header = reader.header
                entry["point_count"] = int(header.point_count)
                entry["epsg"] = _read_epsg(header)
                entry["bounds"] = {
                    "minx": float(header.mins[0]),
                    "miny": float(header.mins[1]),
                    "maxx": float(header.maxs[0]),
                    "maxy": float(header.maxs[1]),
                }
        except Exception as exc:
            ok = False
            entry["error"] = str(exc)
        if entry.get("epsg") != 32632:
            ok = False
        results.append(entry)

    out = {
        "baseline_dir": str(baseline_dir),
        "index_checks": index_checks,
        "bbox_checks": bbox_checks,
        "sampled_laz": len(sampled),
        "ok": ok,
        "results": results,
    }
    (report_dir / "smoke_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

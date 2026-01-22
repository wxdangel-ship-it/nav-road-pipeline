from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_index(path: Path) -> List[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _top_crosswalk_drive(evidence_gpkg: Path) -> str:
    try:
        import pyogrio

        df = pyogrio.read_dataframe(evidence_gpkg, layer="crosswalk")
    except Exception:
        return ""
    if df.empty or "drive_id" not in df.columns:
        return ""
    counts = df["drive_id"].dropna().value_counts()
    if counts.empty:
        return ""
    return str(counts.index[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--config", default="configs/qa_focus.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--evidence-gpkg", default="")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path) if cfg_path.exists() else {}
    focus = cfg.get("focus", {})
    drives = list(focus.get("drives", []))
    max_frames = int(focus.get("max_frames_per_drive", 50))
    stride = int(focus.get("stride", 1))
    include_top = bool(focus.get("include_top_crosswalk_drive", False))

    if include_top and args.evidence_gpkg:
        top_drive = _top_crosswalk_drive(Path(args.evidence_gpkg))
        if top_drive and top_drive not in drives:
            drives.append(top_drive)

    rows = _load_index(Path(args.index))
    if drives:
        rows = [r for r in rows if str(r.get("drive_id")) in drives]
    if not rows:
        Path(args.out).write_text("", encoding="utf-8")
        return 0

    by_drive: Dict[str, List[dict]] = {}
    for row in rows:
        drive_id = str(row.get("drive_id") or "")
        frame_id = str(row.get("frame_id") or "")
        if not drive_id or not frame_id:
            continue
        by_drive.setdefault(drive_id, []).append(row)
    out_rows = []
    for drive_id, items in by_drive.items():
        items = sorted(items, key=lambda r: str(r.get("frame_id")))
        items = items[::max(1, stride)]
        out_rows.extend(items[:max_frames])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in out_rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

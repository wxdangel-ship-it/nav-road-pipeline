from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from shapely.geometry import Point, shape, mapping
from shapely.ops import unary_union, transform as geom_transform
from pyproj import Transformer
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.sat_intersections import load_tile_index, run_sat_intersections


def _read_index(path: Path) -> List[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        items.append(json.loads(line))
    return items


def _read_drives(path: Path) -> List[str]:
    if not path.exists():
        return []
    drives = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        drives.append(line)
    return drives


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_dop20_root(run_cfg: dict) -> Tuple[Optional[Path], str]:
    cfg_value = run_cfg.get("dop20_root")
    if cfg_value is not None:
        cfg_str = str(cfg_value).strip()
        if cfg_str:
            return Path(cfg_str), "config"
    env_key = run_cfg.get("dop20_root_env", "DOP20_ROOT")
    env_value = os.getenv(env_key or "", "").strip()
    if env_value:
        return Path(env_value), "env"
    fallback = str(run_cfg.get("dop20_root_fallback", "") or "").strip()
    if fallback:
        return Path(fallback), "fallback"
    return None, "unset"


def _resolve_index_cache_path(dop20_root: Path) -> Path:
    override = os.getenv("DOP20_INDEX_CACHE", "").strip()
    if override:
        return Path(override)
    return dop20_root / "dop20_tiles_index.json"


def _read_index_payload(cache_path: Path) -> Tuple[Optional[list], Optional[dict]]:
    if not cache_path.exists():
        return None, None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, None
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        items = payload.get("items")
        meta = payload.get("meta")
        if isinstance(items, list):
            return items, meta if isinstance(meta, dict) else None
    return None, None


def _items_bbox(items: List[dict]) -> Optional[Tuple[float, float, float, float]]:
    if not items:
        return None
    return (
        float(min(item["minx"] for item in items)),
        float(min(item["miny"] for item in items)),
        float(max(item["maxx"] for item in items)),
        float(max(item["maxy"] for item in items)),
    )


def _points_bbox(points: List[Point]) -> Optional[Tuple[float, float, float, float]]:
    if not points:
        return None
    return (
        float(min(p.x for p in points)),
        float(min(p.y for p in points)),
        float(max(p.x for p in points)),
        float(max(p.y for p in points)),
    )


def _bbox_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _sample_line_points(geom, step_m: float) -> List[Tuple[float, float]]:
    if geom.is_empty or geom.length <= 0:
        return []
    n = max(2, int(math.ceil(geom.length / step_m)) + 1)
    return [geom.interpolate(float(i) / (n - 1), normalized=True).coords[0] for i in range(n)]


def _load_candidates(outputs_dir: Path) -> List[Point]:
    cand_paths = [
        outputs_dir / "intersections_algo.geojson",
        outputs_dir / "intersections.geojson",
    ]
    for path in cand_paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        feats = data.get("features", [])
        if not feats:
            return []
        points = []
        for feat in feats:
            geom = shape(feat.get("geometry"))
            if geom.is_empty:
                continue
            points.append(geom.centroid)
        return points
    return []


def _load_traj_points(outputs_dir: Path, step_m: float) -> List[Tuple[float, float]]:
    path = outputs_dir / "centerlines.geojson"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    feats = data.get("features", [])
    points: List[Tuple[float, float]] = []
    for feat in feats:
        geom = shape(feat.get("geometry"))
        if geom.is_empty:
            continue
        points.extend(_sample_line_points(geom, step_m))
    return points


def _write_geojson(path: Path, features: List[dict]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_wgs84(features: List[dict], crs_epsg: int) -> List[dict]:
    wgs84 = Transformer.from_crs(f"EPSG:{crs_epsg}", "EPSG:4326", always_xy=True)
    out = []
    for feat in features:
        geom = geom_transform(wgs84.transform, shape(feat["geometry"]))
        out.append({"type": "Feature", "geometry": mapping(geom), "properties": feat.get("properties") or {}})
    return out


def _merge_segment_features(seg_dirs: List[Path], drive_id: str) -> List[dict]:
    feats: List[dict] = []
    for seg in seg_dirs:
        path = seg / "intersections_sat.geojson"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for feat in data.get("features", []):
            props = feat.get("properties") or {}
            props.setdefault("drive_id", drive_id)
            props.setdefault("tile_id", drive_id)
            feat["properties"] = props
            feats.append(feat)
    return feats


def _write_per_drive_outputs(outputs_dir: Path, drive_id: str, features: List[dict], crs_epsg: int) -> None:
    for feat in features:
        props = feat.get("properties") or {}
        props.setdefault("drive_id", drive_id)
        props.setdefault("tile_id", drive_id)
        feat["properties"] = props
    _write_geojson(outputs_dir / "intersections_sat.geojson", features)
    _write_geojson(outputs_dir / "intersections_sat_wgs84.geojson", _to_wgs84(features, crs_epsg))


def _qc_row(
    drive_id: str,
    missing_reason: str,
    counts: dict,
    conf_stats: dict,
    intersections_count: int,
    area_sum: float,
    note: str = "",
) -> dict:
    status = "OK"
    if missing_reason != "OK":
        status = "FAIL" if missing_reason in {"missing_inputs", "read_error", "index_failed"} else "EMPTY"
    return {
        "drive_id": drive_id,
        "tile_id": drive_id,
        "status": status,
        "missing_reason": missing_reason,
        "candidates_total": counts.get("candidates_total"),
        "candidates_used": counts.get("candidates_used"),
        "tiles_hit": counts.get("tiles_hit"),
        "tiles_total": counts.get("tiles_total"),
        "intersections_count": intersections_count,
        "area_sum": round(area_sum, 3),
        "conf_mean": conf_stats.get("conf_mean"),
        "conf_p50": conf_stats.get("conf_p50"),
        "conf_p95": conf_stats.get("conf_p95"),
        "note": note,
    }


def _write_csv(path: Path, rows: List[dict]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("N/A" if row.get(k) is None else row.get(k, "")) for k in fieldnames})


def _write_missing_reason_summary(out_csv: Path, expected_drives: List[str], report_type: str, run_id: str) -> dict:
    import csv
    from collections import Counter

    rows = []
    if out_csv.exists():
        with out_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    counts = Counter()
    non_ok = []
    for row in rows:
        reason = (row.get("missing_reason") or "").strip()
        norm = "" if reason in {"", "N/A", "OK"} else reason
        counts[norm or "OK"] += 1
        if norm:
            non_ok.append({"drive_id": row.get("drive_id"), "reason": norm})
    payload = {
        "expected_drives": expected_drives,
        "missing_reason_counts": dict(counts),
        "non_ok_drives": non_ok,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "report_type": report_type,
    }
    json_path = out_csv.with_name(out_csv.stem + "_missing_reason_summary.json")
    md_path = out_csv.with_name(out_csv.stem + "_missing_reason_summary.md")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Missing Reason Summary",
        "",
        f"- report_type: {report_type}",
        f"- run_id: {run_id}",
        f"- generated_at: {payload['generated_at']}",
        "",
        "## expected_drives",
        "```json",
        json.dumps(expected_drives, ensure_ascii=False, indent=2),
        "```",
        "",
        "## missing_reason_counts",
        "```json",
        json.dumps(payload["missing_reason_counts"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## non_ok_drives",
        "```json",
        json.dumps(non_ok, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def _segment_indices(total: int, seg_size: int) -> List[Tuple[int, int]]:
    if total <= 0:
        return []
    segs = []
    for start in range(0, total, seg_size):
        segs.append((start, min(total, start + seg_size)))
    return segs


def _merge_conf_stats(conf_values: List[float]) -> dict:
    if not conf_values:
        return {"conf_mean": None, "conf_p50": None, "conf_p95": None}
    conf_sorted = sorted(conf_values)
    mean = sum(conf_values) / len(conf_values)
    p50 = conf_sorted[len(conf_sorted) // 2]
    p95 = conf_sorted[int(max(0, len(conf_sorted) * 0.95) - 1)]
    return {"conf_mean": round(float(mean), 4), "conf_p50": round(float(p50), 4), "conf_p95": round(float(p95), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="postopt_index.jsonl or similar")
    ap.add_argument("--stage", default="full", help="index stage filter (quick/full)")
    ap.add_argument("--candidate", default="", help="candidate_id filter (optional)")
    ap.add_argument("--config", default="configs/sat_intersections_full.yaml")
    ap.add_argument("--out-dir", default="", help="sat run output dir")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--write-back", action="store_true", help="write per-drive sat outputs back to geom outputs")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    sat_cfg = cfg.get("sat", {}) or {}
    run_cfg = cfg.get("run", {}) or {}
    expected_file = Path(run_cfg.get("expected_drives_file", "configs/golden_drives.txt"))
    expected_drives = _read_drives(expected_file)
    patch_m = float(sat_cfg.get("patch_m", 256.0))
    conf_thr = float(sat_cfg.get("conf_thr", 0.3))
    crs_epsg = int(sat_cfg.get("crs_epsg", 32632))
    quick_max_candidates = int(run_cfg.get("quick_max_candidates", 20))
    full_segment_size = int(run_cfg.get("full_segment_size", 50))
    traj_sample_m = float(run_cfg.get("traj_sample_m", 8.0))
    dop20_root, dop20_source = _resolve_dop20_root(run_cfg)
    dop20_state = "unset"
    if dop20_root is not None:
        if dop20_root.exists():
            dop20_root = dop20_root.resolve()
            dop20_state = "ok"
        else:
            dop20_state = "missing"

    if dop20_state == "ok":
        print(f"[SAT] dop20_root={dop20_root} (source={dop20_source})")
    else:
        print(f"[SAT] dop20_root={dop20_state} (source={dop20_source})")

    tiles_dir = dop20_root / "tiles_utm32" if dop20_root else None
    tiles_dir_exists = bool(tiles_dir and tiles_dir.exists())
    if tiles_dir:
        print(f"[SAT] dop20_tiles_dir={tiles_dir} exists={str(tiles_dir_exists).lower()}")

    index_items = None
    index_bbox = None
    index_cache_path = _resolve_index_cache_path(dop20_root) if dop20_root else None
    if index_cache_path:
        cache_items, cache_meta = _read_index_payload(index_cache_path)
        force_rebuild = False
        if cache_items is not None:
            cached_root = (cache_meta or {}).get("root_abs")
            if not cached_root:
                force_rebuild = True
            else:
                try:
                    if str(Path(str(cached_root)).resolve()) != str(dop20_root):
                        force_rebuild = True
                except Exception:
                    force_rebuild = True
        if force_rebuild:
            print(f"[SAT] WARNING index_root_mismatch, rebuilding index at {index_cache_path}")
        if tiles_dir_exists:
            index_items, index_meta = load_tile_index(
                dop20_root,
                tiles_dir,
                crs_epsg=crs_epsg,
                force_rebuild=force_rebuild,
            )
            index_bbox = None
            if isinstance(index_meta, dict):
                bbox = index_meta.get("bbox")
                if isinstance(bbox, dict):
                    try:
                        index_bbox = (
                            float(bbox.get("minx")),
                            float(bbox.get("miny")),
                            float(bbox.get("maxx")),
                            float(bbox.get("maxy")),
                        )
                    except Exception:
                        index_bbox = None
            if index_bbox is None and index_items:
                index_bbox = _items_bbox(index_items)
            tiles_count = len(index_items) if index_items else 0
            bbox_text = "None" if not index_bbox else ",".join(f"{v:.3f}" for v in index_bbox)
            print(f"[SAT] dop20_index={index_cache_path} tiles={tiles_count} bbox={bbox_text}")

    run_id = ""
    if args.out_dir:
        run_dir = Path(args.out_dir)
        run_id = run_dir.name
    else:
        run_id = f"sat_intersections_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path("runs") / run_id
    outputs_dir = run_dir / "outputs"
    segments_root = run_dir / "segments"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    segments_root.mkdir(parents=True, exist_ok=True)

    entries = _read_index(Path(args.index))
    entries = [
        e for e in entries
        if e.get("stage") == args.stage
        and e.get("status") == "PASS"
        and e.get("outputs_dir")
    ]
    if args.candidate:
        entries = [e for e in entries if e.get("candidate_id") == args.candidate]

    per_drive_rows = []
    merged_features: List[dict] = []
    seen_drives: set[str] = set()

    for entry in entries:
        drive = str(entry.get("drive") or entry.get("tile_id") or "")
        out_dir = Path(entry.get("outputs_dir"))
        if not drive or not out_dir.exists():
            continue

        candidates = _load_candidates(out_dir)
        traj_points = _load_traj_points(out_dir, traj_sample_m)
        missing_reason = "OK"
        counts = {
            "candidates_total": 0,
            "candidates_used": 0,
            "tiles_hit": 0,
            "tiles_total": 0,
            "patches_read": 0,
            "read_errors": 0,
        }
        conf_values: List[float] = []
        features: List[dict] = []

        query_bbox = _points_bbox(candidates)

        if dop20_state != "ok":
            missing_reason = "dop20_root_unset" if dop20_state == "unset" else "dop20_root_missing"
        elif args.finalize:
            seg_dirs = []
            seg_parent = segments_root / drive
            if seg_parent.exists():
                seg_dirs = sorted([p for p in seg_parent.iterdir() if p.is_dir()])
            features = _merge_segment_features(seg_dirs, drive)
        else:
            if not candidates:
                missing_reason = "missing_inputs"
            else:
                if args.stage == "quick":
                    candidates = candidates[:quick_max_candidates]
                segs = _segment_indices(len(candidates), full_segment_size if args.stage == "full" else len(candidates))
                seg_parent = segments_root / drive
                seg_parent.mkdir(parents=True, exist_ok=True)
                for idx, (s, e) in enumerate(segs):
                    seg_dir = seg_parent / f"seg_{idx:03d}"
                    seg_out = seg_dir / "intersections_sat.geojson"
                    if args.resume and seg_out.exists():
                        continue
                    seg_dir.mkdir(parents=True, exist_ok=True)
                    seg_candidates = candidates[s:e]
                    (seg_dir / "candidates.json").write_text(
                        json.dumps([{"x": p.x, "y": p.y} for p in seg_candidates], ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    result = run_sat_intersections(
                        drive=drive,
                        candidates=seg_candidates,
                        traj_points=traj_points,
                        outputs_dir=seg_dir,
                        crs_epsg=crs_epsg,
                        patch_m=patch_m,
                        conf_thr=conf_thr,
                        dop20_root=dop20_root,
                    )
                    (seg_dir / "segment_meta.json").write_text(
                        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                    )

            seg_dirs = []
            seg_parent = segments_root / drive
            if seg_parent.exists():
                seg_dirs = sorted([p for p in seg_parent.iterdir() if p.is_dir()])
            features = _merge_segment_features(seg_dirs, drive)

        seg_parent = segments_root / drive
        if seg_parent.exists():
            for seg_dir in sorted([p for p in seg_parent.iterdir() if p.is_dir()]):
                meta_path = seg_dir / "segment_meta.json"
                if not meta_path.exists():
                    continue
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                counts["candidates_total"] += int(meta.get("candidates_total") or 0)
                counts["candidates_used"] += int(meta.get("candidates_used") or 0)
                counts["tiles_hit"] += int(meta.get("tiles_hit") or 0)
                counts["tiles_total"] = max(int(meta.get("tiles_total") or 0), counts["tiles_total"])
                counts["patches_read"] += int(meta.get("patches_read") or 0)
                counts["read_errors"] += int(meta.get("read_errors") or 0)

        for feat in features:
            props = feat.get("properties") or {}
            conf = props.get("sat_confidence")
            if isinstance(conf, (int, float)):
                conf_values.append(float(conf))

        intersections_count = len(features)
        area_sum = 0.0
        for feat in features:
            geom = shape(feat.get("geometry"))
            area_sum += float(geom.area)

        if not features and missing_reason == "OK":
            if counts["candidates_total"] <= 0:
                missing_reason = "no_candidates"
            elif not tiles_dir_exists:
                missing_reason = "tiles_dir_missing"
            elif counts["tiles_total"] <= 0:
                missing_reason = "no_tiles"
            elif counts["tiles_hit"] <= 0:
                if query_bbox and index_bbox and not _bbox_intersects(query_bbox, index_bbox):
                    missing_reason = "out_of_coverage"
                else:
                    missing_reason = "no_tiles"
            elif counts["patches_read"] <= 0:
                missing_reason = "read_error"
            else:
                missing_reason = "low_confidence"

        conf_stats = _merge_conf_stats(conf_values)
        per_drive_rows.append(
            _qc_row(
                drive_id=drive,
                missing_reason=missing_reason,
                counts=counts,
                conf_stats=conf_stats,
                intersections_count=intersections_count,
                area_sum=area_sum,
            )
        )
        seen_drives.add(drive)

        if args.write_back:
            _write_per_drive_outputs(out_dir, drive, features, crs_epsg)
            (out_dir / "intersections_sat_qc.json").write_text(
                json.dumps(per_drive_rows[-1], ensure_ascii=False, indent=2), encoding="utf-8"
            )

        merged_features.extend(features)

    for d in expected_drives:
        if d in seen_drives:
            continue
        per_drive_rows.append(
            _qc_row(
                drive_id=d,
                missing_reason="missing_entry",
                counts={
                    "candidates_total": None,
                    "candidates_used": None,
                    "tiles_hit": None,
                    "tiles_total": None,
                    "patches_read": None,
                    "read_errors": None,
                },
                conf_stats={"conf_mean": None, "conf_p50": None, "conf_p95": None},
                intersections_count=0,
                area_sum=0.0,
                note="missing_entry",
            )
        )

    _write_geojson(outputs_dir / "intersections_sat.geojson", merged_features)
    _write_geojson(outputs_dir / "intersections_sat_wgs84.geojson", _to_wgs84(merged_features, crs_epsg))

    qc_csv = outputs_dir / f"{args.stage}_sat_report_per_drive.csv"
    qc_json = outputs_dir / f"{args.stage}_sat_report_per_drive.json"
    _write_csv(qc_csv, per_drive_rows)
    qc_json.write_text(json.dumps(per_drive_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = _write_missing_reason_summary(qc_csv, expected_drives, args.stage, run_id)

    print(f"[SAT] DONE -> {run_dir}")
    print(f"[SAT] report_csv={qc_csv}")
    print(f"[SAT] missing_reason_counts={summary.get('missing_reason_counts')}")
    non_ok = summary.get("non_ok_drives") or []
    print(f"[SAT] non_ok_drives={'empty' if not non_ok else non_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

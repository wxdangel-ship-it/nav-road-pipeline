from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from shapely.geometry import shape


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_index(path: Path, stage: str | None, candidate: str | None) -> List[Dict[str, Any]]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if stage and entry.get("stage") != stage:
            continue
        if candidate and entry.get("candidate_id") != candidate:
            continue
        items.append(entry)
    return items


def _load_baseline(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: Dict[str, Dict[str, Any]] = {}
    for item in data.get("drives", []) or []:
        drive = item.get("drive")
        if drive:
            out[str(drive)] = item
    return out


def _baseline_drives(baseline: Dict[str, Dict[str, Any]], entries: List[Dict[str, Any]]) -> List[str]:
    if baseline:
        return sorted(baseline.keys())
    drives = sorted({e.get("drive") for e in entries if e.get("drive")})
    return drives


def _polygon_metrics(road_path: Path) -> Tuple[Optional[float], Optional[int], Optional[float]]:
    if not road_path.exists():
        return None, None, None
    data = json.loads(road_path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    if not features:
        return None, None, None
    geom = shape(features[0]["geometry"])
    if geom.is_empty:
        return None, None
    area = float(geom.area)
    perim = float(geom.length)
    if area <= 0:
        roughness = None
    else:
        roughness = (perim * perim) / area

    def _count_coords(g) -> int:
        if g.geom_type == "Polygon":
            return len(g.exterior.coords)
        if g.geom_type == "MultiPolygon":
            return sum(len(p.exterior.coords) for p in g.geoms)
        return 0

    vertex_count = _count_coords(geom)
    return roughness, vertex_count, area


def _centerline_metrics(center_path: Path) -> Tuple[Optional[float], Optional[int]]:
    if not center_path.exists():
        return None, None
    data = json.loads(center_path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    if not features:
        return None, None
    total_len = 0.0
    dual_len = 0.0
    for feat in features:
        geom = shape(feat.get("geometry"))
        if geom.is_empty:
            continue
        length = float(geom.length)
        total_len += length
        lane_mode = (feat.get("properties") or {}).get("lane_mode")
        if lane_mode in {"dual_left", "dual_right"}:
            dual_len += length
    dual_ratio = dual_len / total_len if total_len > 0 else None
    return dual_ratio, len(features)


def _score_drive(
    outputs_dir: Path,
    drive: str,
    baseline: Dict[str, Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    summary = _read_json(outputs_dir / "GeomSummary.json")
    if not summary:
        return "FAIL", {"reason": "missing_summary"}
    if summary.get("status") != "PASS":
        return "FAIL", {"reason": summary.get("reason") or "summary_fail"}

    base = baseline.get(drive) or {}
    base_ratio = base.get("centerlines_in_polygon_ratio")
    ratio = summary.get("centerlines_in_polygon_ratio")
    if isinstance(base_ratio, (int, float)) and isinstance(ratio, (int, float)):
        if ratio < 0.98:
            return "FAIL", {"reason": "centerlines_ratio_drop"}

    if summary.get("road_component_count_after") != 1:
        return "FAIL", {"reason": "road_components_not_1"}

    osm = _read_json(outputs_dir / "osm_ref_metrics.json")
    if not osm:
        return "FAIL", {"reason": "missing_osm_metrics"}
    if not osm.get("osm_present"):
        return "FAIL", {"reason": "osm_not_present"}

    coverage_ok = bool(osm.get("coverage_ok"))
    metrics_valid = bool(osm.get("metrics_valid"))
    match_ratio = osm.get("match_ratio") if metrics_valid and coverage_ok else None
    dist_p95 = osm.get("dist_p95_m") if metrics_valid and coverage_ok else None

    roughness, vertex_count, area_m2 = _polygon_metrics(outputs_dir / "road_polygon.geojson")
    if summary.get("roughness") is not None:
        roughness = summary.get("roughness")
    elif summary.get("polygon_roughness") is not None:
        roughness = summary.get("polygon_roughness")
    if summary.get("vertex_count") is not None:
        vertex_count = summary.get("vertex_count")
    elif summary.get("polygon_vertex_count") is not None:
        vertex_count = summary.get("polygon_vertex_count")
    if summary.get("polygon_area_m2") is not None:
        area_m2 = summary.get("polygon_area_m2")
    dual_ratio, center_feat_count = _centerline_metrics(outputs_dir / "centerlines.geojson")

    base_area = base.get("polygon_area_m2") or base.get("road_polygon_area_m2")
    area_penalty = None
    area_ratio = None
    if isinstance(base_area, (int, float)) and isinstance(area_m2, (int, float)) and base_area > 0:
        area_ratio = abs(float(area_m2) - float(base_area)) / float(base_area)
        if area_ratio <= 0.2:
            area_penalty = area_ratio
        else:
            area_penalty = area_ratio * 2.0

    max_rough = os.environ.get("POSTOPT_MAX_ROUGHNESS", "").strip()
    max_vertex = os.environ.get("POSTOPT_MAX_VERTEX_COUNT", "").strip()
    if max_rough:
        try:
            if roughness is not None and float(roughness) > float(max_rough):
                return "FAIL", {"reason": "roughness_cap"}
        except ValueError:
            pass
    if max_vertex:
        try:
            if vertex_count is not None and float(vertex_count) > float(max_vertex):
                return "FAIL", {"reason": "vertex_count_cap"}
        except ValueError:
            pass

    return "PASS", {
        "match_ratio": match_ratio,
        "dist_p95_m": dist_p95,
        "metrics_valid": metrics_valid,
        "coverage_ok": coverage_ok,
        "roughness": roughness,
        "vertex_count": vertex_count,
        "polygon_area_m2": area_m2,
        "area_ratio": area_ratio,
        "area_penalty": area_penalty,
        "dual_ratio": dual_ratio,
        "center_feat_count": center_feat_count,
        "centerlines_in_polygon_ratio": ratio,
    }


def _aggregate(
    entries: List[Dict[str, Any]],
    baseline: Dict[str, Dict[str, Any]],
    weights: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_candidate: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        cid = entry.get("candidate_id") or "unknown"
        by_candidate.setdefault(cid, []).append(entry)

    candidate_rows: List[Dict[str, Any]] = []
    drive_rows: List[Dict[str, Any]] = []

    for cid, items in sorted(by_candidate.items()):
        pass_count = fail_count = skipped_count = 0
        scores: List[float] = []
        reasons: List[str] = []
        drive_list = _baseline_drives(baseline, items)
        entry_map = {(e.get("drive") or ""): e for e in items}
        for drive in drive_list:
            entry = entry_map.get(drive)
            if entry is None:
                fail_count += 1
                reasons.append("missing_entry")
                drive_rows.append(
                    {
                        "candidate_id": cid,
                        "drive": drive,
                        "status": "FAIL",
                        "score": None,
                        "match_ratio": None,
                        "dist_p95_m": None,
                        "roughness": None,
                        "vertex_count": None,
                        "polygon_area_m2": None,
                        "area_ratio": None,
                        "area_penalty": None,
                        "dual_ratio": None,
                        "center_feat_count": None,
                        "centerlines_in_polygon_ratio": None,
                        "missing_reason": "missing_entry",
                    }
                )
                continue

            if entry.get("status") != "PASS":
                fail_count += 1
                reason = entry.get("reason") or "build_geom_failed"
                reasons.append(reason)
                drive_rows.append(
                    {
                        "candidate_id": cid,
                        "drive": entry.get("drive"),
                        "status": "FAIL",
                        "score": None,
                        "match_ratio": None,
                        "dist_p95_m": None,
                        "roughness": None,
                        "vertex_count": None,
                        "polygon_area_m2": None,
                        "area_ratio": None,
                        "area_penalty": None,
                        "dual_ratio": None,
                        "center_feat_count": None,
                        "centerlines_in_polygon_ratio": None,
                        "missing_reason": reason,
                    }
                )
                continue

            outputs_dir = entry.get("outputs_dir")
            if not outputs_dir:
                fail_count += 1
                reasons.append("missing_outputs_dir")
                drive_rows.append(
                    {
                        "candidate_id": cid,
                        "drive": entry.get("drive"),
                        "status": "FAIL",
                        "score": None,
                        "match_ratio": None,
                        "dist_p95_m": None,
                        "roughness": None,
                        "vertex_count": None,
                        "polygon_area_m2": None,
                        "area_ratio": None,
                        "area_penalty": None,
                        "dual_ratio": None,
                        "center_feat_count": None,
                        "centerlines_in_polygon_ratio": None,
                        "missing_reason": "missing_outputs_dir",
                    }
                )
                continue

            status, meta = _score_drive(Path(outputs_dir), drive, baseline)
            score = None
            missing_reason = None
            if status == "PASS":
                pass_count += 1
                score = 0.0
                if meta.get("match_ratio") is not None and meta.get("dist_p95_m") is not None:
                    score += float(meta["match_ratio"]) * weights["osm_match_ratio"]
                    score += float(meta["dist_p95_m"]) * weights["dist_p95_m"]
                if meta.get("roughness") is not None:
                    score += float(meta["roughness"]) * weights["roughness"]
                if meta.get("vertex_count") is not None:
                    score += float(meta["vertex_count"]) * weights["vertex_count"]
                if meta.get("area_penalty") is not None:
                    score += float(meta["area_penalty"]) * weights["area_penalty"]
                if meta.get("dual_ratio") is not None:
                    score += float(meta["dual_ratio"]) * weights["dual_ratio"]
                if meta.get("center_feat_count") is not None:
                    score += -abs(float(meta["center_feat_count"]) - 2.0) * weights["center_feat_gap"]
                scores.append(score)
            else:
                fail_count += 1
                missing_reason = meta.get("reason") or "qc_failed"
                reasons.append(missing_reason)

            drive_rows.append(
                {
                    "candidate_id": cid,
                    "drive": drive,
                    "status": status,
                    "score": score,
                    "match_ratio": meta.get("match_ratio"),
                    "dist_p95_m": meta.get("dist_p95_m"),
                    "roughness": meta.get("roughness"),
                    "vertex_count": meta.get("vertex_count"),
                    "polygon_area_m2": meta.get("polygon_area_m2"),
                    "area_ratio": meta.get("area_ratio"),
                    "area_penalty": meta.get("area_penalty"),
                    "dual_ratio": meta.get("dual_ratio"),
                    "center_feat_count": meta.get("center_feat_count"),
                    "centerlines_in_polygon_ratio": meta.get("centerlines_in_polygon_ratio"),
                    "missing_reason": missing_reason,
                }
            )

        if fail_count > 0:
            status = "FAIL"
        elif pass_count == 0:
            status = "SKIPPED"
        else:
            status = "PASS"

        avg_score = sum(scores) / len(scores) if scores else None
        candidate_rows.append(
            {
                "candidate_id": cid,
                "status": status,
                "avg_score": avg_score,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "skipped_count": skipped_count,
                "reasons": ";".join(sorted(set(reasons))),
            }
        )

    return candidate_rows, drive_rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("N/A" if row.get(k) is None else row.get(k, "")) for k in fieldnames})


def _fmt(val: Any, fmt: str) -> str:
    if val is None or val == "":
        return "N/A"
    try:
        if fmt:
            return format(float(val), fmt)
    except (ValueError, TypeError):
        return str(val)
    return str(val)


def _write_md(path: Path, candidates: List[Dict[str, Any]], drives: List[Dict[str, Any]]) -> None:
    counts = {"PASS": 0, "FAIL": 0, "SKIPPED": 0}
    for row in candidates:
        status = row.get("status")
        if status in counts:
            counts[status] += 1

    lines = [
        "# Geom Postopt Score",
        "",
        f"- total_candidates: {len(candidates)}",
        f"- pass: {counts['PASS']}",
        f"- fail: {counts['FAIL']}",
        f"- skipped: {counts['SKIPPED']}",
        "",
        "## Candidates",
        "",
        "| candidate_id | status | avg_score | pass | fail | skipped | reasons |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in candidates:
        lines.append(
            "| {cid} | {status} | {score} | {p} | {f} | {s} | {r} |".format(
                cid=row.get("candidate_id") or "",
                status=row.get("status") or "",
                score="" if row.get("avg_score") is None else f"{row.get('avg_score'):.4f}",
                p=row.get("pass_count") or 0,
                f=row.get("fail_count") or 0,
                s=row.get("skipped_count") or 0,
                r=row.get("reasons") or "",
            )
        )
    lines.append("")
    lines.append("## Per-Drive")
    lines.append("")
    lines.append(
        "| candidate_id | drive | status | score | match_ratio | dist_p95_m | roughness | vertex_count | area_ratio | dual_ratio | missing_reason |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in drives:
        lines.append(
            "| {cid} | {drive} | {status} | {score} | {mr} | {dp} | {r} | {v} | {ar} | {dr} | {reason} |".format(
                cid=row.get("candidate_id") or "",
                drive=row.get("drive") or "",
                status=row.get("status") or "",
                score=_fmt(row.get("score"), ".4f"),
                mr=_fmt(row.get("match_ratio"), ".4f"),
                dp=_fmt(row.get("dist_p95_m"), ".2f"),
                r=_fmt(row.get("roughness"), ".2f"),
                v=_fmt(row.get("vertex_count"), ""),
                ar=_fmt(row.get("area_ratio"), ".3f"),
                dr=_fmt(row.get("dual_ratio"), ".2f"),
                reason=row.get("missing_reason") or "",
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--baseline", default="configs/geom_regress_baseline.yaml")
    ap.add_argument("--stage", default="")
    ap.add_argument("--candidate", default="")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    weights = {
        "osm_match_ratio": 1.0,
        "dist_p95_m": -0.05,
        "roughness": -0.6,
        "vertex_count": -0.01,
        "area_penalty": -0.3,
        "dual_ratio": 0.2,
        "center_feat_gap": 0.2,
    }

    index_path = Path(args.index)
    entries = _read_index(index_path, args.stage or None, args.candidate or None)
    baseline = _load_baseline(Path(args.baseline))
    candidates, drives = _aggregate(entries, baseline, weights)

    _write_csv(Path(args.out_csv), candidates)
    _write_md(Path(args.out_md), candidates, drives)

    if args.out_json:
        ranked = sorted(
            candidates,
            key=lambda r: (r.get("status") != "PASS", -(r.get("avg_score") or -1e9)),
        )
        Path(args.out_json).write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

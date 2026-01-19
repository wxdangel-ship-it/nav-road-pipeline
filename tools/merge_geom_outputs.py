from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import pandas as pd
import yaml


def _read_index(path: Path) -> List[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        items.append(json.loads(line))
    return items


def _load_best_candidate(path: Path) -> str:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return str(data.get("candidate_id") or "")


def _select_best_from_per_drive(per_drive_csv: Path) -> str:
    if not per_drive_csv.exists():
        return ""
    import csv

    rows = []
    with per_drive_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return ""
    scores: Dict[str, List[float]] = {}
    for row in rows:
        cid = str(row.get("candidate_id") or "")
        if not cid:
            continue
        try:
            score = float(row.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        scores.setdefault(cid, []).append(score)
    if not scores:
        return ""
    best = max(scores.items(), key=lambda kv: sum(kv[1]) / max(1, len(kv[1])))
    return best[0]


def _load_candidate_post(candidates_path: Path, candidate_id: str) -> dict:
    data = yaml.safe_load(candidates_path.read_text(encoding="utf-8")) or {}
    for cand in data.get("candidates", []) or []:
        if cand.get("candidate_id") == candidate_id:
            return {"candidate_id": candidate_id, **(cand.get("post") or {})}
    return {}


def _ensure_best_postopt(best_path: Path, index_path: Path) -> str:
    if best_path.exists():
        return _load_best_candidate(best_path)
    run_dir = index_path.parent
    per_drive_csv = run_dir / "full_report_per_drive.csv"
    best_candidate = _select_best_from_per_drive(per_drive_csv)
    if not best_candidate:
        return ""
    candidates_path = Path("configs") / "geom_postopt_candidates.yaml"
    payload = _load_candidate_post(candidates_path, best_candidate)
    if not payload:
        return ""
    best_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return best_candidate


def _read_summary(outputs_dir: Path) -> Dict[str, object]:
    summary_path = outputs_dir / "GeomSummary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _merge_layers(entries: List[dict], suffix: str, out_dir: Path, report: dict) -> None:
    center_frames = []
    road_frames = []
    inter_frames = []
    inter_algo_frames = []
    inter_sat_frames = []
    inter_final_frames = []

    for entry in entries:
        outputs_dir = Path(entry["outputs_dir"])
        drive = entry.get("drive") or ""
        run_id = entry.get("geom_run_id") or ""
        candidate_id = entry.get("candidate_id") or ""
        summary = _read_summary(outputs_dir)
        backend_used = summary.get("intersection_backend_used") or summary.get("backend_used") or ""
        sat_present = bool(summary.get("sat_present"))
        sat_conf = summary.get("sat_confidence_avg")

        center_path = outputs_dir / f"centerlines{suffix}.geojson"
        road_path = outputs_dir / f"road_polygon{suffix}.geojson"
        inter_path = outputs_dir / f"intersections{suffix}.geojson"
        inter_algo_path = outputs_dir / f"intersections_algo{suffix}.geojson"
        inter_sat_path = outputs_dir / f"intersections_sat{suffix}.geojson"
        inter_final_path = outputs_dir / f"intersections_final{suffix}.geojson"
        if center_path.exists():
            gdf = gpd.read_file(center_path)
            gdf["drive"] = drive
            gdf["drive_id"] = drive
            gdf["tile_id"] = drive
            gdf["geom_run_id"] = run_id
            gdf["candidate_id"] = candidate_id
            center_frames.append(gdf)
        if road_path.exists():
            gdf = gpd.read_file(road_path)
            gdf["drive"] = drive
            gdf["drive_id"] = drive
            gdf["tile_id"] = drive
            gdf["geom_run_id"] = run_id
            gdf["candidate_id"] = candidate_id
            road_frames.append(gdf)
        if inter_path.exists():
            gdf = gpd.read_file(inter_path)
            gdf["drive"] = drive
            gdf["drive_id"] = drive
            gdf["tile_id"] = drive
            gdf["geom_run_id"] = run_id
            gdf["candidate_id"] = candidate_id
            gdf["backend_used"] = backend_used
            gdf["sat_present"] = sat_present
            gdf["sat_confidence"] = sat_conf
            inter_frames.append(gdf)
        else:
            report.setdefault("missing_layer", []).append({"drive_id": drive, "layer": f"intersections{suffix}"})
        if inter_algo_path.exists():
            gdf = gpd.read_file(inter_algo_path)
            gdf["drive"] = drive
            gdf["drive_id"] = drive
            gdf["tile_id"] = drive
            gdf["geom_run_id"] = run_id
            gdf["candidate_id"] = candidate_id
            gdf["backend_used"] = "algo"
            gdf["sat_present"] = sat_present
            gdf["sat_confidence"] = sat_conf
            inter_algo_frames.append(gdf)
        else:
            report.setdefault("missing_layer", []).append({"drive_id": drive, "layer": f"intersections_algo{suffix}"})
        if inter_sat_path.exists():
            gdf = gpd.read_file(inter_sat_path)
            gdf["drive"] = drive
            gdf["drive_id"] = drive
            gdf["tile_id"] = drive
            gdf["geom_run_id"] = run_id
            gdf["candidate_id"] = candidate_id
            gdf["backend_used"] = "sat"
            gdf["sat_present"] = sat_present
            gdf["sat_confidence"] = sat_conf
            inter_sat_frames.append(gdf)
        else:
            report.setdefault("missing_layer", []).append({"drive_id": drive, "layer": f"intersections_sat{suffix}"})
        if inter_final_path.exists():
            gdf = gpd.read_file(inter_final_path)
            gdf["drive"] = drive
            gdf["drive_id"] = drive
            gdf["tile_id"] = drive
            gdf["geom_run_id"] = run_id
            gdf["candidate_id"] = candidate_id
            inter_final_frames.append(gdf)
        else:
            report.setdefault("missing_layer", []).append({"drive_id": drive, "layer": f"intersections_final{suffix}"})

    if center_frames:
        gpd.GeoDataFrame(pd.concat(center_frames, ignore_index=True)).to_file(
            out_dir / f"merged_centerlines{suffix}.geojson", driver="GeoJSON"
        )
    if road_frames:
        gpd.GeoDataFrame(pd.concat(road_frames, ignore_index=True)).to_file(
            out_dir / f"merged_road_polygon{suffix}.geojson", driver="GeoJSON"
        )
    if inter_frames:
        merged = gpd.GeoDataFrame(pd.concat(inter_frames, ignore_index=True))
        merged.to_file(out_dir / f"merged_intersections{suffix}.geojson", driver="GeoJSON")
    else:
        report.setdefault("empty_layer", []).append({"layer": f"merged_intersections{suffix}"})
    if inter_final_frames:
        merged_final = gpd.GeoDataFrame(pd.concat(inter_final_frames, ignore_index=True))
        merged_final.to_file(out_dir / f"merged_intersections_final{suffix}.geojson", driver="GeoJSON")
    elif inter_frames:
        merged.to_file(out_dir / f"merged_intersections_final{suffix}.geojson", driver="GeoJSON")
    else:
        report.setdefault("empty_layer", []).append({"layer": f"merged_intersections_final{suffix}"})
    if inter_algo_frames:
        gpd.GeoDataFrame(pd.concat(inter_algo_frames, ignore_index=True)).to_file(
            out_dir / f"merged_intersections_algo{suffix}.geojson", driver="GeoJSON"
        )
    else:
        report.setdefault("empty_layer", []).append({"layer": f"merged_intersections_algo{suffix}"})
    if inter_sat_frames:
        gpd.GeoDataFrame(pd.concat(inter_sat_frames, ignore_index=True)).to_file(
            out_dir / f"merged_intersections_sat{suffix}.geojson", driver="GeoJSON"
        )
    else:
        report.setdefault("empty_layer", []).append({"layer": f"merged_intersections_sat{suffix}"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="postopt_index.jsonl")
    ap.add_argument("--best-postopt", required=True, help="best_postopt.yaml")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        raise SystemExit(f"ERROR: index not found: {index_path}")

    best_candidate = _ensure_best_postopt(Path(args.best_postopt), index_path)
    if not best_candidate:
        raise SystemExit("ERROR: best_postopt candidate_id not found")

    entries = _read_index(index_path)
    entries = [
        e for e in entries
        if e.get("stage") == "full"
        and e.get("candidate_id") == best_candidate
        and e.get("status") == "PASS"
        and e.get("outputs_dir")
    ]
    if not entries:
        raise SystemExit("ERROR: no full entries found for best candidate")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    _merge_layers(entries, "", out_dir, report)
    _merge_layers(entries, "_wgs84", out_dir, report)
    (out_dir / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

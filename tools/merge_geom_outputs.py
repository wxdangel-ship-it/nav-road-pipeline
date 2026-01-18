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


def _read_summary(outputs_dir: Path) -> Dict[str, object]:
    summary_path = outputs_dir / "GeomSummary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _merge_layers(entries: List[dict], suffix: str, out_dir: Path) -> None:
    center_frames = []
    road_frames = []
    inter_frames = []
    inter_algo_frames = []
    inter_sat_frames = []

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
        if center_path.exists():
            gdf = gpd.read_file(center_path)
            gdf["drive"] = drive
            gdf["geom_run_id"] = run_id
            gdf["candidate_id"] = candidate_id
            center_frames.append(gdf)
        if road_path.exists():
            gdf = gpd.read_file(road_path)
            gdf["drive"] = drive
            gdf["geom_run_id"] = run_id
            gdf["candidate_id"] = candidate_id
            road_frames.append(gdf)
        if inter_path.exists():
            gdf = gpd.read_file(inter_path)
            gdf["drive"] = drive
            gdf["geom_run_id"] = run_id
            gdf["candidate_id"] = candidate_id
            gdf["backend_used"] = backend_used
            gdf["sat_present"] = sat_present
            gdf["sat_confidence"] = sat_conf
            inter_frames.append(gdf)
        if inter_algo_path.exists():
            gdf = gpd.read_file(inter_algo_path)
            gdf["drive"] = drive
            gdf["geom_run_id"] = run_id
            gdf["candidate_id"] = candidate_id
            gdf["backend_used"] = "algo"
            gdf["sat_present"] = sat_present
            gdf["sat_confidence"] = sat_conf
            inter_algo_frames.append(gdf)
        if inter_sat_path.exists():
            gdf = gpd.read_file(inter_sat_path)
            gdf["drive"] = drive
            gdf["geom_run_id"] = run_id
            gdf["candidate_id"] = candidate_id
            gdf["backend_used"] = "sat"
            gdf["sat_present"] = sat_present
            gdf["sat_confidence"] = sat_conf
            inter_sat_frames.append(gdf)

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
        merged.to_file(out_dir / f"merged_intersections_final{suffix}.geojson", driver="GeoJSON")
    if inter_algo_frames:
        gpd.GeoDataFrame(pd.concat(inter_algo_frames, ignore_index=True)).to_file(
            out_dir / f"merged_intersections_algo{suffix}.geojson", driver="GeoJSON"
        )
    if inter_sat_frames:
        gpd.GeoDataFrame(pd.concat(inter_sat_frames, ignore_index=True)).to_file(
            out_dir / f"merged_intersections_sat{suffix}.geojson", driver="GeoJSON"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="postopt_index.jsonl")
    ap.add_argument("--best-postopt", required=True, help="best_postopt.yaml")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        raise SystemExit(f"ERROR: index not found: {index_path}")

    best_candidate = _load_best_candidate(Path(args.best_postopt))
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
    _merge_layers(entries, "", out_dir)
    _merge_layers(entries, "_wgs84", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

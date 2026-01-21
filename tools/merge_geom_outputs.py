from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml
from shapely.geometry import shape, mapping
from shapely.ops import transform as geom_transform
from pyproj import Transformer


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


def _read_geojson_features(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("features", []) or []


def _write_geojson(path: Path, features: List[dict]) -> None:
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _reproject_features(features: List[dict], src_epsg: int, dst_epsg: int) -> List[dict]:
    transformer = Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)
    out = []
    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        shp = shape(geom)
        shp = geom_transform(transformer.transform, shp)
        out.append({"type": "Feature", "geometry": mapping(shp), "properties": feat.get("properties") or {}})
    return out


def _merge_layers(entries: List[dict], suffix: str, out_dir: Path, report: dict, intersections_root: Path | None = None) -> None:
    center_frames = []
    center_single_frames = []
    center_dual_frames = []
    center_both_frames = []
    road_frames = []
    inter_frames = []
    inter_algo_frames = []
    inter_sat_frames = []
    inter_final_frames = []

    def _note_missing(drive: str, layer: str, path: Path):
        report.setdefault("missing_layer_by_drive", []).append({"drive_id": drive, "layer": layer, "path": str(path)})

    def _note_empty(drive: str, layer: str, path: Path):
        report.setdefault("empty_layer_by_drive", []).append({"drive_id": drive, "layer": layer, "path": str(path)})

    def _resolve_path(outputs_dir: Path, run_id: str, name: str) -> Path:
        cand = outputs_dir / name
        if cand.exists():
            return cand
        if run_id:
            alt = Path("runs") / run_id / "outputs" / name
            if alt.exists():
                return alt
        return cand

    def _tag_features(
        feats: List[dict],
        drive: str,
        run_id: str,
        candidate_id: str,
        backend_used: str | None = None,
        sat_present: bool | None = None,
        sat_conf: float | None = None,
    ) -> List[dict]:
        out = []
        for feat in feats:
            props = feat.get("properties") or {}
            props["drive"] = drive
            props["drive_id"] = drive
            props["tile_id"] = drive
            props["geom_run_id"] = run_id
            props["candidate_id"] = candidate_id
            if backend_used is not None:
                props["backend_used"] = backend_used
            if sat_present is not None:
                props["sat_present"] = sat_present
            if sat_conf is not None:
                props["sat_confidence"] = sat_conf
            feat["properties"] = props
            out.append(feat)
        return out

    def _read_with_fallback(path: Path, fallback: Path, drive: str, layer: str, target_epsg: int | None):
        if path.exists():
            feats = _read_geojson_features(path)
            if not feats:
                _note_empty(drive, layer, path)
                return []
            return feats
        if fallback.exists() and target_epsg is not None:
            feats = _read_geojson_features(fallback)
            if not feats:
                _note_empty(drive, layer, fallback)
                return []
            feats = _reproject_features(feats, 32632, target_epsg)
            report.setdefault("generated_from_internal", []).append(
                {"drive_id": drive, "layer": layer, "source": str(fallback)}
            )
            return feats
        _note_missing(drive, layer, path)
        return None

    def _load_layer(path: Path, fallback: Path | None, drive: str, layer: str, target_epsg: int | None):
        if target_epsg is not None:
            return _read_with_fallback(path, fallback or path, drive, layer, target_epsg)
        if path.exists():
            feats = _read_geojson_features(path)
            if not feats:
                _note_empty(drive, layer, path)
                return []
            return feats
        _note_missing(drive, layer, path)
        return None

    for entry in entries:
        outputs_dir = Path(entry["outputs_dir"])
        drive = entry.get("drive") or ""
        run_id = entry.get("geom_run_id") or ""
        candidate_id = entry.get("candidate_id") or ""
        summary = _read_summary(outputs_dir)
        backend_used = summary.get("intersection_backend_used") or summary.get("backend_used") or ""
        sat_present = bool(summary.get("sat_present"))
        sat_conf = summary.get("sat_confidence_avg")

        center_path = _resolve_path(outputs_dir, run_id, f"centerlines{suffix}.geojson")
        center_single_path = _resolve_path(outputs_dir, run_id, f"centerlines_single{suffix}.geojson")
        center_dual_path = _resolve_path(outputs_dir, run_id, f"centerlines_dual{suffix}.geojson")
        center_both_path = _resolve_path(outputs_dir, run_id, f"centerlines_both{suffix}.geojson")
        road_path = _resolve_path(outputs_dir, run_id, f"road_polygon{suffix}.geojson")
        inter_dir = outputs_dir
        if intersections_root is not None:
            candidate = intersections_root / "outputs" / drive
            if candidate.exists():
                inter_dir = candidate
        inter_path = inter_dir / f"intersections{suffix}.geojson"
        inter_algo_path = inter_dir / f"intersections_algo{suffix}.geojson"
        inter_sat_path = inter_dir / f"intersections_sat{suffix}.geojson"
        inter_final_path = inter_dir / f"intersections_final{suffix}.geojson"
        feats = _load_layer(
            center_path,
            _resolve_path(outputs_dir, run_id, "centerlines.geojson"),
            drive,
            f"centerlines{suffix}",
            4326 if suffix == "_wgs84" else None,
        )
        if feats:
            center_frames.extend(_tag_features(feats, drive, run_id, candidate_id))

        feats = _load_layer(
            center_single_path,
            _resolve_path(outputs_dir, run_id, "centerlines_single.geojson"),
            drive,
            f"centerlines_single{suffix}",
            4326 if suffix == "_wgs84" else None,
        )
        if feats:
            center_single_frames.extend(_tag_features(feats, drive, run_id, candidate_id))

        feats = _load_layer(
            center_dual_path,
            _resolve_path(outputs_dir, run_id, "centerlines_dual.geojson"),
            drive,
            f"centerlines_dual{suffix}",
            4326 if suffix == "_wgs84" else None,
        )
        if feats:
            center_dual_frames.extend(_tag_features(feats, drive, run_id, candidate_id))

        feats = _load_layer(
            center_both_path,
            _resolve_path(outputs_dir, run_id, "centerlines_both.geojson"),
            drive,
            f"centerlines_both{suffix}",
            4326 if suffix == "_wgs84" else None,
        )
        if feats:
            center_both_frames.extend(_tag_features(feats, drive, run_id, candidate_id))

        feats = _load_layer(
            road_path,
            _resolve_path(outputs_dir, run_id, "road_polygon.geojson"),
            drive,
            f"road_polygon{suffix}",
            4326 if suffix == "_wgs84" else None,
        )
        if feats:
            road_frames.extend(_tag_features(feats, drive, run_id, candidate_id))

        feats = _load_layer(inter_path, None, drive, f"intersections{suffix}", None)
        if feats:
            inter_frames.extend(_tag_features(feats, drive, run_id, candidate_id, backend_used, sat_present, sat_conf))

        feats = _load_layer(inter_algo_path, None, drive, f"intersections_algo{suffix}", None)
        if feats:
            inter_algo_frames.extend(_tag_features(feats, drive, run_id, candidate_id, "algo", sat_present, sat_conf))

        feats = _load_layer(inter_sat_path, None, drive, f"intersections_sat{suffix}", None)
        if feats:
            inter_sat_frames.extend(_tag_features(feats, drive, run_id, candidate_id, "sat", sat_present, sat_conf))

        feats = _load_layer(inter_final_path, None, drive, f"intersections_final{suffix}", None)
        if feats:
            inter_final_frames.extend(_tag_features(feats, drive, run_id, candidate_id))

    def _coerce_conf(features: List[dict]) -> List[dict]:
        for feat in features:
            props = feat.get("properties") or {}
            if "conf" in props:
                try:
                    props["conf"] = float(props["conf"])
                except (TypeError, ValueError):
                    props["conf"] = None
            feat["properties"] = props
        return features

    def _write_filtered(features: List[dict], src: str, name: str) -> None:
        filtered = [f for f in features if (f.get("properties") or {}).get("src") == src]
        if filtered:
            _write_geojson(out_dir / name, filtered)

    if center_frames:
        _write_geojson(out_dir / f"merged_centerlines{suffix}.geojson", center_frames)
        report.setdefault("feature_count", {})[f"merged_centerlines{suffix}"] = int(len(center_frames))
    else:
        report.setdefault("empty_layer", []).append({"layer": f"merged_centerlines{suffix}"})
    if center_single_frames:
        _write_geojson(out_dir / f"merged_centerlines_single{suffix}.geojson", center_single_frames)
        report.setdefault("feature_count", {})[f"merged_centerlines_single{suffix}"] = int(len(center_single_frames))
    else:
        report.setdefault("empty_layer", []).append({"layer": f"merged_centerlines_single{suffix}"})
    if center_dual_frames:
        _write_geojson(out_dir / f"merged_centerlines_dual{suffix}.geojson", center_dual_frames)
        report.setdefault("feature_count", {})[f"merged_centerlines_dual{suffix}"] = int(len(center_dual_frames))
    else:
        report.setdefault("empty_layer", []).append({"layer": f"merged_centerlines_dual{suffix}"})
    if center_both_frames:
        _write_geojson(out_dir / f"merged_centerlines_both{suffix}.geojson", center_both_frames)
        report.setdefault("feature_count", {})[f"merged_centerlines_both{suffix}"] = int(len(center_both_frames))
    else:
        report.setdefault("empty_layer", []).append({"layer": f"merged_centerlines_both{suffix}"})
    if road_frames:
        _write_geojson(out_dir / f"merged_road_polygon{suffix}.geojson", road_frames)
        report.setdefault("feature_count", {})[f"merged_road_polygon{suffix}"] = int(len(road_frames))
    else:
        report.setdefault("empty_layer", []).append({"layer": f"merged_road_polygon{suffix}"})
    if inter_frames:
        _write_geojson(out_dir / f"merged_intersections{suffix}.geojson", inter_frames)
        report.setdefault("feature_count", {})[f"merged_intersections{suffix}"] = int(len(inter_frames))
    else:
        report.setdefault("empty_layer", []).append({"layer": f"merged_intersections{suffix}"})
    if inter_final_frames:
        merged_final = _coerce_conf(inter_final_frames)
        _write_geojson(out_dir / f"merged_intersections_final{suffix}.geojson", merged_final)
        report.setdefault("feature_count", {})[f"merged_intersections_final{suffix}"] = int(len(merged_final))
        _write_filtered(merged_final, "sat", f"merged_intersections_final_sat_only{suffix}.geojson")
        _write_filtered(merged_final, "algo", f"merged_intersections_final_algo_only{suffix}.geojson")
    elif inter_frames:
        merged = _coerce_conf(inter_frames)
        _write_geojson(out_dir / f"merged_intersections_final{suffix}.geojson", merged)
        report.setdefault("feature_count", {})[f"merged_intersections_final{suffix}"] = int(len(merged))
    else:
        report.setdefault("empty_layer", []).append({"layer": f"merged_intersections_final{suffix}"})
    if inter_algo_frames:
        _write_geojson(out_dir / f"merged_intersections_algo{suffix}.geojson", inter_algo_frames)
        report.setdefault("feature_count", {})[f"merged_intersections_algo{suffix}"] = int(len(inter_algo_frames))
    else:
        report.setdefault("empty_layer", []).append({"layer": f"merged_intersections_algo{suffix}"})
    if inter_sat_frames:
        _write_geojson(out_dir / f"merged_intersections_sat{suffix}.geojson", inter_sat_frames)
        report.setdefault("feature_count", {})[f"merged_intersections_sat{suffix}"] = int(len(inter_sat_frames))
    else:
        report.setdefault("empty_layer", []).append({"layer": f"merged_intersections_sat{suffix}"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="postopt_index.jsonl")
    ap.add_argument("--best-postopt", required=True, help="best_postopt.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--intersections-v2-dir", default="", help="optional intersections_v2 run dir")
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
    v2_root = Path(args.intersections_v2_dir) if args.intersections_v2_dir else None
    _merge_layers(entries, "", out_dir, report, v2_root)
    _merge_layers(entries, "_wgs84", out_dir, report, v2_root)
    report.setdefault("missing_layer", [])
    report.setdefault("empty_layer", [])
    report.setdefault("feature_count", {})
    report.setdefault("missing_layer_by_drive", [])
    report.setdefault("empty_layer_by_drive", [])
    report.setdefault("generated_from_internal", [])
    (out_dir / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

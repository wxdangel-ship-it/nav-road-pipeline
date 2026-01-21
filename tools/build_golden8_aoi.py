from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from pyproj import Transformer
except Exception:  # pragma: no cover - optional dependency
    Transformer = None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_coords(obj: object) -> Iterable[Tuple[float, float]]:
    if isinstance(obj, (list, tuple)):
        if len(obj) == 2 and all(isinstance(v, (int, float)) for v in obj):
            yield float(obj[0]), float(obj[1])
        else:
            for item in obj:
                yield from _iter_coords(item)


def _bbox_from_geojson(path: Path) -> Optional[Tuple[float, float, float, float]]:
    data = _read_json(path)
    feats = data.get("features") or []
    if not feats:
        return None
    xs: List[float] = []
    ys: List[float] = []
    for feat in feats:
        geom = (feat or {}).get("geometry") or {}
        for x, y in _iter_coords(geom.get("coordinates")):
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _merge_bbox(a: Optional[Tuple[float, float, float, float]], b: Optional[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _apply_margin(bbox: Tuple[float, float, float, float], margin: float) -> Tuple[float, float, float, float]:
    return bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin


def _bbox_to_polygon(bbox: Tuple[float, float, float, float]) -> dict:
    minx, miny, maxx, maxy = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [minx, miny],
                [maxx, miny],
                [maxx, maxy],
                [minx, maxy],
                [minx, miny],
            ]
        ],
    }


def _transform_bbox(
    bbox: Tuple[float, float, float, float],
    src_epsg: int,
    dst_epsg: int,
) -> Optional[Tuple[float, float, float, float]]:
    if Transformer is None:
        return None
    transformer = Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)
    minx, miny, maxx, maxy = bbox
    corners = [
        (minx, miny),
        (minx, maxy),
        (maxx, miny),
        (maxx, maxy),
    ]
    xs: List[float] = []
    ys: List[float] = []
    for x, y in corners:
        tx, ty = transformer.transform(x, y)
        xs.append(float(tx))
        ys.append(float(ty))
    return min(xs), min(ys), max(xs), max(ys)


def _read_index_bbox(path: Path) -> Optional[Tuple[float, float, float, float]]:
    if not path.exists():
        return None
    try:
        payload = _read_json(path)
    except json.JSONDecodeError:
        return None
    items = payload if isinstance(payload, list) else payload.get("items")
    if not isinstance(items, list) or not items:
        return None
    xs = [float(item["minx"]) for item in items] + [float(item["maxx"]) for item in items]
    ys = [float(item["miny"]) for item in items] + [float(item["maxy"]) for item in items]
    return min(xs), min(ys), max(xs), max(ys)


def _select_outputs(entries: List[dict]) -> Dict[str, Path]:
    picks: Dict[str, Path] = {}
    scores: Dict[str, int] = {}
    stage_rank = {"full": 2, "quick": 1}
    for entry in entries:
        drive = entry.get("drive") or entry.get("tile_id")
        out_dir = entry.get("outputs_dir")
        if not drive or not out_dir:
            continue
        if entry.get("status") != "PASS":
            continue
        score = stage_rank.get(entry.get("stage"), 0)
        if score <= scores.get(drive, -1):
            continue
        picks[drive] = Path(out_dir)
        scores[drive] = score
    return picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="runs/sweep_geom_postopt_20260119_061421/postopt_index.jsonl")
    ap.add_argument("--out-dir", default="runs/golden8_aoi")
    ap.add_argument("--margin-m", type=float, default=2000.0)
    ap.add_argument("--crs-epsg", type=int, default=32632)
    ap.add_argument("--dop20-root", default="")
    args = ap.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        print(f"[AOI] missing index: {index_path}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))
    drive_outputs = _select_outputs(entries)

    per_drive_bbox: Dict[str, Tuple[float, float, float, float]] = {}
    per_drive_wgs84: Dict[str, Tuple[float, float, float, float]] = {}
    union_bbox: Optional[Tuple[float, float, float, float]] = None

    for drive, out_dir_path in sorted(drive_outputs.items()):
        geom_dir = Path(out_dir_path)
        if not geom_dir.exists():
            continue
        utm_candidates = [
            geom_dir / "road_polygon.geojson",
            geom_dir / "intersections_algo.geojson",
            geom_dir / "centerlines.geojson",
        ]
        wgs_candidates = [
            geom_dir / "road_polygon_wgs84.geojson",
            geom_dir / "intersections_algo_wgs84.geojson",
            geom_dir / "centerlines_wgs84.geojson",
        ]
        bbox_utm = None
        for path in utm_candidates:
            if path.exists():
                bbox_utm = _bbox_from_geojson(path)
                if bbox_utm:
                    break
        bbox_wgs = None
        for path in wgs_candidates:
            if path.exists():
                bbox_wgs = _bbox_from_geojson(path)
                if bbox_wgs:
                    break

        if bbox_utm is None and bbox_wgs is not None:
            bbox_utm = _transform_bbox(bbox_wgs, 4326, args.crs_epsg)

        if bbox_wgs is None and bbox_utm is not None:
            bbox_wgs = _transform_bbox(bbox_utm, args.crs_epsg, 4326)

        if bbox_utm is None:
            print(f"[AOI] drive={drive} bbox_utm=missing")
            continue

        per_drive_bbox[drive] = bbox_utm
        if bbox_wgs is not None:
            per_drive_wgs84[drive] = bbox_wgs
        union_bbox = _merge_bbox(union_bbox, bbox_utm)
        print(f"[AOI] drive={drive} bbox_utm={bbox_utm[0]:.3f},{bbox_utm[1]:.3f},{bbox_utm[2]:.3f},{bbox_utm[3]:.3f}")

    if union_bbox is None:
        print("[AOI] no valid drive bbox found", file=sys.stderr)
        return 3

    union_bbox = _apply_margin(union_bbox, args.margin_m)
    union_wgs84 = _transform_bbox(union_bbox, args.crs_epsg, 4326)

    if union_wgs84 is None:
        print("[AOI] missing pyproj, cannot build WGS84 bbox", file=sys.stderr)
        return 4

    print(f"[AOI] union_bbox_utm={union_bbox[0]:.3f},{union_bbox[1]:.3f},{union_bbox[2]:.3f},{union_bbox[3]:.3f}")
    print(f"[AOI] union_bbox_wgs84={union_wgs84[0]:.6f},{union_wgs84[1]:.6f},{union_wgs84[2]:.6f},{union_wgs84[3]:.6f}")

    dop20_root = Path(args.dop20_root) if args.dop20_root else Path()
    if not dop20_root:
        import os

        env_root_str = os.getenv("DOP20_ROOT", "").strip()
        dop20_root = Path(env_root_str) if env_root_str else Path()
    index_paths = []
    if dop20_root:
        index_paths.append(dop20_root / "dop20_tiles_index.json")
    index_paths.append(Path("cache") / "dop20_tiles_index.json")
    for index_path in index_paths:
        index_bbox = _read_index_bbox(index_path)
        if not index_bbox:
            continue
        delta_miny = union_bbox[1] - index_bbox[1]
        print(
            "[AOI] dop20_index_bbox="
            f"{index_bbox[0]:.3f},{index_bbox[1]:.3f},{index_bbox[2]:.3f},{index_bbox[3]:.3f}"
        )
        print(f"[AOI] miny_delta_m={delta_miny:.3f}")
        break

    aoi_json = {
        "crs_utm": f"EPSG:{args.crs_epsg}",
        "margin_m": args.margin_m,
        "bbox_utm": {
            "minx": union_bbox[0],
            "miny": union_bbox[1],
            "maxx": union_bbox[2],
            "maxy": union_bbox[3],
        },
        "bbox_wgs84": {
            "minx": union_wgs84[0],
            "miny": union_wgs84[1],
            "maxx": union_wgs84[2],
            "maxy": union_wgs84[3],
        },
    }
    (out_dir / "golden8_aoi.json").write_text(json.dumps(aoi_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "golden8_aoi_utm32.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": _bbox_to_polygon(union_bbox), "properties": {}}]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "golden8_aoi_wgs84.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": _bbox_to_polygon(union_wgs84), "properties": {}}]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

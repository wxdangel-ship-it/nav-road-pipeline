import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from pyproj import Transformer
from shapely.geometry import Point, Polygon, shape, mapping
from shapely.ops import transform as geom_transform, unary_union


def _read_geojson(path: Path) -> List[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("features", []) or []


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _is_wgs84_coords(x: float, y: float) -> bool:
    return -180.0 <= x <= 180.0 and -90.0 <= y <= 90.0


def _to_utm32(geom):
    wgs84 = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    return geom_transform(wgs84.transform, geom)


def _to_wgs84(geom):
    wgs84 = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    return geom_transform(wgs84.transform, geom)


def _load_road_polygon(path: Path) -> Polygon:
    feats = _read_geojson(path)
    polys = []
    for feat in feats:
        geom = feat.get("geometry")
        if not geom:
            continue
        shp = shape(geom)
        if shp.is_empty:
            continue
        polys.append(shp)
    if not polys:
        raise SystemExit("ERROR: empty road_polygon")
    return unary_union(polys)


def _bbox_polygon(minx: float, miny: float, maxx: float, maxy: float) -> Polygon:
    return Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)])


def _load_missing_junctions(
    csv_path: Path,
    status_path: Path,
) -> List[Tuple[str, Point, dict]]:
    out = []
    rows = _read_csv(csv_path) if csv_path.exists() else []
    if rows:
        for row in rows:
            if row.get("drop_stage") != "road_local_empty":
                continue
            try:
                lon = float(row.get("lon") or 0.0)
                lat = float(row.get("lat") or 0.0)
            except ValueError:
                continue
            pt = Point(lon, lat)
            out.append((row.get("junction_id") or "", pt, row))
        return out

    feats = _read_geojson(status_path)
    for feat in feats:
        geom = feat.get("geometry")
        if not geom:
            continue
        props = feat.get("properties") or {}
        if props.get("drop_stage") and props.get("drop_stage") != "road_local_empty":
            continue
        pt = shape(geom)
        if pt.is_empty:
            continue
        out.append((props.get("junction_id") or "", pt, props))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--road-poly", required=True)
    parser.add_argument("--junction-csv", required=False)
    parser.add_argument("--junction-status", required=False)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    road_poly = _load_road_polygon(Path(args.road_poly))
    minx, miny, maxx, maxy = road_poly.bounds
    road_poly_wgs = road_poly if _is_wgs84_coords(minx, miny) else _to_wgs84(road_poly)
    road_poly_utm = road_poly if not _is_wgs84_coords(minx, miny) else _to_utm32(road_poly)

    bbox_poly = _bbox_polygon(*road_poly_wgs.bounds)
    bbox_feat = {"type": "Feature", "geometry": mapping(bbox_poly), "properties": {}}
    (out_dir / "road_polygon_0007_bbox_wgs84.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [bbox_feat]}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    csv_path = Path(args.junction_csv) if args.junction_csv else out_dir / "junction_diagnose_0007.csv"
    status_path = Path(args.junction_status) if args.junction_status else out_dir / "junction_status_wgs84.geojson"

    missing = _load_missing_junctions(csv_path, status_path)
    radii = [20, 30, 40, 60]
    summary_lines = []
    summary_lines.append("road_local_empty_audit_0007")
    summary_lines.append("")

    for junction_id, pt_wgs, _ in missing:
        pt_utm = _to_utm32(pt_wgs)
        area_by_r: Dict[int, float] = {}
        features = []
        for r in radii:
            probe = pt_utm.buffer(float(r))
            inter = road_poly_utm.intersection(probe)
            area = float(inter.area) if inter is not None and not inter.is_empty else 0.0
            area_by_r[r] = area
            probe_wgs = _to_wgs84(probe)
            inter_wgs = _to_wgs84(inter) if inter is not None and not inter.is_empty else None
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(probe_wgs),
                    "properties": {"junction_id": junction_id, "layer": f"probe_circle_{r}m"},
                }
            )
            if inter_wgs is not None:
                features.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(inter_wgs),
                        "properties": {"junction_id": junction_id, "layer": f"road_local_intersection_{r}m"},
                    }
                )

        tag = "covered_at_small_radius"
        if area_by_r[20] == 0.0 and area_by_r[30] == 0.0 and area_by_r[60] > 0.0:
            tag = "radius_mismatch"
        elif area_by_r[60] == 0.0:
            tag = "road_polygon_missing"

        summary_lines.append(
            f"{junction_id}: R20={area_by_r[20]:.3f}, R30={area_by_r[30]:.3f}, "
            f"R40={area_by_r[40]:.3f}, R60={area_by_r[60]:.3f} => {tag}"
        )

        probe_path = out_dir / f"road_local_probe_{junction_id}.geojson"
        probe_path.write_text(
            json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.etree.ElementTree import iterparse

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, shape, mapping, box, Point
from shapely.ops import unary_union, transform as shapely_transform
from pyproj import Transformer


def _resolve_outputs_dir(path: Path) -> Path:
    if path.name == "outputs":
        return path
    candidate = path / "outputs"
    if candidate.exists():
        return candidate
    return path


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_geojson(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("features", [])


def _geom_to_lines(geom) -> List[LineString]:
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    if isinstance(geom, Polygon):
        return [geom.boundary]
    if isinstance(geom, MultiPolygon):
        return [g.boundary for g in geom.geoms]
    return []


def _find_osm_source(osm_root: Path) -> Tuple[Optional[Path], Optional[str]]:
    if not osm_root.exists():
        return None, "osm_root_missing"
    drivable = osm_root / "drivable_roads.geojson"
    if drivable.exists():
        return drivable, None
    osm_full = osm_root / "osm_full.osm"
    if osm_full.exists():
        return osm_full, None
    geojsons = list(osm_root.rglob("*.geojson"))
    if geojsons:
        geojsons.sort(key=lambda p: (0 if "roads" in p.name.lower() else 1, len(str(p))))
        return geojsons[0], None
    return None, "osm_source_missing"


def _bbox_from_features(features: List[Dict[str, Any]]) -> Optional[Tuple[float, float, float, float]]:
    geoms = [shape(f["geometry"]) for f in features if f.get("geometry")]
    if not geoms:
        return None
    unioned = unary_union(geoms)
    return unioned.bounds


def _read_aoi_bbox(path: Path) -> Optional[Tuple[float, float, float, float]]:
    if not path.exists():
        return None
    data = _read_json(path)
    if not data:
        return None
    if "bbox" in data and isinstance(data["bbox"], list) and len(data["bbox"]) == 4:
        return tuple(float(v) for v in data["bbox"])
    keys = ("min_lon", "min_lat", "max_lon", "max_lat")
    if all(k in data for k in keys):
        return (float(data["min_lon"]), float(data["min_lat"]), float(data["max_lon"]), float(data["max_lat"]))
    return None


def _strip_tag(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _iter_osm_roads(osm_path: Path, bbox: Optional[Tuple[float, float, float, float]]) -> List[Dict[str, Any]]:
    # pass 1: collect node refs used by highway ways
    refs: set[str] = set()
    for _, elem in iterparse(str(osm_path), events=("end",)):
        if _strip_tag(elem.tag) == "way":
            tags: Dict[str, str] = {}
            nd_refs: List[str] = []
            for child in list(elem):
                name = _strip_tag(child.tag)
                if name == "tag":
                    k = child.get("k")
                    v = child.get("v")
                    if k and v:
                        tags[k] = v
                elif name == "nd":
                    ref = child.get("ref")
                    if ref:
                        nd_refs.append(ref)
            if tags.get("highway"):
                refs.update(nd_refs)
            elem.clear()

    # pass 2: collect node coordinates for referenced nodes
    nodes: Dict[str, Tuple[float, float]] = {}
    for _, elem in iterparse(str(osm_path), events=("end",)):
        if _strip_tag(elem.tag) == "node":
            node_id = elem.get("id")
            if node_id and node_id in refs:
                lat = elem.get("lat")
                lon = elem.get("lon")
                if lat and lon:
                    nodes[node_id] = (float(lon), float(lat))
            elem.clear()

    # pass 3: build highways from cached nodes
    features: List[Dict[str, Any]] = []
    for _, elem in iterparse(str(osm_path), events=("end",)):
        if _strip_tag(elem.tag) == "way":
            tags: Dict[str, str] = {}
            coords: List[Tuple[float, float]] = []
            for child in list(elem):
                name = _strip_tag(child.tag)
                if name == "tag":
                    k = child.get("k")
                    v = child.get("v")
                    if k and v:
                        tags[k] = v
                elif name == "nd":
                    ref = child.get("ref")
                    if ref in nodes:
                        coords.append(nodes[ref])
            highway = tags.get("highway")
            if highway and len(coords) >= 2:
                line = LineString(coords)
                features.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(line),
                        "properties": {"highway": highway},
                    }
                )
            elem.clear()
    return features


def _load_osm_features(osm_root: Path, source_path: Path, bbox: Optional[Tuple[float, float, float, float]]) -> List[Dict[str, Any]]:
    cache_path = osm_root / "osm_roads_cache.geojson"
    if cache_path.exists() and source_path.suffix.lower() == ".osm":
        cached = _load_geojson(cache_path)
        if cached:
            return cached
    if source_path.suffix.lower() == ".osm":
        features = _iter_osm_roads(source_path, bbox=bbox)
        cache_path.write_text(
            json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return features
    return _load_geojson(source_path)


def _sample_points(line: LineString, step_m: float) -> Iterable[Point]:
    if line.length == 0:
        return []
    count = max(1, int(line.length / step_m) + 1)
    return [line.interpolate(i * step_m) for i in range(count)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True, help="geom outputs dir or geom run dir")
    ap.add_argument("--buffer-m", type=float, default=10.0, help="OSM buffer distance (m)")
    ap.add_argument("--sample-step-m", type=float, default=5.0, help="sampling step along centerlines (m)")
    ap.add_argument("--osm-root", default="", help="optional OSM root (default POC_DATA_ROOT/_osm_download)")
    args = ap.parse_args()

    outputs_dir = _resolve_outputs_dir(Path(args.outputs_dir))
    road_wgs84 = outputs_dir / "road_polygon_wgs84.geojson"
    center_wgs84 = outputs_dir / "centerlines_wgs84.geojson"
    if not center_wgs84.exists():
        print(f"[OSM-REF] WARN: missing {center_wgs84}")
        return 0

    crs = _read_json(outputs_dir / "crs.json") or {}
    internal_epsg = int(crs.get("internal_epsg") or 32632)

    osm_root = Path(args.osm_root) if args.osm_root else None
    if osm_root is None:
        data_root = os.environ.get("POC_DATA_ROOT", "")
        osm_root = Path(os.environ.get("OSM_ROOT", "")) if os.environ.get("OSM_ROOT") else None
        if osm_root is None:
            osm_root = Path(data_root) / "_osm_download" if data_root else Path()
    source_path, source_err = _find_osm_source(osm_root)
    aoi_bbox = _read_aoi_bbox(osm_root / "aoi_bbox.json")

    ref_path = outputs_dir / "osm_ref_roads.geojson"
    metrics_path = outputs_dir / "osm_ref_metrics.json"
    buffer_m = float(os.environ.get("OSM_BUFFER_M", args.buffer_m))
    ref_osm = {
        "osm_present": False,
        "coverage_ok": False,
        "metrics_valid": False,
        "osm_source": str(source_path) if source_path else None,
        "buffer_m": buffer_m,
        "match_ratio": None,
        "dist_p50_m": None,
        "dist_p95_m": None,
        "sample_step_m": float(args.sample_step_m),
    }

    if source_path is None:
        if source_err:
            print(f"[OSM-REF] WARN: {source_err}")
        _write_json(ref_path, {"type": "FeatureCollection", "features": []})
        _write_json(metrics_path, ref_osm)
        return 0

    bbox = None
    if road_wgs84.exists():
        bbox = _bbox_from_features(_load_geojson(road_wgs84))
    if bbox is None:
        bbox = _bbox_from_features(_load_geojson(center_wgs84))

    ref_osm["osm_present"] = True
    osm_features = _load_osm_features(osm_root, source_path, bbox=bbox or aoi_bbox)
    if not osm_features:
        print("[OSM-REF] WARN: empty OSM geojson")
        _write_json(ref_path, {"type": "FeatureCollection", "features": []})
        _write_json(metrics_path, ref_osm)
        return 0
    bbox_poly = box(*bbox) if bbox else None
    if bbox_poly is not None:
        to_internal = Transformer.from_crs("EPSG:4326", f"EPSG:{internal_epsg}", always_xy=True)
        to_wgs84 = Transformer.from_crs(f"EPSG:{internal_epsg}", "EPSG:4326", always_xy=True)

        def _to_internal(x, y, z=None):
            xx, yy = to_internal.transform(x, y)
            if z is None:
                return xx, yy
            return xx, yy, z

        def _to_wgs84(x, y, z=None):
            xx, yy = to_wgs84.transform(x, y)
            if z is None:
                return xx, yy
            return xx, yy, z

        bbox_poly = shapely_transform(_to_wgs84, shapely_transform(_to_internal, bbox_poly).buffer(buffer_m))

    clipped_features = []
    osm_lines_wgs84: List[LineString] = []
    for feat in osm_features:
        geom = feat.get("geometry")
        if geom is None:
            continue
        shp = shape(geom)
        if bbox_poly is not None and not bbox_poly.intersects(shp):
            continue
        clipped = shp.intersection(bbox_poly) if bbox_poly is not None else shp
        if clipped.is_empty:
            continue
        for line in _geom_to_lines(clipped):
            osm_lines_wgs84.append(line)
        clipped_features.append(
            {
                "type": "Feature",
                "geometry": mapping(clipped),
                "properties": feat.get("properties", {}),
            }
        )

    _write_json(ref_path, {"type": "FeatureCollection", "features": clipped_features})

    if not osm_lines_wgs84:
        allow_full = os.environ.get("OSM_ALLOW_FULL_METRICS", "0") == "1"
        if allow_full:
            print("[OSM-REF] WARN: no OSM road lines after clipping; using full OSM for metrics")
            for feat in osm_features:
                geom = feat.get("geometry")
                if geom is None:
                    continue
                osm_lines_wgs84.extend(_geom_to_lines(shape(geom)))
        else:
            print("[OSM-REF] WARN: no OSM road lines after clipping")

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{internal_epsg}", always_xy=True)

    def _proj(x, y, z=None):
        xx, yy = transformer.transform(x, y)
        if z is None:
            return xx, yy
        return xx, yy, z

    osm_lines_proj = [shapely_transform(_proj, line) for line in osm_lines_wgs84]
    osm_union = unary_union(osm_lines_proj) if osm_lines_proj else None

    center_features = _load_geojson(center_wgs84)
    center_lines_proj = []
    for feat in center_features:
        geom = feat.get("geometry")
        if geom is None:
            continue
        shp = shape(geom)
        center_lines_proj.extend(_geom_to_lines(shapely_transform(_proj, shp)))

    ref_osm["coverage_ok"] = bool(osm_lines_wgs84)
    ref_osm["metrics_valid"] = bool(center_lines_proj and osm_union and not osm_union.is_empty)
    if not ref_osm["metrics_valid"]:
        _write_json(metrics_path, ref_osm)
        return 0

    sample_step = float(args.sample_step_m)
    osm_buffer = osm_union.buffer(buffer_m)
    distances: List[float] = []
    matched_len = 0.0
    total_len = 0.0
    for line in center_lines_proj:
        total_len += float(line.length)
        matched_len += float(line.intersection(osm_buffer).length)
        for pt in _sample_points(line, sample_step):
            dist = float(pt.distance(osm_union))
            distances.append(dist)

    if total_len > 0 and distances:
        dist_arr = np.asarray(distances, dtype=float)
        ref_osm["match_ratio"] = round(float(matched_len) / float(total_len), 4)
        ref_osm["dist_p50_m"] = round(float(np.percentile(dist_arr, 50)), 3)
        ref_osm["dist_p95_m"] = round(float(np.percentile(dist_arr, 95)), 3)

    _write_json(metrics_path, ref_osm)
    print(f"[OSM-REF] wrote {ref_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

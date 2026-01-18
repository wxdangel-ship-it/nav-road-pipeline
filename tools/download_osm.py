import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def _find_latest_geojson(runs_root: Path) -> Path | None:
    candidates = []
    for path in runs_root.rglob("road_polygon_wgs84.geojson"):
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _bbox_from_geojson(path: Path) -> tuple[float, float, float, float]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    minx = miny = 1e30
    maxx = maxy = -1e30

    def update_coords(coords):
        nonlocal minx, miny, maxx, maxy
        for c in coords:
            if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
                x, y = c[:2]
                minx = min(minx, x)
                miny = min(miny, y)
                maxx = max(maxx, x)
                maxy = max(maxy, y)
            else:
                update_coords(c)

    for feat in obj.get("features", []):
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        if coords is not None:
            update_coords(coords)

    if minx > maxx or miny > maxy:
        raise ValueError(f"no coordinates found in {path}")
    return minx, miny, maxx, maxy


def _expand_bbox(
    bbox: tuple[float, float, float, float], margin_m: float
) -> tuple[float, float, float, float]:
    if margin_m <= 0:
        return bbox
    minx, miny, maxx, maxy = bbox
    center_lat = (miny + maxy) / 2.0
    deg_lat = margin_m / 111320.0
    deg_lon = margin_m / (111320.0 * math.cos(math.radians(center_lat)))
    return (minx - deg_lon, miny - deg_lat, maxx + deg_lon, maxy + deg_lat)


def _area_km2(bbox: tuple[float, float, float, float]) -> float:
    minx, miny, maxx, maxy = bbox
    center_lat = (miny + maxy) / 2.0
    dx_m = (maxx - minx) * 111320.0 * math.cos(math.radians(center_lat))
    dy_m = (maxy - miny) * 111320.0
    return abs(dx_m * dy_m) / 1e6


def _write_aoi_bbox(path: Path, raw_bbox, expanded_bbox, margin_m, timeout_sec):
    data = {
        "bbox_wgs84_raw": {
            "min_lon": raw_bbox[0],
            "min_lat": raw_bbox[1],
            "max_lon": raw_bbox[2],
            "max_lat": raw_bbox[3],
        },
        "bbox_wgs84_expanded": {
            "min_lon": expanded_bbox[0],
            "min_lat": expanded_bbox[1],
            "max_lon": expanded_bbox[2],
            "max_lat": expanded_bbox[3],
        },
        "margin_m": margin_m,
        "area_km2_est": _area_km2(expanded_bbox),
        "overpass_endpoints": [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.nchc.org.tw/api/interpreter",
        ],
        "timeout_sec": timeout_sec,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _download_osm(osm_root: Path, bbox, timeout_sec: int) -> int:
    minx, miny, maxx, maxy = bbox
    query = (
        f'[out:xml][timeout:{timeout_sec}];'
        f'(way["highway"]({miny},{minx},{maxy},{maxx});>;);'
        "out body;"
    )
    out_path = osm_root / "osm_full.osm"
    cmd = [
        "curl.exe",
        "-s",
        "-o",
        str(out_path),
        "--data-urlencode",
        f"data={query}",
        "https://overpass-api.de/api/interpreter",
    ]
    result = subprocess.run(cmd)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download OSM highway ways for current experiment area."
    )
    parser.add_argument(
        "--osm-root",
        default=r"E:\KITTI360\KITTI-360\_osm_download",
        help="OSM output root.",
    )
    parser.add_argument(
        "--runs-root",
        default=str(Path(__file__).resolve().parents[1] / "runs"),
        help="Runs root to auto-discover latest road_polygon_wgs84.",
    )
    parser.add_argument(
        "--road-polygon",
        default="",
        help="Path to road_polygon_wgs84.geojson (overrides auto-discovery).",
    )
    parser.add_argument("--minlon", type=float, default=None)
    parser.add_argument("--minlat", type=float, default=None)
    parser.add_argument("--maxlon", type=float, default=None)
    parser.add_argument("--maxlat", type=float, default=None)
    parser.add_argument("--margin-m", type=float, default=800.0)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep existing osm_roads_cache.geojson.",
    )
    args = parser.parse_args()

    osm_root = Path(args.osm_root)
    osm_root.mkdir(parents=True, exist_ok=True)

    if (
        args.minlon is not None
        or args.minlat is not None
        or args.maxlon is not None
        or args.maxlat is not None
    ):
        if None in (args.minlon, args.minlat, args.maxlon, args.maxlat):
            print("minlon/minlat/maxlon/maxlat must be all provided.", file=sys.stderr)
            return 2
        raw_bbox = (args.minlon, args.minlat, args.maxlon, args.maxlat)
        source = "bbox args"
    else:
        if args.road_polygon:
            geojson_path = Path(args.road_polygon)
        else:
            runs_root = Path(args.runs_root)
            geojson_path = _find_latest_geojson(runs_root)
        if not geojson_path or not geojson_path.exists():
            print("No road_polygon_wgs84.geojson found.", file=sys.stderr)
            return 2
        raw_bbox = _bbox_from_geojson(geojson_path)
        source = str(geojson_path)

    expanded_bbox = _expand_bbox(raw_bbox, args.margin_m)
    print(f"bbox_raw: {raw_bbox}")
    print(f"bbox_expanded: {expanded_bbox}")
    print(f"source: {source}")

    aoi_bbox_path = osm_root / "aoi_bbox.json"
    _write_aoi_bbox(aoi_bbox_path, raw_bbox, expanded_bbox, args.margin_m, args.timeout_sec)
    print(f"wrote: {aoi_bbox_path}")

    cache_path = osm_root / "osm_roads_cache.geojson"
    if cache_path.exists() and not args.keep_cache:
        cache_path.unlink()
        print(f"removed cache: {cache_path}")

    rc = _download_osm(osm_root, expanded_bbox, args.timeout_sec)
    if rc != 0:
        print("OSM download failed. Check network access.", file=sys.stderr)
        return rc

    out_path = osm_root / "osm_full.osm"
    try:
        size = out_path.stat().st_size
    except OSError:
        size = -1
    print(f"downloaded: {out_path} size={size}")
    if size > 0 and size < 1024:
        print("warning: osm_full.osm is very small; check Overpass response.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

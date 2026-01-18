import argparse
import datetime
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from pyproj import Transformer


def _find_oxts_dir(data_root: Path, drive: str) -> Path:
    candidates = [
        data_root / "data_poses_oxts_extract" / drive / "oxts" / "data",
        data_root / "data_poses_oxts" / drive / "oxts" / "data",
        data_root / "data_poses" / drive / "oxts" / "data",
        data_root / "data_poses" / "oxts" / drive / "oxts" / "data",
        data_root / "data_poses" / "oxts" / drive / "data",
        data_root / drive / "oxts" / "data",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"ERROR: oxts data not found for drive: {drive}")


def _read_latlon(oxts_dir: Path, stride: int) -> List[Tuple[float, float]]:
    files = sorted(oxts_dir.glob("*.txt"))
    if stride > 1:
        files = files[::stride]
    if not files:
        raise SystemExit(f"ERROR: no oxts txt files found in {oxts_dir}")
    pts: List[Tuple[float, float]] = []
    for fp in files:
        text = fp.read_text(encoding="utf-8").strip()
        if not text:
            continue
        parts = re.split(r"\s+", text)
        if len(parts) < 2:
            continue
        try:
            lat = float(parts[0])
            lon = float(parts[1])
        except ValueError:
            continue
        pts.append((lat, lon))
    if len(pts) < 2:
        raise SystemExit(f"ERROR: insufficient oxts points in {oxts_dir}")
    return pts


def _bbox_from_points(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return (min(lons), min(lats), max(lons), max(lats))


def _expand_bbox_utm32(
    bbox: Tuple[float, float, float, float], margin_m: float
) -> Tuple[float, float, float, float]:
    if margin_m <= 0:
        return bbox
    fwd = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    inv = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    minlon, minlat, maxlon, maxlat = bbox
    corners = [
        (minlon, minlat),
        (minlon, maxlat),
        (maxlon, minlat),
        (maxlon, maxlat),
    ]
    xs = []
    ys = []
    for lon, lat in corners:
        x, y = fwd.transform(lon, lat)
        xs.append(x)
        ys.append(y)
    minx = min(xs) - margin_m
    maxx = max(xs) + margin_m
    miny = min(ys) - margin_m
    maxy = max(ys) + margin_m
    minlon2, minlat2 = inv.transform(minx, miny)
    maxlon2, maxlat2 = inv.transform(maxx, maxy)
    return (minlon2, minlat2, maxlon2, maxlat2)


def _discover_drives(data_root: Path) -> List[str]:
    base = data_root / "data_3d_raw"
    if not base.exists():
        raise SystemExit(f"ERROR: data_3d_raw not found under {data_root}")
    drives = [p.name for p in base.glob("*_drive_*_sync") if p.is_dir()]
    drives.sort()
    return drives


def _write_outputs(
    out_dir: Path,
    drives: List[str],
    per_drive: Dict[str, Tuple[float, float, float, float]],
    raw_bbox: Tuple[float, float, float, float],
    expanded_bbox: Tuple[float, float, float, float],
    margin_m: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    minlon, minlat, maxlon, maxlat = expanded_bbox

    payload = {
        "drives": drives,
        "margin_m": float(margin_m),
        "bbox_wgs84_raw": {
            "min_lon": raw_bbox[0],
            "min_lat": raw_bbox[1],
            "max_lon": raw_bbox[2],
            "max_lat": raw_bbox[3],
        },
        "bbox_wgs84_expanded": {
            "min_lon": minlon,
            "min_lat": minlat,
            "max_lon": maxlon,
            "max_lat": maxlat,
        },
        "per_drive_bbox": {
            k: {"min_lon": v[0], "min_lat": v[1], "max_lon": v[2], "max_lat": v[3]}
            for k, v in per_drive.items()
        },
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "aoi_bbox.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    polygon = [
        [minlon, minlat],
        [minlon, maxlat],
        [maxlon, maxlat],
        [maxlon, minlat],
        [minlon, minlat],
    ]
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [polygon]},
                "properties": {"margin_m": float(margin_m)},
            }
        ],
    }
    (out_dir / "aoi_bbox_wgs84.geojson").write_text(
        json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    line = f"--minlon {minlon} --minlat {minlat} --maxlon {maxlon} --maxlat {maxlat}"
    (out_dir / "aoi_bbox.txt").write_text(line + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate AOI bbox from KITTI-360 OXTS.")
    ap.add_argument("--data-root", default="", help="POC data root")
    ap.add_argument("--margin-m", type=float, default=1000.0, help="margin in meters")
    ap.add_argument("--stride", type=int, default=10, help="oxts stride")
    ap.add_argument("--out-dir", default="", help="output directory")
    ap.add_argument(
        "--drives",
        default="",
        help="comma-separated drive list (optional)",
    )
    args = ap.parse_args()

    data_root = args.data_root or os.environ.get("POC_DATA_ROOT", "")
    if not data_root:
        data_root = r"E:\KITTI360\KITTI-360"
    data_root_path = Path(data_root)
    if not data_root_path.exists():
        raise SystemExit(f"ERROR: data root not found: {data_root}")

    if args.drives:
        drives = [d.strip() for d in args.drives.split(",") if d.strip()]
    else:
        drives = _discover_drives(data_root_path)
    if not drives:
        raise SystemExit("ERROR: no drives found.")

    per_drive: Dict[str, Tuple[float, float, float, float]] = {}
    for drive in drives:
        oxts_dir = _find_oxts_dir(data_root_path, drive)
        pts = _read_latlon(oxts_dir, stride=max(1, int(args.stride)))
        per_drive[drive] = _bbox_from_points(pts)

    raw_bbox = (
        min(v[0] for v in per_drive.values()),
        min(v[1] for v in per_drive.values()),
        max(v[2] for v in per_drive.values()),
        max(v[3] for v in per_drive.values()),
    )
    expanded_bbox = _expand_bbox_utm32(raw_bbox, float(args.margin_m))

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).resolve().parents[1] / "runs" / f"aoi_{ts}"

    _write_outputs(out_dir, drives, per_drive, raw_bbox, expanded_bbox, float(args.margin_m))

    print(f"[AOI] out_dir={out_dir}")
    print(f"[AOI] bbox_raw={raw_bbox}")
    print(f"[AOI] bbox_expanded={expanded_bbox}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

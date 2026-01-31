from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import geopandas as gpd
except Exception:
    gpd = None


def _write_png(path: Path, arr: np.ndarray) -> None:
    if Image is None:
        raise RuntimeError("PIL is required to write PNG masks.")
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(path)


def _make_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 2 - 2 : h // 2 + 2, 10 : w - 10] = 2  # divider_median
    mask[20:24, 10 : w - 10] = 1  # lane_marking
    mask[h - 24 : h - 20, 10 : w - 10] = 1  # lane_marking
    mask[60:80, 30:70] = 5  # crosswalk
    mask[90:110, 30:70] = 6  # gore_marking
    mask[40:50, 40:60] = 4  # stop_line
    mask[120:140, 40:60] = 7  # arrow
    return mask


def _write_mock_vector(path: Path, road_polygon_path: Path) -> None:
    if gpd is None:
        return
    gdf = gpd.read_file(road_polygon_path)
    if gdf.empty:
        return
    poly = gdf.geometry.iloc[0]
    minx, miny, maxx, maxy = poly.bounds
    midy = (miny + maxy) * 0.5
    divider = LineString([(minx + 5, midy), (maxx - 5, midy)])
    out = gpd.GeoDataFrame([{"class": "divider_median"}], geometry=[divider])
    out.to_file(path, driver="GeoJSON")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--drive", default="2013_05_28_drive_0007_sync")
    ap.add_argument("--frame-id", default="0000000000")
    ap.add_argument("--road-polygon", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    seg_dir = out_dir / "seg_masks"
    det_dir = out_dir / "det_outputs"
    seg_dir.mkdir(parents=True, exist_ok=True)
    det_dir.mkdir(parents=True, exist_ok=True)

    mask = _make_mask(160, 160)
    npy_path = seg_dir / f"{args.drive}_{args.frame_id}_seg.npy"
    png_path = seg_dir / f"{args.drive}_{args.frame_id}_seg.png"
    np.save(npy_path, mask)
    _write_png(png_path, mask)

    det = [
        {
            "class": "traffic_light",
            "bbox": [20, 20, 40, 60],
            "conf": 0.92,
            "track_id": 1,
            "frame_id": args.frame_id,
        },
        {
            "class": "arrow",
            "bbox": [50, 120, 80, 145],
            "conf": 0.88,
            "frame_id": args.frame_id,
        },
    ]
    det_path = det_dir / f"{args.drive}_{args.frame_id}_det.json"
    det_path.write_text(json.dumps(det, indent=2), encoding="utf-8")

    if args.road_polygon:
        _write_mock_vector(out_dir / "divider_map_hint.geojson", Path(args.road_polygon))

    print("[MOCK] wrote:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode

import requests

try:
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover - optional dependency
    CRS = None
    Transformer = None

try:
    from rasterio.warp import transform_bounds
except Exception:  # pragma: no cover - optional dependency
    transform_bounds = None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _transform_bbox(
    bbox: Tuple[float, float, float, float],
    src_crs: str,
    dst_crs: str,
) -> Tuple[float, float, float, float]:
    if transform_bounds is not None:
        minx, miny, maxx, maxy = transform_bounds(src_crs, dst_crs, *bbox, densify_pts=21)
        return float(minx), float(miny), float(maxx), float(maxy)
    if Transformer is None:
        raise RuntimeError("missing_transformer")
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
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


def _bbox_from_aoi(aoi_path: Path, target_crs: str) -> Tuple[float, float, float, float]:
    payload = _read_json(aoi_path)
    bbox_utm = payload.get("bbox_utm")
    bbox_wgs = payload.get("bbox_wgs84")
    crs_utm = payload.get("crs_utm")
    if bbox_utm and crs_utm:
        bbox = (
            float(bbox_utm["minx"]),
            float(bbox_utm["miny"]),
            float(bbox_utm["maxx"]),
            float(bbox_utm["maxy"]),
        )
        if crs_utm == target_crs:
            return bbox
        return _transform_bbox(bbox, crs_utm, target_crs)
    if bbox_wgs:
        bbox = (
            float(bbox_wgs["minx"]),
            float(bbox_wgs["miny"]),
            float(bbox_wgs["maxx"]),
            float(bbox_wgs["maxy"]),
        )
        return _transform_bbox(bbox, "EPSG:4326", target_crs)
    raise RuntimeError("missing_bbox")


def _align_bbox(
    bbox: Tuple[float, float, float, float],
    tile_size: float,
) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bbox
    minx = math.floor(minx / tile_size) * tile_size
    miny = math.floor(miny / tile_size) * tile_size
    maxx = math.ceil(maxx / tile_size) * tile_size
    maxy = math.ceil(maxy / tile_size) * tile_size
    return minx, miny, maxx, maxy


def _axis_order(crs: str) -> str:
    if CRS is None:
        return "xy"
    try:
        axis_info = CRS.from_user_input(crs).axis_info
    except Exception:
        return "xy"
    if not axis_info:
        return "xy"
    first = axis_info[0].direction.lower()
    if first in {"north", "up"}:
        return "yx"
    return "xy"


def _wms_bbox(
    bbox: Tuple[float, float, float, float],
    crs: str,
) -> str:
    minx, miny, maxx, maxy = bbox
    if _axis_order(crs) == "yx":
        return f"{miny},{minx},{maxy},{maxx}"
    return f"{minx},{miny},{maxx},{maxy}"


def _write_world_file(path: Path, bbox: Tuple[float, float, float, float], width: int, height: int) -> None:
    minx, miny, maxx, maxy = bbox
    res_x = (maxx - minx) / float(width)
    res_y = -(maxy - miny) / float(height)
    top_left_x = minx + res_x / 2.0
    top_left_y = maxy + res_y / 2.0
    lines = [
        f"{res_x:.10f}",
        "0.0",
        "0.0",
        f"{res_y:.10f}",
        f"{top_left_x:.10f}",
        f"{top_left_y:.10f}",
    ]
    path.write_text("\n".join(lines), encoding="ascii")


def _get_capabilities(wms_url: str, timeout_sec: int) -> str:
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetCapabilities",
        "VERSION": "1.3.0",
    }
    resp = requests.get(wms_url, params=params, timeout=timeout_sec)
    resp.raise_for_status()
    return resp.text


def _extract_layers(xml_text: str) -> List[str]:
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    layers = []
    for elem in root.findall(".//{*}Layer/{*}Name"):
        if elem.text:
            layers.append(elem.text.strip())
    return layers


def _select_layer(layers: List[str]) -> Optional[str]:
    if not layers:
        return None
    for name in layers:
        upper = name.upper()
        if "DOP" in upper and "RGB" in upper:
            return name
    for name in layers:
        if "DOP" in name.upper():
            return name
    return layers[0]


def _grid_tiles(
    bbox: Tuple[float, float, float, float],
    tile_size: float,
) -> Iterable[Tuple[int, int, Tuple[float, float, float, float]]]:
    minx, miny, maxx, maxy = bbox
    cols = int(round((maxx - minx) / tile_size))
    rows = int(round((maxy - miny) / tile_size))
    for row in range(rows):
        y0 = miny + row * tile_size
        y1 = y0 + tile_size
        for col in range(cols):
            x0 = minx + col * tile_size
            x1 = x0 + tile_size
            yield row, col, (x0, y0, x1, y1)


def _print_layer_hint(layers: List[str]) -> None:
    if not layers:
        print("[DOP20] capabilities returned no layers")
        return
    filtered = [name for name in layers if ("DOP" in name.upper() or "RGB" in name.upper())]
    sample = filtered[:10] if filtered else layers[:10]
    print("[DOP20] available_layers_sample=" + ", ".join(sample))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tile-size-m", type=float, default=500.0)
    ap.add_argument("--resolution-m", type=float, default=0.2)
    ap.add_argument("--crs", default="EPSG:25832")
    ap.add_argument("--wms", required=True)
    ap.add_argument("--layer", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--retry", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    aoi_path = Path(args.aoi)
    out_root = Path(str(args.out_root).strip())
    tiles_dir = out_root / "tiles_utm32"
    errors_dir = out_root / "errors"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)

    bbox = _bbox_from_aoi(aoi_path, args.crs)
    bbox = _align_bbox(bbox, args.tile_size_m)
    width = int(math.ceil(args.tile_size_m / args.resolution_m))
    height = int(math.ceil(args.tile_size_m / args.resolution_m))
    if width <= 0 or height <= 0:
        raise RuntimeError("invalid_tile_size")

    try:
        xml_text = _get_capabilities(args.wms, args.timeout)
    except Exception as exc:
        print(f"[DOP20] capabilities_failed: {exc}")
        return 2
    layers = _extract_layers(xml_text)
    if not layers:
        print("[DOP20] no layers found, check WMS capabilities")
        return 2

    layer = args.layer.strip()
    if not layer:
        layer = _select_layer(layers) or ""
        if not layer:
            _print_layer_hint(layers)
            return 2
        print(f"[DOP20] selected_layer={layer}")
    elif layer not in layers:
        print(f"[DOP20] layer_not_found={layer}")
        _print_layer_hint(layers)
        return 2

    session_local = threading.local()

    def _get_session() -> requests.Session:
        sess = getattr(session_local, "session", None)
        if sess is None:
            sess = requests.Session()
            session_local.session = sess
        return sess

    report = {
        "tiles_total": 0,
        "ok": 0,
        "failed": 0,
        "skipped": 0,
        "failures": [],
    }

    def _download_one(
        row: int,
        col: int,
        tile_bbox: Tuple[float, float, float, float],
    ) -> Tuple[str, str, Optional[str]]:
        tile_name = f"tile_r{row:04d}_c{col:04d}"
        jpg_path = tiles_dir / f"{tile_name}.jpg"
        jgw_path = tiles_dir / f"{tile_name}.jgw"
        if jpg_path.exists() and jgw_path.exists():
            return "skipped", tile_name, None
        if jpg_path.exists() and not jgw_path.exists():
            _write_world_file(jgw_path, tile_bbox, width, height)
            return "skipped", tile_name, None

        bbox_param = _wms_bbox(tile_bbox, args.crs)
        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "CRS": args.crs,
            "BBOX": bbox_param,
            "WIDTH": str(width),
            "HEIGHT": str(height),
            "FORMAT": "image/jpeg",
            "LAYERS": layer,
            "STYLES": "",
        }

        for attempt in range(args.retry):
            try:
                resp = _get_session().get(args.wms, params=params, timeout=args.timeout)
                if resp.status_code != 200:
                    raise RuntimeError(f"status_{resp.status_code}")
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if "image" not in ctype:
                    error_path = errors_dir / f"{tile_name}.txt"
                    error_path.write_text(resp.text, encoding="utf-8", errors="ignore")
                    return "failed", tile_name, f"non_image:{ctype}"
                jpg_path.write_bytes(resp.content)
                _write_world_file(jgw_path, tile_bbox, width, height)
                return "ok", tile_name, None
            except Exception as exc:
                if attempt >= args.retry - 1:
                    return "failed", tile_name, str(exc)
        return "failed", tile_name, "unknown"

    futures = []
    tiles = list(_grid_tiles(bbox, args.tile_size_m))
    report["tiles_total"] = len(tiles)
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        for row, col, tile_bbox in tiles:
            futures.append(exe.submit(_download_one, row, col, tile_bbox))
        for fut in as_completed(futures):
            status, tile_name, error = fut.result()
            if status == "ok":
                report["ok"] += 1
            elif status == "skipped":
                report["skipped"] += 1
            else:
                report["failed"] += 1
                report["failures"].append({"tile": tile_name, "error": error or "unknown"})

    report_path = out_root / "download_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DOP20] tiles_total={report['tiles_total']} ok={report['ok']} failed={report['failed']} skipped={report['skipped']}")
    print(f"[DOP20] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

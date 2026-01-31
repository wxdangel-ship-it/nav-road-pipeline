from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def _resolve_outputs_dir(path: Path) -> Path:
    if path.name == "outputs":
        return path
    candidate = path / "outputs"
    if candidate.exists():
        return candidate
    return path


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_rasters(root: Path, max_count: int = 200) -> List[Path]:
    rasters = []
    if not root.exists():
        return rasters
    for ext in (".tif", ".tiff", ".jp2"):
        for p in root.rglob(f"*{ext}"):
            rasters.append(p)
            if len(rasters) >= max_count:
                return rasters
    return rasters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", required=True, help="geom outputs dir or geom run dir")
    ap.add_argument("--dop20-root", default="", help="DOP20 root dir")
    args = ap.parse_args()

    outputs_dir = _resolve_outputs_dir(Path(args.outputs_dir))
    qgis_dir = outputs_dir / "qgis_package"
    qgis_dir.mkdir(parents=True, exist_ok=True)

    crs = _read_json(outputs_dir / "crs.json")
    internal_epsg = int(crs.get("internal_epsg", 32632))
    wgs84_epsg = int(crs.get("wgs84", 4326))

    dop20_root = Path(args.dop20_root) if args.dop20_root else None
    if dop20_root is None or not dop20_root.exists():
        env_root = os.environ.get("DOP20_ROOT", "")
        if env_root:
            dop20_root = Path(env_root)
        else:
            dop20_root = Path(r"E:\KITTI360\KITTI-360\_lglbw_dop20")

    tiles_dir = dop20_root / "tiles_utm32"
    dop20_rasters = _find_rasters(tiles_dir)
    dop20_present = bool(dop20_rasters)

    vector_layers = []
    def _add_vector(path: Path, name: str, epsg: int, role: str) -> None:
        if not path.exists():
            return
        vector_layers.append(
            {
                "name": name,
                "path": str(path),
                "epsg": epsg,
                "role": role,
            }
        )

    _add_vector(outputs_dir / "road_polygon.geojson", "road_polygon", internal_epsg, "geom")
    _add_vector(outputs_dir / "centerlines.geojson", "centerlines", internal_epsg, "geom")
    _add_vector(outputs_dir / "intersections.geojson", "intersections", internal_epsg, "geom")
    _add_vector(outputs_dir / "road_polygon_wgs84.geojson", "road_polygon_wgs84", wgs84_epsg, "geom_wgs84")
    _add_vector(outputs_dir / "centerlines_wgs84.geojson", "centerlines_wgs84", wgs84_epsg, "geom_wgs84")
    _add_vector(outputs_dir / "intersections_wgs84.geojson", "intersections_wgs84", wgs84_epsg, "geom_wgs84")
    _add_vector(outputs_dir / "osm_ref_roads.geojson", "osm_ref_roads", wgs84_epsg, "osm_ref")

    layers_meta = {
        "outputs_dir": str(outputs_dir),
        "internal_epsg": internal_epsg,
        "wgs84_epsg": wgs84_epsg,
        "dop20_present": dop20_present,
        "dop20_root": str(dop20_root),
        "tiles_dir": str(tiles_dir),
        "raster_files": [str(p) for p in dop20_rasters],
        "vector_layers": vector_layers,
    }
    _write_json(qgis_dir / "layers.json", layers_meta)

    md_lines = [
        "# QGIS Package",
        "",
        f"- outputs_dir: {outputs_dir}",
        f"- internal_epsg: {internal_epsg}",
        f"- wgs84_epsg: {wgs84_epsg}",
        f"- dop20_present: {str(dop20_present).lower()}",
        f"- dop20_root: {dop20_root}",
        "",
        "建议图层顺序：",
        "1) DOP20 影像（EPSG: internal）",
        "2) road_polygon / intersections（EPSG: internal）",
        "3) centerlines（EPSG: internal）",
        "4) osm_ref_roads（EPSG:4326，如存在）",
        "",
        "图层清单：",
    ]
    for layer in vector_layers:
        md_lines.append(f"- {layer['name']}: {layer['path']} (EPSG:{layer['epsg']}, {layer['role']})")
    for raster in dop20_rasters:
        md_lines.append(f"- dop20: {raster} (EPSG:{internal_epsg}, dop20)")
    (qgis_dir / "qgis_layers.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[QGIS] wrote {qgis_dir / 'layers.json'}")
    print(f"[QGIS] wrote {qgis_dir / 'qgis_layers.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

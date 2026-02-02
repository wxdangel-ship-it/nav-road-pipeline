from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import laspy

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_epsg_laz(path: Path) -> int | None:
    try:
        with laspy.open(path) as reader:
            crs = reader.header.parse_crs()
            return int(crs.to_epsg() or 0) if crs else None
    except Exception:
        return None


def main() -> int:
    pointer = REPO_ROOT / "baselines" / "ACTIVE_SURFACE_EVIDENCE_BASELINE.txt"
    baseline_dir = Path(pointer.read_text(encoding="utf-8").strip())
    report_dir = baseline_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = baseline_dir / "artifact_location.json"
    if not artifact_path.exists():
        raise SystemExit("artifact_location_missing")
    artifacts_base = Path(_load_json(artifact_path).get("artifacts_base_abs", ""))
    if not artifacts_base.exists():
        raise SystemExit("artifacts_base_missing")

    manifest_path = baseline_dir / "outputs" / "large_files_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("large_files_manifest_missing")
    manifest = _load_json(manifest_path)

    laz_samples = []
    dem_samples = []
    tile_samples = []
    for item in manifest:
        rel = item.get("rel_path", "")
        if rel.endswith(".laz") and "road_surface_points" in rel and len(laz_samples) < 8:
            laz_samples.append(artifacts_base / rel)
        if rel.endswith("surface_dem_utm32.tif") and len(dem_samples) < 8:
            dem_samples.append(artifacts_base / rel)
        if rel.endswith(".tif") and "bev_markings_utm32_tiles" in rel and len(tile_samples) < 8:
            tile_samples.append(artifacts_base / rel)

    ok = True
    results: List[Dict[str, object]] = []

    def _check_raster(path: Path) -> Dict[str, object]:
        try:
            import rasterio

            with rasterio.open(path) as ds:
                epsg = ds.crs.to_epsg() if ds.crs else None
                return {"path": str(path), "exists": True, "epsg": epsg}
        except Exception as exc:
            return {"path": str(path), "exists": path.exists(), "error": str(exc)}

    for p in laz_samples:
        entry = {"path": str(p), "exists": p.exists()}
        if not p.exists():
            ok = False
            results.append(entry)
            continue
        epsg = _read_epsg_laz(p)
        entry["epsg"] = epsg
        if epsg != 32632:
            ok = False
        results.append(entry)

    for p in dem_samples:
        entry = _check_raster(p)
        if not entry.get("exists") or entry.get("epsg") != 32632:
            ok = False
        results.append(entry)

    for p in tile_samples:
        entry = _check_raster(p)
        if not entry.get("exists") or entry.get("epsg") != 32632:
            ok = False
        results.append(entry)

    out = {"baseline_dir": str(baseline_dir), "ok": ok, "checked": results}
    (report_dir / "smoke_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

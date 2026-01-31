from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from uuid import uuid4

import geopandas as gpd
import rasterio
from shapely.geometry import LineString, MultiPoint

from pipeline.datasets.kitti360_io import _find_oxts_dir, load_kitti360_pose
from scripts.pipeline_common import (
    LOG,
    bbox_polygon,
    ensure_dir,
    ensure_overwrite,
    ensure_required_columns,
    load_yaml,
    now_ts,
    relpath,
    setup_logging,
    write_csv,
    write_json,
    write_text,
    write_gpkg_layer,
)

PRIMITIVE_FIELDS = [
    "evid_id",
    "source",
    "drive_id",
    "evid_type",
    "crs_epsg",
    "path_rel",
    "frame_start",
    "frame_end",
    "conf",
    "uncert",
    "meta_json",
]

WORLD_FIELDS = [
    "cand_id",
    "source",
    "drive_id",
    "class",
    "crs_epsg",
    "conf",
    "uncert",
    "evid_ref",
    "conflict",
    "attr_json",
]


def _scan_tiles(tiles_dir: Path) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for ext in ("*.tif", "*.tiff", "*.jp2", "*.jpg", "*.jpeg"):
        for path in tiles_dir.rglob(ext):
            try:
                with rasterio.open(path) as ds:
                    bounds = ds.bounds
                items.append(
                    {
                        "path": str(path),
                        "minx": float(bounds.left),
                        "miny": float(bounds.bottom),
                        "maxx": float(bounds.right),
                        "maxy": float(bounds.top),
                    }
                )
            except Exception:
                continue
    return items


def _load_tiles_index(dop20_root: Path) -> List[Dict[str, object]]:
    cache_path = dop20_root / "dop20_tiles_index.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    tiles_dir = dop20_root / "tiles_utm32"
    if not tiles_dir.exists():
        return []
    items = _scan_tiles(tiles_dir)
    payload = json.dumps(items, ensure_ascii=False, indent=2)
    try:
        cache_path.write_text(payload, encoding="utf-8")
    except PermissionError:
        fallback = Path("cache") / "dop20_tiles_index.json"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(payload, encoding="utf-8")
    return items


def _list_oxts_frames(oxts_dir: Path) -> List[Path]:
    return sorted(oxts_dir.glob("*.txt"))


def _sample_frames(paths: List[Path], frames_per_drive: int, stride: int) -> List[Path]:
    if not paths:
        return []
    if stride > 0:
        sampled = paths[::stride]
        return sampled[:frames_per_drive] if frames_per_drive > 0 else sampled
    if frames_per_drive <= 0 or frames_per_drive >= len(paths):
        return paths
    step = max(1, len(paths) // frames_per_drive)
    return paths[::step][:frames_per_drive]


def _extract_frame_id(path: Path) -> str:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return digits if digits else path.stem


def _build_roi(points: List[Tuple[float, float]], buffer_m: float) -> Tuple[object, Tuple[float, float, float, float]]:
    if not points:
        poly = bbox_polygon((0.0, 0.0, 0.0, 0.0))
        return poly, poly.bounds
    if len(points) == 1:
        poly = MultiPoint(points).buffer(buffer_m)
        return poly, poly.bounds
    line = LineString(points)
    poly = line.buffer(buffer_m)
    return poly, poly.bounds


def _load_drive_list(config: dict) -> List[str]:
    drives = [str(d) for d in (config.get("drives") or []) if str(d).strip()]
    drives_file = str(config.get("drives_file") or "")
    if drives_file:
        path = Path(drives_file)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    drives.append(line.strip())
    if not drives:
        return []
    seen = []
    for d in drives:
        if d not in seen:
            seen.append(d)
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sat_pipeline.yaml")
    ap.add_argument("--max-drives", type=int, default=0)
    ap.add_argument("--run-id", default="")
    args = ap.parse_args()

    config = load_yaml(Path(args.config))
    run_id = args.run_id or now_ts()
    run_dir = Path("runs") / f"sat_{run_id}"

    overwrite = bool(config.get("overwrite", True))
    if overwrite:
        ensure_overwrite(run_dir)
    ensure_dir(run_dir)

    setup_logging(run_dir / "run.log")
    LOG.info("run_id=%s", run_id)

    warnings: List[str] = []
    errors: List[str] = []

    data_root = Path(str(config.get("data_root") or os.environ.get("POC_DATA_ROOT", "")))
    if not data_root.exists():
        LOG.error("POC_DATA_ROOT not set or invalid.")
        return 2

    dop20_root = Path(
        str(config.get("dop20_root") or os.environ.get("DOP20_ROOT", r"E:\KITTI360\KITTI-360\_lglbw_dop20"))
    )
    if not dop20_root.exists():
        LOG.error("DOP20 root not found: %s", dop20_root)
        return 3

    tiles_index = _load_tiles_index(dop20_root)
    if not tiles_index:
        LOG.error("No DOP20 tiles found under %s", dop20_root)
        return 4

    tiles_rows = []
    for idx, item in enumerate(tiles_index):
        bounds = (item["minx"], item["miny"], item["maxx"], item["maxy"])
        tiles_rows.append(
            {
                "tile_id": f"tile_{idx:05d}",
                "path": str(item["path"]),
                "minx": item["minx"],
                "miny": item["miny"],
                "maxx": item["maxx"],
                "maxy": item["maxy"],
                "geometry": bbox_polygon(bounds),
            }
        )
    tiles_gdf = gpd.GeoDataFrame(tiles_rows, geometry="geometry", crs="EPSG:32632")
    tiles_path = run_dir / "tiles" / "sat_tiles_index.gpkg"
    write_gpkg_layer(tiles_path, "sat_tiles", tiles_gdf, 32632, warnings, overwrite=True)

    frames_per_drive = int(config.get("frames_per_drive", 50))
    stride = int(config.get("stride", 5))
    roi_buffer_m = float(config.get("roi_buffer_m", 30.0))

    drives = _load_drive_list(config)
    if not drives:
        velodyne_root = data_root / "data_3d_raw"
        if velodyne_root.exists():
            drives = sorted([p.name for p in velodyne_root.iterdir() if p.is_dir()])
    if args.max_drives > 0:
        drives = drives[: args.max_drives]
    if not drives:
        LOG.error("No drives found for SAT pipeline.")
        return 5

    summary_drives = {}
    for drive_id in drives:
        try:
            oxts_dir = _find_oxts_dir(data_root, drive_id)
        except Exception as exc:
            msg = f"drive={drive_id} oxts missing: {exc}"
            LOG.warning(msg)
            errors.append(msg)
            continue
        frames = _list_oxts_frames(oxts_dir)
        sampled = _sample_frames(frames, frames_per_drive, stride)
        if not sampled:
            msg = f"drive={drive_id} no oxts frames"
            LOG.warning(msg)
            errors.append(msg)
            continue

        frame_ids = [_extract_frame_id(p) for p in sampled]
        frame_start = int(frame_ids[0]) if frame_ids else -1
        frame_end = int(frame_ids[-1]) if frame_ids else -1

        traj_points: List[Tuple[float, float]] = []
        for frame_id in frame_ids:
            try:
                x, y, _ = load_kitti360_pose(data_root, drive_id, frame_id)
                traj_points.append((x, y))
            except Exception as exc:
                msg = f"drive={drive_id} frame={frame_id} pose error: {exc}"
                LOG.warning(msg)
                errors.append(msg)
                continue

        roi_geom, bounds = _build_roi(traj_points, roi_buffer_m)
        drive_dir = run_dir / "drives" / drive_id
        evidence_dir = ensure_dir(drive_dir / "evidence")
        candidates_dir = ensure_dir(drive_dir / "candidates")
        qa_dir = ensure_dir(drive_dir / "qa")

        evidence_rows = [
            {
                "evid_id": str(uuid4()),
                "source": "sat",
                "drive_id": drive_id,
                "evid_type": "roi_polygon",
                "crs_epsg": 32632,
                "path_rel": relpath(run_dir, evidence_dir / "sat_evidence_utm32.gpkg"),
                "frame_start": frame_start,
                "frame_end": frame_end,
                "conf": 0.5 if traj_points else 0.0,
                "uncert": None,
                "meta_json": json.dumps(
                    {
                        "roi_buffer_m": roi_buffer_m,
                        "traj_points": len(traj_points),
                    }
                ),
                "geometry": roi_geom,
            }
        ]
        evidence_gdf = gpd.GeoDataFrame(evidence_rows, geometry="geometry", crs="EPSG:32632")
        ensure_required_columns(evidence_gdf, PRIMITIVE_FIELDS)
        evidence_gpkg = evidence_dir / "sat_evidence_utm32.gpkg"
        write_gpkg_layer(evidence_gpkg, "primitive_evidence", evidence_gdf, 32632, warnings, overwrite=True)

        cand_rows = [
            {
                "cand_id": str(uuid4()),
                "source": "sat",
                "drive_id": drive_id,
                "class": "intersection_roi",
                "crs_epsg": 32632,
                "conf": 0.5 if traj_points else 0.0,
                "uncert": None,
                "evid_ref": json.dumps([evidence_rows[0]["evid_id"]]),
                "conflict": "",
                "attr_json": json.dumps({"roi_buffer_m": roi_buffer_m, "traj_points": len(traj_points)}),
                "geometry": roi_geom,
            }
        ]
        candidates_gdf = gpd.GeoDataFrame(cand_rows, geometry="geometry", crs="EPSG:32632")
        ensure_required_columns(candidates_gdf, WORLD_FIELDS)
        candidates_gpkg = candidates_dir / "sat_candidates_utm32.gpkg"
        write_gpkg_layer(candidates_gpkg, "world_candidates", candidates_gdf, 32632, warnings, overwrite=True)

        qa_rows = [
            {
                "drive_id": drive_id,
                "item": "sat_evidence_utm32.gpkg",
                "path": relpath(run_dir, evidence_gpkg),
                "minx": bounds[0],
                "miny": bounds[1],
                "maxx": bounds[2],
                "maxy": bounds[3],
                "traj_points": len(traj_points),
            },
            {
                "drive_id": drive_id,
                "item": "sat_candidates_utm32.gpkg",
                "path": relpath(run_dir, candidates_gpkg),
                "minx": bounds[0],
                "miny": bounds[1],
                "maxx": bounds[2],
                "maxy": bounds[3],
                "traj_points": len(traj_points),
            },
        ]
        write_csv(qa_dir / "qa_index.csv", qa_rows, list(qa_rows[0].keys()))
        report_lines = [
            "# SAT QA Report",
            "",
            f"- drive_id: {drive_id}",
            f"- frames: {len(frame_ids)}",
            f"- roi_buffer_m: {roi_buffer_m}",
            f"- traj_points: {len(traj_points)}",
            "",
            "## Parameters",
            "```json",
            json.dumps(
                {
                    "frames_per_drive": frames_per_drive,
                    "stride": stride,
                    "roi_buffer_m": roi_buffer_m,
                    "dop20_root": str(dop20_root),
                },
                indent=2,
            ),
            "```",
            "",
            "## CRS Checks",
        ]
        report_lines.extend([f"- {w}" for w in warnings] if warnings else ["- ok"])
        report_lines.extend(
            [
                "",
                "## Outputs",
                f"- {relpath(run_dir, evidence_gpkg)}",
                f"- {relpath(run_dir, candidates_gpkg)}",
                f"- {relpath(run_dir, qa_dir / 'qa_index.csv')}",
                "",
                "## Failures",
            ]
        )
        report_lines.extend([f"- {e}" for e in errors] if errors else ["- none"])
        report_lines.extend(
            [
                "",
                "## QA Index",
                "- qa_index.csv contains per-drive artifacts with bbox and trajectory support.",
            ]
        )
        write_text(qa_dir / "report.md", "\n".join(report_lines))

        summary_drives[drive_id] = {
            "frames": len(frame_ids),
            "traj_points": len(traj_points),
            "evidence": relpath(run_dir, evidence_gpkg),
            "candidates": relpath(run_dir, candidates_gpkg),
        }

    summary = {
        "run_id": run_id,
        "config": args.config,
        "data_root": str(data_root),
        "dop20_root": str(dop20_root),
        "drives": summary_drives,
        "warnings": warnings,
        "errors": errors,
    }
    write_json(run_dir / "run_summary.json", summary)
    summary_md = [
        "# SAT Run Summary",
        "",
        f"- run_id: {run_id}",
        f"- config: {args.config}",
        f"- data_root: {data_root}",
        f"- dop20_root: {dop20_root}",
        f"- drives: {len(summary_drives)}",
        "",
        "## Warnings",
    ]
    summary_md.extend([f"- {w}" for w in warnings] if warnings else ["- none"])
    summary_md.extend(["", "## Errors"])
    summary_md.extend([f"- {e}" for e in errors] if errors else ["- none"])
    summary_md.extend(["", "## Outputs"])
    summary_md.extend(
        [f"- drives/{drive_id}/evidence/sat_evidence_utm32.gpkg" for drive_id in summary_drives]
    )
    write_text(run_dir / "run_summary.md", "\n".join(summary_md))
    LOG.info("completed sat run: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

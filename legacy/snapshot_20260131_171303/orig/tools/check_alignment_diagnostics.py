import argparse
import csv
import datetime as dt
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rio_features
from rasterio.crs import CRS
from shapely.geometry import Point, box


def _parse_frame_id(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    try:
        return int(float(text))
    except ValueError:
        return None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows, header):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _load_layers(path: Path):
    layers = None
    try:
        import fiona

        layers = list(fiona.listlayers(path))
    except ImportError:
        try:
            import pyogrio

            layer_rows = pyogrio.list_layers(path)
            layers = [row[0] for row in layer_rows]
        except ImportError:
            layers = None

    if not layers:
        gdf = gpd.read_file(path)
        gdf["_source_layer"] = "default"
        return [gdf]

    gdfs = []
    for layer in layers:
        gdf = gpd.read_file(path, layer=layer)
        gdf["_source_layer"] = layer
        gdfs.append(gdf)
    return gdfs


def _ensure_wgs84_range(gdf: gpd.GeoDataFrame) -> bool:
    if "geometry" in gdf.columns and not gdf.geometry.is_empty.all():
        xs = gdf.geometry.x.to_numpy()
        ys = gdf.geometry.y.to_numpy()
    elif {"lon", "lat"}.issubset(gdf.columns):
        xs = gdf["lon"].to_numpy()
        ys = gdf["lat"].to_numpy()
    else:
        return False
    if xs.size == 0 or ys.size == 0:
        return False
    return (
        np.isfinite(xs).all()
        and np.isfinite(ys).all()
        and (xs >= -180).all()
        and (xs <= 180).all()
        and (ys >= -90).all()
        and (ys <= 90).all()
    )


def _frame_range_from_gdf(gdf: gpd.GeoDataFrame):
    if "frame_id" not in gdf.columns:
        return None, None
    parsed = gdf["frame_id"].apply(_parse_frame_id)
    parsed = parsed.dropna()
    if parsed.empty:
        return None, None
    return int(parsed.min()), int(parsed.max())


def _filter_frame_range(gdf: gpd.GeoDataFrame, frame_start, frame_end):
    if "frame_id" not in gdf.columns:
        return gdf
    parsed = gdf["frame_id"].apply(_parse_frame_id)
    mask = parsed.notna()
    if frame_start is not None:
        mask &= parsed >= frame_start
    if frame_end is not None:
        mask &= parsed <= frame_end
    return gdf.loc[mask]


def _filter_drive(gdf: gpd.GeoDataFrame, drive_id: str):
    if "drive_id" not in gdf.columns:
        return gdf
    return gdf.loc[gdf["drive_id"].astype(str) == drive_id]


def _collect_traj_points(
    qa_gdf: gpd.GeoDataFrame, drive_id: str, frame_start, frame_end
):
    qa = _filter_drive(qa_gdf, drive_id)
    qa = _filter_frame_range(qa, frame_start, frame_end)
    if qa.empty:
        return qa
    if qa.crs is None:
        qa = qa.set_crs("EPSG:4326")
    if "geometry" not in qa.columns or qa.geometry.is_empty.all():
        if {"lon", "lat"}.issubset(qa.columns):
            qa = qa.copy()
            qa["geometry"] = [Point(xy) for xy in zip(qa["lon"], qa["lat"])]
            qa = gpd.GeoDataFrame(qa, geometry="geometry", crs="EPSG:4326")
        else:
            raise RuntimeError("qa_index_missing_geometry")
    qa_utm32 = qa.to_crs("EPSG:32632")
    qa_utm32 = qa_utm32.copy()
    qa_utm32["x"] = qa_utm32.geometry.x
    qa_utm32["y"] = qa_utm32.geometry.y
    qa_utm32["source"] = "qa_index"
    return qa_utm32


def _write_gpkg(gdf: gpd.GeoDataFrame, path: Path, layer_name: str):
    if path.exists():
        path.unlink()
    gdf.to_file(path, layer=layer_name, driver="GPKG")


def _dist_stats(values):
    if not values:
        return {"p50": None, "p90": None, "max": None}
    arr = np.array(values)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _summarize_offset(best_offsets):
    if not best_offsets:
        return None, None
    series = pd.Series(best_offsets)
    mode = series.mode()
    return int(mode.iloc[0]) if not mode.empty else None, int(series.size)


def main():
    parser = argparse.ArgumentParser(description="Check alignment diagnostics.")
    parser.add_argument("--qa-index", required=True)
    parser.add_argument("--lidar-raster", required=True)
    parser.add_argument("--frame-evidence", required=True)
    parser.add_argument("--lidar-evidence", required=True)
    parser.add_argument("--drive", required=True)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--drives-file")
    args = parser.parse_args()

    if args.drives_file:
        drives = [
            line.strip()
            for line in Path(args.drives_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        drives = [args.drive]

    outputs_root = Path(args.out_dir)
    _ensure_dir(outputs_root)
    overview_rows = []

    for drive_id in drives:
        outputs_dir = outputs_root / drive_id if args.drives_file else outputs_root
        _ensure_dir(outputs_dir)

        qa_index_path = Path(args.qa_index.format(drive=drive_id))
        lidar_raster_path = Path(args.lidar_raster.format(drive=drive_id))
        frame_evidence_path = Path(args.frame_evidence.format(drive=drive_id))
        lidar_evidence_path = Path(args.lidar_evidence.format(drive=drive_id))
        if not qa_index_path.exists():
            raise FileNotFoundError(qa_index_path)
        if not lidar_raster_path.exists():
            raise FileNotFoundError(lidar_raster_path)
        if not frame_evidence_path.exists():
            raise FileNotFoundError(frame_evidence_path)
        if not lidar_evidence_path.exists():
            raise FileNotFoundError(lidar_evidence_path)

        qa_gdf = gpd.read_file(qa_index_path)
        if args.frame_start is None or args.frame_end is None:
            auto_start, auto_end = _frame_range_from_gdf(qa_gdf)
            frame_start = args.frame_start if args.frame_start is not None else auto_start
            frame_end = args.frame_end if args.frame_end is not None else auto_end
        else:
            frame_start = args.frame_start
            frame_end = args.frame_end

        frame_layers = _load_layers(frame_evidence_path)
        if not frame_layers:
            raise RuntimeError("frame_evidence_empty")
        frame_gdf = pd.concat(frame_layers, ignore_index=True)
        frame_gdf = gpd.GeoDataFrame(frame_gdf, geometry="geometry", crs=frame_layers[0].crs)
        frame_gdf = _filter_drive(frame_gdf, drive_id)
        frame_gdf = _filter_frame_range(frame_gdf, frame_start, frame_end)

        total_records = int(len(frame_gdf))
        if total_records == 0:
            raise RuntimeError("frame_evidence_empty_after_filter")
        frame_id_series = frame_gdf["frame_id"].astype(str)
        frame_id_counts = frame_id_series.value_counts().sort_index()
        frame_id_distinct = int(frame_id_counts.size)
        frame_id_dist_path = outputs_dir / "frame_id_distribution.csv"
        _write_csv(
            frame_id_dist_path,
            [(fid, int(cnt)) for fid, cnt in frame_id_counts.items()],
            ["frame_id", "count"],
        )

        frame_id_bug_path = outputs_dir / "frame_id_bug_flag.txt"
        frame_id_bug = frame_id_distinct <= 3 and total_records > 3
        if frame_id_bug:
            frame_id_bug_path.write_text(
                f"ERROR: frame_id distinct={frame_id_distinct} records={total_records}\n",
                encoding="utf-8",
            )
            report_path = outputs_dir / "alignment_check_report.md"
            report_lines = [
                "# Alignment Check Report",
                f"- report_time: {dt.datetime.now():%Y-%m-%d %H:%M:%S}",
                f"- drive_id: {drive_id}",
                f"- frame_range: {frame_start}-{frame_end}",
                "",
                "## Frame ID Diversity",
                f"- total_records: {total_records}",
                f"- distinct_frame_id: {frame_id_distinct}",
                "",
                "## ERROR",
                "- reason: frame_id distinct count too small; aborting further checks",
                f"- frame_id_distribution: {frame_id_dist_path}",
                f"- frame_id_bug_flag: {frame_id_bug_path}",
            ]
            report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
            raise SystemExit(2)
        frame_id_bug_path.write_text("OK\n", encoding="utf-8")

        crs_summary_path = outputs_dir / "crs_check_summary.md"
        qa_range_ok = _ensure_wgs84_range(qa_gdf)
        with rasterio.open(lidar_raster_path) as ds:
            raster_crs = ds.crs
            raster_bounds = ds.bounds
            raster_nodata = ds.nodata
            raster_bbox = box(raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top)

        raster_ok = raster_crs == CRS.from_epsg(32632)
        qa_utm32 = _collect_traj_points(qa_gdf, drive_id, frame_start, frame_end)
        qa_points = qa_utm32.geometry
        inside_mask = qa_points.within(raster_bbox)
        inside_ratio = float(inside_mask.mean()) if len(inside_mask) else 0.0
        crs_summary_lines = [
            "# CRS Check Summary",
            f"- drive_id: {drive_id}",
            f"- frame_range: {frame_start}-{frame_end}",
            f"- qa_index_path: {qa_index_path}",
            f"- lidar_raster_path: {lidar_raster_path}",
            f"- frame_evidence_path: {frame_evidence_path}",
            f"- lidar_evidence_path: {lidar_evidence_path}",
            f"- qa_index_wgs84_range_ok: {qa_range_ok}",
            f"- lidar_raster_crs: {raster_crs}",
            f"- lidar_raster_crs_ok: {raster_ok}",
            f"- raster_bounds_utm32: {raster_bounds}",
            f"- qa_points_inside_raster_ratio: {inside_ratio:.3f}",
        ]
        crs_summary_path.write_text("\n".join(crs_summary_lines) + "\n", encoding="utf-8")
        if not qa_range_ok or not raster_ok or inside_ratio == 0.0:
            report_path = outputs_dir / "alignment_check_report.md"
            report_lines = [
                "# Alignment Check Report",
                f"- report_time: {dt.datetime.now():%Y-%m-%d %H:%M:%S}",
                f"- drive_id: {drive_id}",
                f"- frame_range: {frame_start}-{frame_end}",
                "",
                "## CRS Check",
                f"- qa_index_wgs84_range_ok: {qa_range_ok}",
                f"- lidar_raster_crs_ok: {raster_ok}",
                f"- qa_points_inside_raster_ratio: {inside_ratio:.3f}",
                "",
                "## ERROR",
                "- reason: CRS check failed; aborting further checks",
                f"- crs_check_summary: {crs_summary_path}",
            ]
            report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
            raise SystemExit(2)

        traj_points_path = outputs_dir / "traj_points_utm32.gpkg"
        _write_gpkg(
            qa_utm32[["drive_id", "frame_id", "x", "y", "source", "geometry"]],
            traj_points_path,
            "traj_points_utm32",
        )

        frame_subset_path = outputs_dir / "frame_evidence_subset_utm32.gpkg"
        _write_gpkg(frame_gdf, frame_subset_path, "frame_evidence_subset")

        traj_lookup = {}
        for _, row in qa_utm32.iterrows():
            frame_id_int = _parse_frame_id(row.get("frame_id"))
            if frame_id_int is None:
                continue
            traj_lookup.setdefault(frame_id_int, (row["x"], row["y"]))

        delta_rows = []
        best_offset_rows = []
        dist_values = []
        best_offsets = []
        for _, row in frame_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            centroid = geom.centroid
            frame_id_int = _parse_frame_id(row.get("frame_id"))
            if frame_id_int is None:
                continue
            traj = traj_lookup.get(frame_id_int)
            if traj is None:
                continue
            dx = centroid.x - traj[0]
            dy = centroid.y - traj[1]
            dist = math.hypot(dx, dy)
            dist_values.append(dist)
            delta_rows.append(
                [
                    drive_id,
                    row.get("frame_id"),
                    row.get("_source_layer", ""),
                    centroid.x,
                    centroid.y,
                    traj[0],
                    traj[1],
                    dx,
                    dy,
                    dist,
                ]
            )

            best_dist = None
            best_offset = None
            for offset in range(-2, 3):
                traj_off = traj_lookup.get(frame_id_int + offset)
                if traj_off is None:
                    continue
                d_off = math.hypot(centroid.x - traj_off[0], centroid.y - traj_off[1])
                if best_dist is None or d_off < best_dist:
                    best_dist = d_off
                    best_offset = offset
            if best_offset is not None:
                best_offsets.append(best_offset)
                best_offset_rows.append(
                    [
                        drive_id,
                        row.get("frame_id"),
                        best_offset,
                        best_dist,
                    ]
                )

        delta_path = outputs_dir / "evidence_vs_traj_delta.csv"
        _write_csv(
            delta_path,
            delta_rows,
            [
                "drive_id",
                "frame_id",
                "source_layer",
                "evidence_x",
                "evidence_y",
                "traj_x",
                "traj_y",
                "dx",
                "dy",
                "dist",
            ],
        )

        best_offset_path = outputs_dir / "best_offset_summary.csv"
        _write_csv(
            best_offset_path,
            best_offset_rows,
            ["drive_id", "frame_id", "best_offset", "best_dist"],
        )

        traj_sample_rows = []
        with rasterio.open(lidar_raster_path) as ds:
            for _, row in qa_utm32.iterrows():
                x, y = row["x"], row["y"]
                val = next(ds.sample([(x, y)]))[0]
                is_nodata = False
                if raster_nodata is not None and val == raster_nodata:
                    is_nodata = True
                traj_sample_rows.append(
                    [drive_id, row.get("frame_id"), x, y, val, int(is_nodata)]
                )
        traj_sample_path = outputs_dir / "traj_raster_sample.csv"
        _write_csv(
            traj_sample_path,
            traj_sample_rows,
            ["drive_id", "frame_id", "x", "y", "raster_val", "is_nodata"],
        )

        dist_stats = _dist_stats(dist_values)
        dx_std = float(np.std([r[7] for r in delta_rows])) if delta_rows else None
        dy_std = float(np.std([r[8] for r in delta_rows])) if delta_rows else None
        drift_mode = "unknown"
        if dx_std is not None and dy_std is not None and dist_stats["p50"] is not None:
            if dx_std <= 2.0 and dy_std <= 2.0:
                drift_mode = "constant_shift"
            else:
                drift_mode = "frame_drift"
        offset_mode, offset_count = _summarize_offset(best_offsets)

        report_path = outputs_dir / "alignment_check_report.md"
        report_lines = [
            "# Alignment Check Report",
            f"- report_time: {dt.datetime.now():%Y-%m-%d %H:%M:%S}",
            f"- drive_id: {drive_id}",
            f"- frame_range: {frame_start}-{frame_end}",
            "",
            "## CRS Check",
            f"- qa_index_wgs84_range_ok: {qa_range_ok}",
            f"- lidar_raster_crs_ok: {raster_ok}",
            f"- qa_points_inside_raster_ratio: {inside_ratio:.3f}",
            "",
            "## Frame ID Diversity",
            f"- total_records: {total_records}",
            f"- distinct_frame_id: {frame_id_distinct}",
            "",
            "## Evidence vs Trajectory Delta",
            f"- dist_p50: {dist_stats['p50']}",
            f"- dist_p90: {dist_stats['p90']}",
            f"- dist_max: {dist_stats['max']}",
            f"- dx_std: {dx_std}",
            f"- dy_std: {dy_std}",
            f"- drift_mode: {drift_mode}",
            "",
            "## Offset Scan",
            f"- best_offset_mode: {offset_mode}",
            f"- best_offset_samples: {offset_count}",
            "",
            "## Outputs",
            f"- frame_id_distribution: {frame_id_dist_path}",
            f"- crs_check_summary: {crs_summary_path}",
            f"- traj_points_utm32: {traj_points_path}",
            f"- frame_evidence_subset: {frame_subset_path}",
            f"- evidence_vs_traj_delta: {delta_path}",
            f"- best_offset_summary: {best_offset_path}",
            f"- traj_raster_sample: {traj_sample_path}",
        ]
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

        overview_rows.append(
            [
                drive_id,
                dist_stats["p90"],
                offset_mode,
                frame_id_bug,
                drift_mode,
            ]
        )

    if args.drives_file:
        overview_path = outputs_root / "overview_all_drives.csv"
        _write_csv(
            overview_path,
            overview_rows,
            ["drive_id", "dist_p90", "best_offset_mode", "frame_id_bug", "drift_mode"],
        )


if __name__ == "__main__":
    main()

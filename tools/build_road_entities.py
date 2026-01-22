from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon, box
from shapely.ops import linemerge, unary_union

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.datasets.kitti360_io import load_kitti360_lidar_points_world, load_kitti360_pose


LOG = logging.getLogger("build_road_entities")


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("build_road_entities")


def _load_index(path: Path) -> List[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _group_by_drive(rows: Iterable[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for row in rows:
        drive_id = str(row.get("drive_id") or "")
        frame_id = str(row.get("frame_id") or "")
        if not drive_id or not frame_id:
            continue
        out.setdefault(drive_id, []).append(row)
    for drive_id in out:
        out[drive_id] = sorted(out[drive_id], key=lambda r: str(r.get("frame_id")))
    return out


def _ensure_wgs84_range(gdf: gpd.GeoDataFrame) -> bool:
    if gdf.empty:
        return True
    try:
        bounds = gdf.total_bounds
    except Exception:
        return False
    minx, miny, maxx, maxy = bounds
    return -180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0 and -90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), p))


def _read_map_evidence(path: Path) -> Dict[str, gpd.GeoDataFrame]:
    layers = []
    out = {}
    try:
        import pyogrio

        layers = [name for name, _ in pyogrio.list_layers(path)]
        for layer in layers:
            try:
                out[layer] = pyogrio.read_dataframe(path, layer=layer)
            except Exception:
                continue
        return out
    except Exception:
        layers = []
    try:
        layers = gpd.io.file.fiona.listlayers(str(path))
    except Exception:
        layers = []
    for layer in layers:
        try:
            out[layer] = gpd.read_file(path, layer=layer)
        except Exception:
            continue
    return out


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_gpkg_layers(path: Path, layers: Dict[str, gpd.GeoDataFrame], crs: str) -> None:
    if path.exists():
        path.unlink()
    for name, gdf in layers.items():
        if gdf is None:
            gdf = gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs=crs)
        if gdf.crs is None:
            gdf = gdf.set_crs(crs)
        gdf.to_file(path, layer=name, driver="GPKG")


def _rasterize_points(
    points: np.ndarray,
    values: np.ndarray,
    grid_m: float,
    min_points: int,
) -> Tuple[np.ndarray, Tuple[float, float, float, float], float]:
    if points.shape[0] == 0:
        return np.zeros((1, 1), dtype=float), (0.0, 0.0, 0.0, 0.0), grid_m
    minx = float(points[:, 0].min())
    miny = float(points[:, 1].min())
    maxx = float(points[:, 0].max())
    maxy = float(points[:, 1].max())
    width = int(np.ceil((maxx - minx) / grid_m)) + 1
    height = int(np.ceil((maxy - miny) / grid_m)) + 1
    raster = np.zeros((height, width), dtype=float)
    counts = np.zeros((height, width), dtype=int)
    xs = ((points[:, 0] - minx) / grid_m).astype(int)
    ys = ((points[:, 1] - miny) / grid_m).astype(int)
    for i in range(points.shape[0]):
        x = xs[i]
        y = ys[i]
        if 0 <= x < width and 0 <= y < height:
            raster[y, x] = max(raster[y, x], float(values[i]))
            counts[y, x] += 1
    raster[counts < min_points] = 0.0
    return raster, (minx, miny, maxx, maxy), grid_m


def _cells_to_polygons(raster: np.ndarray, bounds: Tuple[float, float, float, float], grid_m: float) -> Polygon:
    minx, miny, _, _ = bounds
    boxes = []
    ys, xs = np.where(raster > 0)
    for y, x in zip(ys, xs):
        x0 = minx + x * grid_m
        y0 = miny + y * grid_m
        boxes.append(box(x0, y0, x0 + grid_m, y0 + grid_m))
    if not boxes:
        return Polygon()
    return unary_union(boxes)


def _to_lines(gdf: gpd.GeoDataFrame, min_length_m: float) -> gpd.GeoDataFrame:
    lines = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, (Polygon, MultiPolygon)):
            line = geom.boundary
        elif isinstance(geom, (LineString, MultiLineString)):
            line = geom
        else:
            continue
        if line.length < min_length_m:
            continue
        rec = row.copy()
        rec.geometry = line
        lines.append(rec)
    if not lines:
        return gpd.GeoDataFrame(columns=gdf.columns, geometry=[], crs=gdf.crs)
    return gpd.GeoDataFrame(lines, geometry="geometry", crs=gdf.crs)


def _cluster_geoms(gdf: gpd.GeoDataFrame, buffer_m: float) -> List[Polygon]:
    if gdf.empty:
        return []
    geoms = [geom.buffer(buffer_m) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geoms:
        return []
    merged = unary_union(geoms)
    if isinstance(merged, Polygon):
        return [merged]
    if isinstance(merged, MultiPolygon):
        return list(merged.geoms)
    return []


def _line_direction_deg(geom: LineString) -> Optional[float]:
    coords = np.array(geom.coords, dtype=float)
    if coords.shape[0] < 2:
        return None
    mean = coords.mean(axis=0)
    cov = np.cov((coords - mean).T)
    vals, vecs = np.linalg.eig(cov)
    idx = int(np.argmax(vals))
    direction = vecs[:, idx]
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    if angle < 0:
        angle += 180.0
    return angle


def _angle_diff_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def _line_endpoints(geom: LineString) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    coords = list(geom.coords)
    return (coords[0][0], coords[0][1]), (coords[-1][0], coords[-1][1])


def _min_endpoint_gap(a: LineString, b: LineString) -> float:
    a0, a1 = _line_endpoints(a)
    b0, b1 = _line_endpoints(b)
    pairs = [(a0, b0), (a0, b1), (a1, b0), (a1, b1)]
    return min(float(np.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1])) for p in pairs)


def _aggregate_lines(
    gdf: gpd.GeoDataFrame,
    max_merge_dist_m: float,
    max_gap_m: float,
    max_angle_diff_deg: float,
    min_length_m: float,
    simplify_tol_m: float,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    gdf = gdf.copy()
    if simplify_tol_m > 0:
        gdf["geometry"] = gdf["geometry"].apply(lambda g: g.simplify(simplify_tol_m) if g is not None else g)

    lines = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            lines.append((geom, row))
        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                lines.append((part, row))
    if not lines:
        return gpd.GeoDataFrame(columns=gdf.columns, geometry=[], crs=gdf.crs)

    geoms = [item[0] for item in lines]
    directions = [(_line_direction_deg(g) if g.length > 0 else None) for g in geoms]
    centers = [g.interpolate(0.5, normalized=True) for g in geoms]

    try:
        from shapely.strtree import STRtree

        tree = STRtree(geoms)
        geom_id = {id(g): idx for idx, g in enumerate(geoms)}
    except Exception:
        tree = None
        geom_id = {}

    parent = list(range(len(geoms)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri = _find(i)
        rj = _find(j)
        if ri != rj:
            parent[rj] = ri

    for i, geom in enumerate(geoms):
        if directions[i] is None:
            continue
        candidates = []
        if tree is not None:
            query_geom = geom.buffer(max_merge_dist_m + max_gap_m)
            for cand in tree.query(query_geom):
                idx = geom_id.get(id(cand))
                if idx is not None:
                    candidates.append(idx)
        else:
            candidates = list(range(len(geoms)))
        for j in candidates:
            if j <= i:
                continue
            if directions[j] is None:
                continue
            if _angle_diff_deg(directions[i], directions[j]) > max_angle_diff_deg:
                continue
            mid_dist = centers[i].distance(centers[j])
            if mid_dist <= max_merge_dist_m or _min_endpoint_gap(geoms[i], geoms[j]) <= max_gap_m:
                _union(i, j)

    clusters: Dict[int, List[int]] = {}
    for idx in range(len(geoms)):
        root = _find(idx)
        clusters.setdefault(root, []).append(idx)

    merged_rows = []
    for indices in clusters.values():
        cluster_geoms = [geoms[i] for i in indices]
        merged = linemerge(cluster_geoms)
        if isinstance(merged, MultiLineString):
            merged = linemerge(list(merged.geoms))
        if merged is None or merged.is_empty:
            continue
        if merged.length < min_length_m:
            continue
        base_row = lines[indices[0]][1].copy()
        base_row.geometry = merged
        merged_rows.append(base_row)

    if not merged_rows:
        return gpd.GeoDataFrame(columns=gdf.columns, geometry=[], crs=gdf.crs)
    return gpd.GeoDataFrame(merged_rows, geometry="geometry", crs=gdf.crs)


def _entity_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:04d}"


def _build_entities_from_clusters(
    gdf: gpd.GeoDataFrame,
    clusters: List[Polygon],
    entity_type: str,
    drive_id: str,
    line_merge: bool,
    qa_flag: str,
    evidence_sources: dict,
    created_from_run: str,
) -> List[dict]:
    entities = []
    if gdf.empty or not clusters:
        return entities

    if gdf.sindex is None:
        gdf.sindex
    for idx, cluster in enumerate(clusters):
        hits = gdf[gdf.geometry.intersects(cluster)]
        if hits.empty:
            continue
        frames = set()
        providers = {}
        for _, row in hits.iterrows():
            frame_id = row.get("frame_id")
            if frame_id:
                frames.add(str(frame_id))
            model_id = row.get("model_id") or row.get("provider_id")
            if model_id:
                providers[str(model_id)] = providers.get(str(model_id), 0) + 1

        if line_merge:
            merged = linemerge(list(hits.geometry))
            geom = merged if isinstance(merged, (LineString, MultiLineString)) else unary_union(hits.geometry.values)
        else:
            geom = unary_union(hits.geometry.values)
        if geom is None or geom.is_empty:
            continue

        entities.append(
            {
                "geometry": geom,
                "properties": {
                    "entity_id": _entity_id(f"{drive_id}_{entity_type}", idx),
                    "drive_id": drive_id,
                    "entity_type": entity_type,
                    "confidence": 0.7 if evidence_sources.get("image") else 0.5,
                    "evidence_sources": json.dumps(evidence_sources, ensure_ascii=True),
                    "frames_hit": len(frames),
                    "provider_hits": json.dumps(providers, ensure_ascii=True),
                    "created_from_run": created_from_run,
                    "qa_flag": qa_flag,
                },
            }
        )
    return entities


def _gdf_from_entities(feats: List[dict], crs: str) -> gpd.GeoDataFrame:
    if not feats:
        return gpd.GeoDataFrame(columns=["entity_id"], geometry=[], crs=crs)
    rows = [feat["properties"] for feat in feats]
    geoms = [feat["geometry"] for feat in feats]
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)


def _line_stats_by_drive(gdf: gpd.GeoDataFrame) -> Dict[str, dict]:
    if gdf.empty:
        return {}
    if "drive_id" not in gdf.columns:
        gdf = gdf.copy()

        def _infer_drive_id(row: pd.Series) -> str:
            entity_id = str(row.get("entity_id") or "")
            entity_type = str(row.get("entity_type") or "")
            if entity_type and f"_{entity_type}_" in entity_id:
                return entity_id.split(f"_{entity_type}_")[0]
            parts = entity_id.split("_")
            return "_".join(parts[:-1]) if len(parts) > 1 else ""

        gdf["drive_id"] = gdf.apply(_infer_drive_id, axis=1)
    stats = {}
    for drive_id, group in gdf.groupby("drive_id"):
        lengths = [float(geom.length) for geom in group.geometry if geom is not None and not geom.is_empty]
        stats[str(drive_id)] = {
            "segments": int(len(group)),
            "length_p50": _percentile(lengths, 50),
            "length_p90": _percentile(lengths, 90),
        }
    return stats


def _format_stats(stats: Dict[str, dict]) -> List[str]:
    lines = []
    for drive_id, row in stats.items():
        lines.append(
            f"- {drive_id}: segments={row.get('segments')} length_p50={row.get('length_p50')} length_p90={row.get('length_p90')}"
        )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--image-run", required=True)
    ap.add_argument("--image-provider", default="grounded_sam2_v1")
    ap.add_argument("--image-evidence-gpkg", default="")
    ap.add_argument("--config", default="configs/road_entities.yaml")
    ap.add_argument("--road-root", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--lidar-grid-m", type=float, default=0.5)
    ap.add_argument("--lidar-min-points", type=int, default=5)
    ap.add_argument("--lidar-max-points-per-frame", type=int, default=20000)
    ap.add_argument("--lidar-z-min", type=float, default=-2.0)
    ap.add_argument("--lidar-z-max", type=float, default=0.5)
    ap.add_argument("--qa-radius-m", type=float, default=20.0)
    ap.add_argument("--line-min-length-m", type=float, default=2.0)
    ap.add_argument("--cluster-buffer-m", type=float, default=2.0)
    args = ap.parse_args()

    log = _setup_logger()
    data_root = Path(os.environ.get("POC_DATA_ROOT", ""))
    if not data_root.exists():
        log.error("POC_DATA_ROOT not set or invalid.")
        return 2

    index_path = Path(args.index)
    if not index_path.exists():
        log.error("index not found: %s", index_path)
        return 3

    run_dir = Path(args.out_dir) if args.out_dir else Path("runs") / f"road_entities_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_index(index_path)
    by_drive = _group_by_drive(rows)
    if not by_drive:
        log.error("no frames in index")
        return 4

    image_run = Path(args.image_run)
    image_map_root = image_run / f"feature_store_map_{args.image_provider}"
    image_evidence_gpkg = Path(args.image_evidence_gpkg) if args.image_evidence_gpkg else None
    image_evidence_all = {}
    if image_evidence_gpkg and image_evidence_gpkg.exists():
        image_evidence_all = _read_map_evidence(image_evidence_gpkg)

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path) if cfg_path.exists() else {}
    lane_cfg = cfg.get("lane_marking", {})
    stop_cfg = cfg.get("stop_line", {})

    image_layers_all: Dict[str, List[gpd.GeoDataFrame]] = {
        "lane_marking_img": [],
        "stop_line_img": [],
        "crosswalk_img": [],
        "road_surface_img": [],
    }
    image_layer_map = {
        "lane_marking": "lane_marking_img",
        "stop_line": "stop_line_img",
        "crosswalk": "crosswalk_img",
    }
    road_candidates = [
        "road_polygon_utm32.gpkg",
        "road_polygon_utm32.geojson",
        "road_polygon.geojson",
        "road_polygon_wgs84.geojson",
    ]
    lidar_layers_all: Dict[str, List[gpd.GeoDataFrame]] = {
        "road_surface_lidar": [],
        "lane_marking_lidar": [],
        "crosswalk_lidar": [],
        "stop_line_lidar": [],
    }
    entity_layers: Dict[str, List[dict]] = {
        "road_surface_poly": [],
        "lane_marking_line": [],
        "crosswalk_poly": [],
        "stop_line_line": [],
    }

    qa_rows = []
    lidar_rasters = []

    for drive_id, frames in by_drive.items():
        image_present = False
        image_layers = {}
        image_layers_drive: Dict[str, List[gpd.GeoDataFrame]] = {
            "lane_marking_img": [],
            "stop_line_img": [],
            "crosswalk_img": [],
            "road_surface_img": [],
        }

        def _push_layer(layer_name: str, target_name: str, frame_id: str) -> None:
            gdf = image_layers.get(layer_name)
            if gdf is None or gdf.empty:
                return
            gdf = gdf.copy()
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:32632")
            if "frame_id" not in gdf.columns:
                gdf["frame_id"] = frame_id
            image_layers_all[target_name].append(gdf)
            image_layers_drive[target_name].append(gdf)

        if image_evidence_all:
            image_present = True
            for layer_name, gdf in image_evidence_all.items():
                if layer_name.endswith("_wgs84"):
                    continue
                target = image_layer_map.get(layer_name)
                if target is None:
                    continue
                if gdf.empty:
                    continue
                if "drive_id" in gdf.columns and gdf["drive_id"].notna().any():
                    gdf_drive = gdf[gdf["drive_id"] == drive_id]
                else:
                    gdf_drive = gdf
                if gdf_drive.empty:
                    continue
                gdf_drive = gdf_drive.copy()
                if gdf_drive.crs is None:
                    gdf_drive = gdf_drive.set_crs("EPSG:32632")
                if "frame_id" not in gdf_drive.columns:
                    gdf_drive["frame_id"] = ""
                image_layers_all[target].append(gdf_drive)
                image_layers_drive[target].append(gdf_drive)
        else:
            for row in frames:
                frame_id = str(row.get("frame_id"))
                frame_dir = image_map_root / drive_id / frame_id
                map_path = frame_dir / "map_evidence_utm32.gpkg"
                if not map_path.exists():
                    continue
                image_present = True
                image_layers = _read_map_evidence(map_path)
                _push_layer("lane_marking", "lane_marking_img", frame_id)
                _push_layer("stop_line", "stop_line_img", frame_id)
                _push_layer("crosswalk", "crosswalk_img", frame_id)

        road_poly = None
        road_path = None
        for name in road_candidates:
            candidate = Path(args.road_root) / drive_id / "geom_outputs" / name
            if candidate.exists():
                road_path = candidate
                break
        if road_path is not None:
            road_gdf = gpd.read_file(road_path)
            if not road_gdf.empty:
                if "wgs84" in road_path.name.lower():
                    road_gdf = road_gdf.set_crs("EPSG:4326", allow_override=True).to_crs("EPSG:32632")
                elif road_gdf.crs is None:
                    road_gdf = road_gdf.set_crs("EPSG:32632")
                road_poly = road_gdf.geometry.union_all()

        lidar_points = []
        lidar_intensity = []
        for row in frames:
            frame_id = str(row.get("frame_id"))
            try:
                pts = load_kitti360_lidar_points_world(data_root, drive_id, frame_id)
            except Exception as exc:
                log.warning("lidar missing: %s %s (%s)", drive_id, frame_id, exc)
                continue
            if pts.size == 0:
                continue
            mask = (pts[:, 2] >= args.lidar_z_min) & (pts[:, 2] <= args.lidar_z_max)
            pts = pts[mask]
            if pts.shape[0] == 0:
                continue
            if pts.shape[0] > args.lidar_max_points_per_frame:
                idx = np.random.choice(pts.shape[0], size=args.lidar_max_points_per_frame, replace=False)
                pts = pts[idx]
            lidar_points.append(pts[:, :3])
            lidar_intensity.append(np.ones(pts.shape[0], dtype=float))

        if lidar_points:
            pts = np.vstack(lidar_points)
            vals = np.concatenate(lidar_intensity)
            raster, bounds, res = _rasterize_points(pts[:, :2], vals, args.lidar_grid_m, args.lidar_min_points)
            surface_poly = _cells_to_polygons(raster, bounds, res)
            if road_poly is not None and not surface_poly.is_empty:
                surface_poly = surface_poly.intersection(road_poly)
            gdf_surface = gpd.GeoDataFrame(
                [
                    {
                        "drive_id": drive_id,
                        "geometry": surface_poly,
                    }
                ],
                geometry="geometry",
                crs="EPSG:32632",
            )
            lidar_layers_all["road_surface_lidar"].append(gdf_surface)

            try:
                import rasterio
                from rasterio.transform import from_origin

                minx, miny, maxx, maxy = bounds
                transform = from_origin(minx, maxy + res, res, res)
                tif_intensity = outputs_dir / f"lidar_intensity_utm32_{drive_id}.tif"
                tif_height = outputs_dir / f"lidar_height_utm32_{drive_id}.tif"
                with rasterio.open(
                    tif_intensity,
                    "w",
                    driver="GTiff",
                    height=raster.shape[0],
                    width=raster.shape[1],
                    count=1,
                    dtype=raster.dtype,
                    crs="EPSG:32632",
                    transform=transform,
                    nodata=0.0,
                ) as dst:
                    dst.write(raster, 1)
                height_vals = raster.copy()
                with rasterio.open(
                    tif_height,
                    "w",
                    driver="GTiff",
                    height=height_vals.shape[0],
                    width=height_vals.shape[1],
                    count=1,
                    dtype=height_vals.dtype,
                    crs="EPSG:32632",
                    transform=transform,
                    nodata=0.0,
                ) as dst:
                    dst.write(height_vals, 1)
                lidar_rasters.append(str(tif_intensity))
                lidar_rasters.append(str(tif_height))
            except Exception as exc:
                log.warning("raster write failed: %s", exc)

        def _make_entities_for_class(
            layer_key: str,
            entity_type: str,
            line_merge: bool,
            agg_cfg: dict,
        ) -> None:
            gdf_list = image_layers_drive.get(layer_key, [])
            if not gdf_list:
                return
            gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), geometry="geometry", crs="EPSG:32632")
            if line_merge:
                gdf = _to_lines(gdf, args.line_min_length_m)
                gdf = _aggregate_lines(
                    gdf,
                    max_merge_dist_m=float(agg_cfg.get("max_merge_dist_m", 1.2)),
                    max_gap_m=float(agg_cfg.get("max_gap_m", 2.5)),
                    max_angle_diff_deg=float(agg_cfg.get("max_angle_diff_deg", 15.0)),
                    min_length_m=float(agg_cfg.get("min_length_m", 3.0)),
                    simplify_tol_m=float(agg_cfg.get("simplify_tol_m", 0.2)),
                )
            clusters = _cluster_geoms(gdf, args.cluster_buffer_m)
            entities = _build_entities_from_clusters(
                gdf,
                clusters,
                entity_type,
                drive_id,
                line_merge=line_merge,
                qa_flag="needs_review",
                evidence_sources={"image": True, "lidar": False, "aerial": False},
                created_from_run=str(run_dir),
            )
            entity_layers[entity_type + ("_line" if line_merge else "_poly")].extend(entities)

        _make_entities_for_class("lane_marking_img", "lane_marking", True, lane_cfg)
        _make_entities_for_class("stop_line_img", "stop_line", True, stop_cfg)
        _make_entities_for_class("crosswalk_img", "crosswalk", False, {})

        if road_poly is not None and not road_poly.is_empty:
            evidence_sources = {"image": image_present, "lidar": bool(lidar_points), "aerial": False}
            confidence = 0.9 if evidence_sources["image"] and evidence_sources["lidar"] else 0.7
            entity_layers["road_surface_poly"].append(
                {
                    "geometry": road_poly,
                    "properties": {
                        "entity_id": _entity_id(f"{drive_id}_road_surface", 0),
                        "entity_type": "road_surface",
                        "confidence": confidence,
                        "evidence_sources": json.dumps(evidence_sources, ensure_ascii=True),
                        "frames_hit": len(frames),
                        "provider_hits": json.dumps({"road_polygon": 1}, ensure_ascii=True),
                        "created_from_run": str(run_dir),
                        "qa_flag": "ok" if evidence_sources["lidar"] else "needs_review",
                    },
                }
            )

        transformer = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
        for row in frames:
            frame_id = str(row.get("frame_id"))
            image_path = str(row.get("image_path") or "")
            try:
                x, y, _ = load_kitti360_pose(data_root, drive_id, frame_id)
            except Exception:
                continue
            lon, lat = transformer.transform(x, y)
            if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                continue

            overlay_dir = image_run / "debug" / args.image_provider / drive_id
            overlay_path = ""
            if overlay_dir.exists():
                matches = list(overlay_dir.glob(f"*{frame_id}*overlay*.png"))
                if matches:
                    overlay_path = str(matches[0])

            lidar_raster = outputs_dir / f"lidar_intensity_utm32_{drive_id}.tif"
            entity_ids = []
            point = Point(x, y)
            for layer_name, feats in entity_layers.items():
                for feat in feats:
                    geom = feat["geometry"]
                    if geom is None or geom.is_empty:
                        continue
                    if geom.distance(point) <= args.qa_radius_m:
                        entity_ids.append(feat["properties"]["entity_id"])

            qa_rows.append(
                {
                    "drive_id": drive_id,
                    "frame_id": frame_id,
                    "timestamp": row.get("timestamp", ""),
                    "lon": lon,
                    "lat": lat,
                    "image_path": image_path,
                    "image_overlay_path": overlay_path,
                    "lidar_raster_path": str(lidar_raster) if lidar_raster.exists() else "",
                    "entity_ids": json.dumps(sorted(set(entity_ids)), ensure_ascii=True),
                }
            )
    entity_layers_gdf = {
        "road_surface_poly": _gdf_from_entities(entity_layers["road_surface_poly"], "EPSG:32632"),
        "lane_marking_line": _gdf_from_entities(entity_layers["lane_marking_line"], "EPSG:32632"),
        "crosswalk_poly": _gdf_from_entities(entity_layers["crosswalk_poly"], "EPSG:32632"),
        "stop_line_line": _gdf_from_entities(entity_layers["stop_line_line"], "EPSG:32632"),
    }
    _write_gpkg_layers(outputs_dir / "road_entities_utm32.gpkg", entity_layers_gdf, "EPSG:32632")

    entity_layers_wgs84 = {}
    for name, gdf in entity_layers_gdf.items():
        if gdf.empty:
            entity_layers_wgs84[name] = gdf
            continue
        gdf_wgs = gdf.to_crs("EPSG:4326")
        if not _ensure_wgs84_range(gdf_wgs):
            log.error("wgs84 range check failed: %s", name)
            return 5
        entity_layers_wgs84[name] = gdf_wgs
    _write_gpkg_layers(outputs_dir / "road_entities_wgs84.gpkg", entity_layers_wgs84, "EPSG:4326")

    image_layers_out = {}
    for name, frames_list in image_layers_all.items():
        if frames_list:
            image_layers_out[name] = gpd.GeoDataFrame(pd.concat(frames_list, ignore_index=True), geometry="geometry", crs="EPSG:32632")
        else:
            image_layers_out[name] = gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32632")
    _write_gpkg_layers(outputs_dir / "image_evidence_utm32.gpkg", image_layers_out, "EPSG:32632")

    lidar_layers_out = {}
    for name, frames_list in lidar_layers_all.items():
        if frames_list:
            lidar_layers_out[name] = gpd.GeoDataFrame(pd.concat(frames_list, ignore_index=True), geometry="geometry", crs="EPSG:32632")
        else:
            lidar_layers_out[name] = gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32632")
    _write_gpkg_layers(outputs_dir / "lidar_evidence_utm32.gpkg", lidar_layers_out, "EPSG:32632")

    aerial_layers_out = {
        "aerial_road_surface": gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32632")
    }
    _write_gpkg_layers(outputs_dir / "aerial_road_surface_utm32.gpkg", aerial_layers_out, "EPSG:32632")

    qa_gdf = gpd.GeoDataFrame(
        qa_rows,
        geometry=[Point(row["lon"], row["lat"]) for row in qa_rows],
        crs="EPSG:4326",
    )
    if not _ensure_wgs84_range(qa_gdf):
        log.error("qa_index wgs84 range check failed")
        return 6
    qa_path = outputs_dir / "qa_index_wgs84.geojson"
    qa_path.write_text(qa_gdf.to_json(), encoding="utf-8")

    report_lines = [
        "# Road Entities Report",
        "",
        f"- run_dir: {run_dir}",
        f"- drives: {len(by_drive)}",
        "",
        "## Entity Counts",
    ]
    for name, gdf in entity_layers_gdf.items():
        report_lines.append(f"- {name}: {len(gdf)}")
    report_lines.append("")
    report_lines.append("## QA Assets")
    report_lines.append(f"- qa_index: {qa_path}")
    report_lines.append(f"- image_evidence: {outputs_dir / 'image_evidence_utm32.gpkg'}")
    report_lines.append(f"- lidar_evidence: {outputs_dir / 'lidar_evidence_utm32.gpkg'}")
    report_lines.append(f"- road_entities: {outputs_dir / 'road_entities_utm32.gpkg'}")
    if lidar_rasters:
        report_lines.append(f"- lidar_rasters: {', '.join(sorted(set(lidar_rasters)))}")
    report_lines.append("")
    report_lines.append("## Fragmentation Stats (After)")
    lane_stats = _line_stats_by_drive(entity_layers_gdf["lane_marking_line"])
    stop_stats = _line_stats_by_drive(entity_layers_gdf["stop_line_line"])
    report_lines.append("### lane_marking")
    report_lines.extend(_format_stats(lane_stats))
    report_lines.append("### stop_line")
    report_lines.extend(_format_stats(stop_stats))
    report_lines.append("")
    report_lines.append("## Needs Review (Top 10)")
    needs_review = []
    for layer_name, feats in entity_layers.items():
        for feat in feats:
            if feat["properties"].get("qa_flag") in {"needs_review", "weak_only", "conflict"}:
                needs_review.append((layer_name, feat["properties"]["entity_id"]))
    for layer_name, entity_id in needs_review[:10]:
        report_lines.append(f"- {layer_name}: {entity_id}")
    if not needs_review:
        report_lines.append("- none")
    report_lines.append("")

    baseline_path = outputs_dir.parent / "baseline_road_entities_utm32.gpkg"
    if not baseline_path.exists():
        run_baseline = Path("runs") / "road_entities_baseline_utm32.gpkg"
        if run_baseline.exists():
            baseline_path = run_baseline
    if baseline_path.exists():
        try:
            base_layers = _read_map_evidence(baseline_path)
            report_lines.append("## Fragmentation Stats (Before)")
            base_lane = base_layers.get("lane_marking_line")
            base_stop = base_layers.get("stop_line_line")
            if base_lane is not None:
                base_lane_stats = _line_stats_by_drive(base_lane)
                report_lines.append("### lane_marking")
                report_lines.extend(_format_stats(base_lane_stats))
            if base_stop is not None:
                base_stop_stats = _line_stats_by_drive(base_stop)
                report_lines.append("### stop_line")
                report_lines.extend(_format_stats(base_stop_stats))
            report_lines.append("")
        except Exception as exc:
            report_lines.append(f"- baseline_read_failed: {exc}")
    (outputs_dir / "road_entities_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    log.info("road entities written: %s", outputs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

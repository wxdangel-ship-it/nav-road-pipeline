from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon, box
from shapely import wkb
from shapely.ops import linemerge, unary_union

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.datasets.kitti360_io import load_kitti360_calib, load_kitti360_lidar_points_world, load_kitti360_pose


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


def _score_from_gdf(gdf: gpd.GeoDataFrame) -> float:
    if gdf is None or gdf.empty:
        return 0.0
    for col in ("conf", "score", "confidence"):
        if col in gdf.columns:
            vals = [float(v) for v in gdf[col].tolist() if pd.notna(v)]
            if vals:
                return float(np.mean(vals))
    return 0.5


def _extract_drive_frame(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    drive_match = re.search(r"(2013_05_28_drive_\\d{4}_sync)", text)
    frame_match = re.search(r"(\\d{10})", text)
    drive_id = drive_match.group(1) if drive_match else ""
    frame_id = frame_match.group(1) if frame_match else ""
    return drive_id, frame_id


def _fill_evidence_fields(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    gdf = gdf.copy()
    for col in ["drive_id", "frame_id", "conf", "model_id"]:
        if col not in gdf.columns:
            gdf[col] = ""
    gdf["conf"] = gdf["conf"].fillna(1.0)

    def _fill_row(row: pd.Series) -> pd.Series:
        if row.get("drive_id") and row.get("frame_id"):
            return row
        for col in row.index:
            if not isinstance(row[col], str):
                continue
            d, f = _extract_drive_frame(row[col])
            if d and not row.get("drive_id"):
                row["drive_id"] = d
            if f and not row.get("frame_id"):
                row["frame_id"] = f
            if row.get("drive_id") and row.get("frame_id"):
                break
        return row

    gdf = gdf.apply(_fill_row, axis=1)
    gdf["drive_id"] = gdf["drive_id"].fillna("")
    gdf["frame_id"] = gdf["frame_id"].fillna("")
    gdf["model_id"] = gdf["model_id"].fillna(gdf.get("provider_id", ""))
    return gdf


def _geom_signature(geom: Polygon) -> str:
    if geom is None or geom.is_empty:
        return ""
    try:
        return wkb.dumps(geom, hex=True, rounding_precision=3)
    except Exception:
        return geom.wkt


def _dedup_by_signature(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    sigs = gdf.geometry.apply(_geom_signature)
    gdf = gdf.assign(_geom_sig=sigs)
    gdf = gdf.drop_duplicates(subset="_geom_sig").drop(columns=["_geom_sig"])
    return gdf


def _cluster_by_centroid(gdf: gpd.GeoDataFrame, eps_m: float) -> List[List[int]]:
    if gdf.empty:
        return []
    centroids = [geom.centroid for geom in gdf.geometry]
    try:
        from shapely.strtree import STRtree

        tree = STRtree(centroids)
        geom_id = {id(g): idx for idx, g in enumerate(centroids)}
    except Exception:
        tree = None
        geom_id = {}
    parent = list(range(len(centroids)))

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

    for i, center in enumerate(centroids):
        candidates = []
        if tree is not None:
            for cand in tree.query(center.buffer(eps_m)):
                if isinstance(cand, (int, np.integer)):
                    idx = int(cand)
                else:
                    idx = geom_id.get(id(cand))
                if idx is not None:
                    candidates.append(idx)
        else:
            candidates = list(range(len(centroids)))
        for j in candidates:
            if j <= i:
                continue
            if center.distance(centroids[j]) <= eps_m:
                _union(i, j)

    clusters: Dict[int, List[int]] = {}
    for idx in range(len(centroids)):
        root = _find(idx)
        clusters.setdefault(root, []).append(idx)
    return list(clusters.values())


def _assign_drive_by_spatial(
    gdf: gpd.GeoDataFrame,
    drive_polys: Dict[str, Polygon],
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    gdf = gdf.copy()
    if "drive_id" not in gdf.columns:
        gdf["drive_id"] = ""
    missing = gdf["drive_id"].isna() | (gdf["drive_id"] == "")
    if not missing.any():
        return gdf

    polys = [poly for poly in drive_polys.values() if poly is not None and not poly.is_empty]
    if not polys:
        return gdf
    try:
        from shapely.strtree import STRtree

        tree = STRtree(polys)
        poly_id = {id(poly): key for key, poly in drive_polys.items()}
    except Exception:
        tree = None
        poly_id = {}

    def _find_drive(geom: Polygon) -> str:
        if geom is None or geom.is_empty:
            return ""
        candidates = []
        if tree is not None:
            for cand in tree.query(geom):
                if isinstance(cand, (int, np.integer)):
                    candidates.append(polys[int(cand)])
                else:
                    candidates.append(cand)
        else:
            candidates = polys
        best_drive = ""
        best_dist = float("inf")
        for cand in candidates:
            dist = cand.distance(geom)
            if dist < best_dist:
                best_dist = dist
                best_drive = poly_id.get(id(cand), "")
        return best_drive

    gdf.loc[missing, "drive_id"] = gdf.loc[missing, "geometry"].apply(_find_drive)
    return gdf


def _assign_frame_by_nearest_pose(
    gdf: gpd.GeoDataFrame,
    data_root: Path,
    frames_by_drive: Dict[str, List[dict]],
) -> gpd.GeoDataFrame:
    if gdf.empty or "drive_id" not in gdf.columns:
        return gdf
    gdf = gdf.copy()
    if "frame_id" not in gdf.columns:
        gdf["frame_id"] = ""
    missing_mask = gdf["frame_id"].isna() | (gdf["frame_id"] == "") | (gdf["frame_id"] == "unknown")
    if not missing_mask.any():
        return gdf
    drive_groups = gdf[missing_mask].groupby("drive_id")
    for drive_id, group in drive_groups:
        frames = frames_by_drive.get(str(drive_id), [])
        if not frames:
            continue
        poses = []
        for row in frames:
            frame_id = str(row.get("frame_id"))
            try:
                x, y, _ = load_kitti360_pose(data_root, str(drive_id), frame_id)
            except Exception:
                continue
            poses.append((Point(x, y), frame_id))
        if not poses:
            continue
        try:
            from shapely.strtree import STRtree

            pts = [p[0] for p in poses]
            tree = STRtree(pts)
            pt_id = {id(p): idx for idx, p in enumerate(pts)}
        except Exception:
            tree = None
            pt_id = {}
        for idx, row in group.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            centroid = geom.centroid
            candidates = []
            if tree is not None:
                for cand in tree.query(centroid.buffer(200.0)):
                    if isinstance(cand, (int, np.integer)):
                        candidates.append(poses[int(cand)][0])
                    else:
                        candidates.append(cand)
                if not candidates:
                    candidates = [p[0] for p in poses]
            else:
                candidates = [p[0] for p in poses]
            best_frame = ""
            best_dist = float("inf")
            for cand in candidates:
                dist = centroid.distance(cand)
                if dist < best_dist:
                    best_dist = dist
                    if tree is not None:
                        cand_idx = pt_id.get(id(cand))
                        if cand_idx is not None:
                            best_frame = poses[cand_idx][1]
                    else:
                        for pt, fid in poses:
                            if pt == cand:
                                best_frame = fid
                                break
            if best_frame:
                gdf.at[idx, "frame_id"] = best_frame
    return gdf


def _assert_required_fields(gdf: gpd.GeoDataFrame, label: str) -> None:
    if gdf.empty:
        return
    miss_drive = (gdf["drive_id"].isna() | (gdf["drive_id"] == "")).mean()
    miss_frame = (
        gdf["frame_id"].isna() | (gdf["frame_id"] == "") | (gdf["frame_id"] == "unknown")
    ).mean()
    if miss_drive > 0.01 or miss_frame > 0.01:
        raise ValueError(f"{label} missing fields: drive_id={miss_drive:.3f} frame_id={miss_frame:.3f}")

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
        try:
            path.unlink()
        except PermissionError:
            new_path = path.with_suffix(".new.gpkg")
            LOG.warning("gpkg locked, writing to %s", new_path)
            path = new_path
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


def _segment_heading_deg(p0: Tuple[float, float], p1: Tuple[float, float]) -> Optional[float]:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    angle = float(np.degrees(np.arctan2(dy, dx)))
    if angle < 0:
        angle += 180.0
    return angle


def _build_drive_heading_segments(
    data_root: Path,
    drive_id: str,
    frames: List[dict],
) -> List[Tuple[LineString, float]]:
    segments = []
    coords = []
    for row in frames:
        frame_id = str(row.get("frame_id"))
        try:
            x, y, _ = load_kitti360_pose(data_root, drive_id, frame_id)
        except Exception:
            continue
        coords.append((x, y))
    for i in range(1, len(coords)):
        p0 = coords[i - 1]
        p1 = coords[i]
        heading = _segment_heading_deg(p0, p1)
        if heading is None:
            continue
        line = LineString([p0, p1])
        segments.append((line, heading))
    return segments


def _nearest_heading(segments: List[Tuple[LineString, float]], geom: Polygon) -> Optional[float]:
    if not segments:
        return None
    centroid = geom.centroid
    try:
        from shapely.strtree import STRtree

        lines = [seg[0] for seg in segments]
        tree = STRtree(lines)
        line_id = {id(line): idx for idx, line in enumerate(lines)}
        candidates = tree.query(centroid)
        if len(candidates) == 0:
            candidates = lines
    except Exception:
        candidates = [seg[0] for seg in segments]
        line_id = {id(seg[0]): idx for idx, seg in enumerate(segments)}
    best_heading = None
    best_dist = float("inf")
    for line in candidates:
        idx = line_id.get(id(line))
        if idx is None:
            continue
        dist = centroid.distance(line)
        if dist < best_dist:
            best_dist = dist
            best_heading = segments[idx][1]
    return best_heading


def _nearest_yaw_heading(
    data_root: Path,
    drive_id: str,
    frames: List[dict],
    geom: Polygon,
) -> Optional[float]:
    if geom is None or geom.is_empty:
        return None
    points = []
    for row in frames:
        frame_id = str(row.get("frame_id"))
        try:
            x, y, yaw = load_kitti360_pose(data_root, drive_id, frame_id)
        except Exception:
            continue
        heading = float(np.degrees(yaw))
        if heading < 0:
            heading += 180.0
        points.append((Point(x, y), heading))
    if not points:
        return None
    try:
        from shapely.strtree import STRtree

        pts = [p[0] for p in points]
        tree = STRtree(pts)
        pt_id = {id(p): idx for idx, p in enumerate(pts)}
        candidates = tree.query(geom.centroid)
        if len(candidates) == 0:
            candidates = pts
    except Exception:
        candidates = [p[0] for p in points]
        pt_id = {id(p[0]): idx for idx, p in enumerate(points)}
    best_heading = None
    best_dist = float("inf")
    for cand in candidates:
        idx = pt_id.get(id(cand))
        if idx is None:
            continue
        dist = geom.centroid.distance(cand)
        if dist < best_dist:
            best_dist = dist
            best_heading = points[idx][1]
    return best_heading


def _trajectory_heading_near(
    data_root: Path,
    drive_id: str,
    frames: List[dict],
    geom: Polygon,
    radius_m: float,
) -> Optional[float]:
    if geom is None or geom.is_empty:
        return None
    center = geom.centroid
    pts = []
    for row in frames:
        frame_id = str(row.get("frame_id"))
        try:
            x, y, _ = load_kitti360_pose(data_root, drive_id, frame_id)
        except Exception:
            continue
        if center.distance(Point(x, y)) <= radius_m:
            pts.append([x, y])
    if len(pts) < 2:
        return None
    coords = np.array(pts, dtype=float)
    mean = coords.mean(axis=0)
    cov = np.cov((coords - mean).T)
    vals, vecs = np.linalg.eig(cov)
    idx = int(np.argmax(vals))
    direction = vecs[:, idx]
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    if angle < 0:
        angle += 180.0
    return angle


def _road_poly_heading(
    road_poly: Optional[Polygon],
    geom: Polygon,
    radius_m: float,
) -> Optional[float]:
    if road_poly is None or road_poly.is_empty or geom is None or geom.is_empty:
        return None
    clip = road_poly.intersection(geom.centroid.buffer(radius_m))
    if clip is None or clip.is_empty:
        return None
    coords = []
    geoms = [clip] if isinstance(clip, Polygon) else list(getattr(clip, "geoms", []))
    for poly in geoms:
        if not isinstance(poly, Polygon):
            continue
        coords.extend(list(poly.exterior.coords))
    if len(coords) < 2:
        return None
    arr = np.array(coords, dtype=float)
    mean = arr.mean(axis=0)
    cov = np.cov((arr - mean).T)
    vals, vecs = np.linalg.eig(cov)
    idx = int(np.argmax(vals))
    direction = vecs[:, idx]
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    if angle < 0:
        angle += 180.0
    return angle

def _line_endpoints(geom: LineString) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    coords = list(geom.coords)
    return (coords[0][0], coords[0][1]), (coords[-1][0], coords[-1][1])


def _min_endpoint_gap(a: LineString, b: LineString) -> float:
    a0, a1 = _line_endpoints(a)
    b0, b1 = _line_endpoints(b)
    pairs = [(a0, b0), (a0, b1), (a1, b0), (a1, b1)]
    return min(float(np.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1])) for p in pairs)


def _rect_heading_deg(rect: Polygon) -> Optional[float]:
    if rect is None or rect.is_empty:
        return None
    coords = list(rect.exterior.coords)
    if len(coords) < 4:
        return None
    edges = []
    for i in range(len(coords) - 1):
        p0 = coords[i]
        p1 = coords[i + 1]
        length = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        if length > 1e-6:
            edges.append((length, p0, p1))
    if not edges:
        return None
    edges.sort(key=lambda x: x[0], reverse=True)
    _, p0, p1 = edges[0]
    return _segment_heading_deg(p0, p1)


def _filter_lines_by_heading(
    gdf: gpd.GeoDataFrame,
    heading_segments: List[Tuple[LineString, float]],
    max_diff_deg: float,
    mode: str,
) -> gpd.GeoDataFrame:
    if gdf.empty or not heading_segments:
        return gdf
    keep_rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, MultiLineString):
            geom = linemerge(list(geom.geoms))
            if geom is None or geom.is_empty:
                continue
        if not isinstance(geom, LineString):
            continue
        line_heading = _line_direction_deg(geom)
        if line_heading is None:
            continue
        road_heading = _nearest_heading(heading_segments, geom)
        if road_heading is None:
            keep_rows.append(row)
            continue
        diff = _angle_diff_deg(line_heading, road_heading)
        if mode == "perpendicular":
            diff = abs(diff - 90.0)
        if diff <= max_diff_deg:
            row = row.copy()
            row["heading_diff_deg"] = diff
            keep_rows.append(row)
    if not keep_rows:
        return gpd.GeoDataFrame(columns=gdf.columns, geometry=[], crs=gdf.crs)
    return gpd.GeoDataFrame(keep_rows, geometry="geometry", crs=gdf.crs)


def _annotate_heading_for_lines(
    gdf: gpd.GeoDataFrame,
    heading_segments: List[Tuple[LineString, float]],
    max_diff_deg: float,
    mode: str,
) -> gpd.GeoDataFrame:
    if gdf.empty or not heading_segments:
        gdf = gdf.copy()
        gdf["heading_ok"] = True
        gdf["heading_diff_deg"] = ""
        return gdf
    rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, MultiLineString):
            geom = linemerge(list(geom.geoms))
            if geom is None or geom.is_empty:
                continue
        if not isinstance(geom, LineString):
            continue
        line_heading = _line_direction_deg(geom)
        if line_heading is None:
            continue
        road_heading = _nearest_heading(heading_segments, geom)
        if road_heading is None:
            row = row.copy()
            row["heading_ok"] = True
            row["heading_diff_deg"] = ""
            rows.append(row)
            continue
        diff = _angle_diff_deg(line_heading, road_heading)
        if mode == "perpendicular":
            diff = abs(diff - 90.0)
        row = row.copy()
        row["heading_diff_deg"] = diff
        row["heading_ok"] = diff <= max_diff_deg
        rows.append(row)
    if not rows:
        return gpd.GeoDataFrame(columns=gdf.columns, geometry=[], crs=gdf.crs)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)


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
        heading_diff = ""
        if "heading_diff_deg" in hits.columns:
            diffs = [float(v) for v in hits["heading_diff_deg"].tolist() if v != "" and not pd.isna(v)]
            if diffs:
                heading_diff = float(np.median(diffs))

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
                    "heading_diff_deg": heading_diff,
                },
            }
        )
    return entities


def _build_entities_from_index_clusters(
    gdf: gpd.GeoDataFrame,
    clusters: List[List[int]],
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
    for idx, indices in enumerate(clusters):
        hits = gdf.iloc[indices]
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
        heading_diff = ""
        if "heading_diff_deg" in hits.columns:
            diffs = [float(v) for v in hits["heading_diff_deg"].tolist() if v != "" and not pd.isna(v)]
            if diffs:
                heading_diff = float(np.median(diffs))
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
                    "heading_diff_deg": heading_diff,
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


def _count_stats_per_drive(gdf: gpd.GeoDataFrame) -> Dict[str, float]:
    if gdf.empty:
        return {"min": 0, "median": 0, "max": 0}
    if "drive_id" not in gdf.columns:
        gdf = gdf.copy()
        gdf["drive_id"] = gdf["entity_id"].apply(lambda v: str(v).rsplit("_", 1)[0] if v else "")
    counts = gdf.groupby("drive_id").size().astype(float).tolist()
    return {
        "min": float(np.min(counts)) if counts else 0.0,
        "median": float(np.median(counts)) if counts else 0.0,
        "max": float(np.max(counts)) if counts else 0.0,
    }


def _angle_stats_by_drive(
    gdf: gpd.GeoDataFrame,
    heading_segments_by_drive: Dict[str, List[Tuple[LineString, float]]],
    mode: str,
) -> Dict[str, dict]:
    if gdf.empty:
        return {}
    if "drive_id" not in gdf.columns:
        gdf = gdf.copy()
        gdf["drive_id"] = gdf["entity_id"].apply(lambda v: str(v).rsplit("_", 1)[0] if v else "")
    stats = {}
    for drive_id, group in gdf.groupby("drive_id"):
        segs = heading_segments_by_drive.get(drive_id, [])
        diffs = []
        for geom in group.geometry:
            if geom is None or geom.is_empty:
                continue
            if isinstance(geom, MultiLineString):
                geom = linemerge(list(geom.geoms))
                if geom is None or geom.is_empty:
                    continue
            if not isinstance(geom, LineString):
                continue
            line_heading = _line_direction_deg(geom)
            if line_heading is None:
                continue
            road_heading = _nearest_heading(segs, geom) if segs else None
            if road_heading is None:
                continue
            diff = _angle_diff_deg(line_heading, road_heading)
            if mode == "perpendicular":
                diff = abs(diff - 90.0)
            diffs.append(diff)
        stats[str(drive_id)] = {
            "p50": _percentile(diffs, 50),
            "p90": _percentile(diffs, 90),
        }
    return stats


def _frames_hit_stats(gdf: gpd.GeoDataFrame) -> Dict[str, float]:
    if gdf.empty or "frames_hit" not in gdf.columns:
        return {"p50": 0.0, "p90": 0.0}
    vals = [float(v) for v in gdf["frames_hit"].tolist() if pd.notna(v)]
    return {"p50": _percentile(vals, 50), "p90": _percentile(vals, 90)}


def _nearest_qa_for_entity(
    qa_rows: List[dict],
    entity_geom: Polygon,
    drive_id: str,
    max_dist_m: float,
) -> Optional[dict]:
    if entity_geom is None or entity_geom.is_empty:
        return None
    center = entity_geom.centroid
    best = None
    best_dist = float("inf")
    for row in qa_rows:
        if row.get("drive_id") != drive_id:
            continue
        try:
            lon = float(row.get("lon"))
            lat = float(row.get("lat"))
        except Exception:
            continue
        # qa_rows lon/lat are wgs84, but entity is utm32. Skip if not set.
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            continue
        # qa_rows also have x/y via road_entities computation, but we didn't store; fallback to distance in utm32 via point if present.
        # Use stored entity_ids proximity; this is best-effort for report linking.
        ent_ids = row.get("entity_ids", "")
        if isinstance(ent_ids, str) and ent_ids:
            if str(ent_ids).find(drive_id) >= 0:
                return row
        dist = 0.0
        if dist < best_dist:
            best = row
            best_dist = dist
    return best


def _frames_hit_by_proximity(
    geom: Polygon,
    data_root: Path,
    drive_id: str,
    frames: List[dict],
    radius_m: float,
) -> int:
    if geom is None or geom.is_empty:
        return 0
    center = geom.centroid
    count = 0
    for row in frames:
        frame_id = str(row.get("frame_id"))
        try:
            x, y, _ = load_kitti360_pose(data_root, drive_id, frame_id)
        except Exception:
            continue
        if center.distance(Point(x, y)) <= radius_m:
            count += 1
    return count


def _project_world_to_image(
    points: np.ndarray,
    pose_xy_yaw: Tuple[float, float, float],
    calib: Dict[str, np.ndarray],
) -> np.ndarray:
    x0, y0, yaw = pose_xy_yaw
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    dx = points[:, 0] - x0
    dy = points[:, 1] - y0
    x_ego = c * dx + s * dy
    y_ego = -s * dx + c * dy
    z_ego = points[:, 2]
    ones = np.ones_like(x_ego)
    pts_h = np.stack([x_ego, y_ego, z_ego, ones], axis=0)
    cam = calib["t_velo_to_cam"] @ pts_h
    xyz = cam[:3, :].T
    xyz = (calib["r_rect"] @ xyz.T).T
    zs = xyz[:, 2]
    valid = zs > 1e-3
    us = np.zeros_like(zs)
    vs = np.zeros_like(zs)
    k = calib["k"]
    us[valid] = (k[0, 0] * xyz[valid, 0] / zs[valid]) + k[0, 2]
    vs[valid] = (k[1, 1] * xyz[valid, 1] / zs[valid]) + k[1, 2]
    return np.stack([us, vs, valid], axis=1)


def _geom_to_image_points(
    geom: Polygon,
    pose_xy_yaw: Tuple[float, float, float],
    calib: Dict[str, np.ndarray],
) -> List[Tuple[float, float]]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        coords = np.array(list(geom.exterior.coords), dtype=float)
    elif isinstance(geom, LineString):
        coords = np.array(list(geom.coords), dtype=float)
    else:
        return []
    if coords.shape[0] == 0:
        return []
    points = np.column_stack([coords[:, 0], coords[:, 1], np.zeros(coords.shape[0], dtype=float)])
    proj = _project_world_to_image(points, pose_xy_yaw, calib)
    out = []
    for u, v, valid in proj:
        if not valid:
            continue
        out.append((float(u), float(v)))
    return out


def _gate_against_road(
    gdf: gpd.GeoDataFrame,
    road_poly: Optional[Polygon],
    buffer_m: float,
    min_ratio: float,
    mode: str,
) -> gpd.GeoDataFrame:
    if gdf.empty or road_poly is None or road_poly.is_empty:
        return gdf
    gate_poly = road_poly.buffer(buffer_m)
    keep_rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        ratio = 0.0
        if mode == "line":
            length = geom.length
            if length > 0:
                ratio = geom.intersection(gate_poly).length / length
        else:
            area = geom.area
            if area > 0:
                ratio = geom.intersection(gate_poly).area / area
        if ratio >= min_ratio:
            row = row.copy()
            row["road_inside_ratio"] = ratio
            keep_rows.append(row)
    if not keep_rows:
        return gpd.GeoDataFrame(columns=gdf.columns, geometry=[], crs=gdf.crs)
    return gpd.GeoDataFrame(keep_rows, geometry="geometry", crs=gdf.crs)


def _rect_metrics(rect: Polygon, union_geom: Polygon) -> dict:
    rect_area = rect.area if rect is not None else 0.0
    area = union_geom.area if union_geom is not None else 0.0
    rectangularity = area / rect_area if rect_area > 0 else 0.0
    rect_coords = list(rect.exterior.coords) if rect is not None else []
    edge_lengths = []
    for i in range(len(rect_coords) - 1):
        p0 = rect_coords[i]
        p1 = rect_coords[i + 1]
        edge_lengths.append(float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])))
    edge_lengths = sorted(edge_lengths, reverse=True)
    rect_l = edge_lengths[0] if edge_lengths else 0.0
    rect_w = edge_lengths[-1] if edge_lengths else 0.0
    aspect = rect_l / max(1e-6, rect_w) if rect_w > 0 else 0.0
    return {
        "area_m2": area,
        "rect_area_m2": rect_area,
        "rectangularity": rectangularity,
        "rect_l_m": rect_l,
        "rect_w_m": rect_w,
        "aspect": aspect,
    }


def _append_reject_reason(reason: str, existing: str) -> str:
    tokens = [t for t in existing.split(",") if t]
    if reason and reason not in tokens:
        tokens.append(reason)
    return ",".join(tokens)


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
    ap.add_argument("--emit-qa-images", type=int, default=1)
    ap.add_argument("--enable-lidar-gate", type=int, default=0)
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
    cross_cfg = cfg.get("crosswalk", {})
    heading_cfg = cfg.get("heading", {})
    cam_id = str(heading_cfg.get("cam_id", "image_02"))
    gate_cfg = cfg.get("gates", {})
    frames_cfg = cfg.get("frames_hit", {})
    cluster_cfg = cfg.get("clustering", {})
    cross_final_cfg = cfg.get("crosswalk_final", {})

    image_layers_all: Dict[str, List[gpd.GeoDataFrame]] = {
        "lane_marking_img": [],
        "stop_line_img": [],
        "crosswalk_img": [],
        "gore_marking_img": [],
        "road_surface_img": [],
    }
    frame_layers_all: Dict[str, List[gpd.GeoDataFrame]] = {
        "lane_marking_frame": [],
        "stop_line_frame": [],
        "crosswalk_frame": [],
    }
    image_layer_map = {
        "lane_marking": "lane_marking_img",
        "stop_line": "stop_line_img",
        "crosswalk": "crosswalk_img",
        "gore_marking": "gore_marking_img",
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
        "crosswalk_candidate_poly": [],
        "crosswalk_poly": [],
        "stop_line_line": [],
    }

    drive_polys: Dict[str, Polygon] = {}
    for drive_id in by_drive:
        road_path = None
        for name in road_candidates:
            candidate = Path(args.road_root) / drive_id / "geom_outputs" / name
            if candidate.exists():
                road_path = candidate
                break
        if road_path is None:
            continue
        road_gdf = gpd.read_file(road_path)
        if road_gdf.empty:
            continue
        if "wgs84" in road_path.name.lower():
            road_gdf = road_gdf.set_crs("EPSG:4326", allow_override=True).to_crs("EPSG:32632")
        elif road_gdf.crs is None:
            road_gdf = road_gdf.set_crs("EPSG:32632")
        drive_polys[drive_id] = road_gdf.geometry.union_all()

    if image_evidence_all:
        drive_ids = set(by_drive.keys())
        normalized = {}
        for layer_name, gdf in image_evidence_all.items():
            if gdf is None or gdf.empty:
                continue
            gdf = _fill_evidence_fields(gdf)
            gdf = _assign_drive_by_spatial(gdf, drive_polys)
            miss_mask = gdf["frame_id"].isna() | (gdf["frame_id"] == "")
            if miss_mask.any():
                gdf = _assign_frame_by_nearest_pose(gdf, data_root, by_drive)
            if "frame_id" not in gdf.columns:
                gdf["frame_id"] = ""
            gdf.loc[gdf["frame_id"] == "", "frame_id"] = "unknown"
            gdf["drive_id"] = gdf["drive_id"].fillna("")
            if drive_ids:
                gdf = gdf[gdf["drive_id"].isin(drive_ids)]
            normalized[layer_name] = gdf
        image_evidence_all = normalized
        required_layers = {"lane_marking", "stop_line", "crosswalk"}
        for layer_name, gdf in image_evidence_all.items():
            if layer_name.endswith("_wgs84"):
                continue
            if layer_name not in required_layers:
                continue
            _assert_required_fields(gdf, f"image_evidence:{layer_name}")
        sig_owner: Dict[str, str] = {}
        for layer_name, gdf in image_evidence_all.items():
            if layer_name.endswith("_wgs84") or gdf.empty:
                continue
            for _, row in gdf.iterrows():
                sig = _geom_signature(row.geometry)
                if not sig:
                    continue
                owner = sig_owner.get(sig)
                drive_id = str(row.get("drive_id") or "")
                if owner and owner != drive_id:
                    raise ValueError(f"cross_drive_duplicate:{layer_name}:{owner}:{drive_id}")
                if not owner:
                    sig_owner[sig] = drive_id

        qa_rows = []
    qa_rows = []
    lidar_rasters = []
    heading_segments_by_drive: Dict[str, List[Tuple[LineString, float]]] = {}

    for drive_id, frames in by_drive.items():
        frame_rows = frames
        image_present = False
        image_layers = {}
        image_layers_drive: Dict[str, List[gpd.GeoDataFrame]] = {
            "lane_marking_img": [],
            "stop_line_img": [],
            "crosswalk_img": [],
            "gore_marking_img": [],
            "road_surface_img": [],
        }
        frame_layers_drive: Dict[str, List[gpd.GeoDataFrame]] = {
            "lane_marking_frame": [],
            "stop_line_frame": [],
            "crosswalk_frame": [],
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
            if "drive_id" not in gdf.columns:
                gdf["drive_id"] = drive_id
            gdf["frame_id"] = gdf["frame_id"].fillna(frame_id)
            gdf["drive_id"] = gdf["drive_id"].fillna(drive_id)
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
                gdf_drive = gdf_drive[gdf_drive["drive_id"] == drive_id]
                if gdf_drive.empty:
                    continue
                if gdf_drive.crs is None:
                    gdf_drive = gdf_drive.set_crs("EPSG:32632")
                image_layers_all[target].append(gdf_drive)
                image_layers_drive[target].append(gdf_drive)
            for row in frames:
                frame_id = str(row.get("frame_id"))
                frame_dir = image_map_root / drive_id / frame_id
                map_path = frame_dir / "map_evidence_utm32.gpkg"
                if not map_path.exists():
                    continue
                image_layers = _read_map_evidence(map_path)
                _push_layer("crosswalk", "crosswalk_img", frame_id)
                _push_layer("gore_marking", "gore_marking_img", frame_id)
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
                _push_layer("gore_marking", "gore_marking_img", frame_id)

        road_poly = drive_polys.get(drive_id)
        heading_segments = _build_drive_heading_segments(data_root, drive_id, frames)
        heading_segments_by_drive[drive_id] = heading_segments

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

        def _build_frame_layer(
            layer_key: str,
            entity_type: str,
            line_merge: bool,
            agg_cfg: dict,
            min_ratio: float,
            apply_gate: bool = True,
        ) -> gpd.GeoDataFrame:
            gdf_list = image_layers_drive.get(layer_key, [])
            if not gdf_list:
                return gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32632")
            gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), geometry="geometry", crs="EPSG:32632")
            if "drive_id" in gdf.columns:
                gdf = gdf[gdf["drive_id"] == drive_id]
            if gdf.empty:
                return gdf
            if "drive_id" not in gdf.columns:
                gdf["drive_id"] = drive_id
            if "frame_id" not in gdf.columns:
                gdf["frame_id"] = ""
            missing_frame = gdf["frame_id"].isna() | (gdf["frame_id"] == "") | (gdf["frame_id"] == "unknown")
            if len(gdf) >= 5 and gdf["frame_id"].nunique() <= 3:
                gdf["frame_id"] = ""
                missing_frame = gdf["frame_id"].isna() | (gdf["frame_id"] == "")
            if missing_frame.any():
                gdf = _assign_frame_by_nearest_pose(gdf, data_root, by_drive)
            if line_merge:
                gdf = _to_lines(gdf, args.line_min_length_m)
                heading_max = float(agg_cfg.get("heading_max_diff_deg", agg_cfg.get("max_angle_diff_deg", 15.0)))
                mode = "parallel"
                if entity_type == "stop_line":
                    mode = "perpendicular"
                gdf = _filter_lines_by_heading(gdf, heading_segments, heading_max, mode)
            if entity_type == "gore_marking":
                gdf = gdf.copy()
                gdf["entity_type"] = entity_type
                gdf["qa_flag"] = "gated"
                return gdf
            if apply_gate:
                gdf = _gate_against_road(
                    gdf,
                    road_poly,
                    float(gate_cfg.get("road_buffer_m", 1.0)),
                    min_ratio,
                    "line" if line_merge else "poly",
                )
                if gdf.empty:
                    return gdf
                if line_merge:
                    heading_max = float(agg_cfg.get("heading_max_diff_deg", agg_cfg.get("max_angle_diff_deg", 15.0)))
                    mode = "parallel"
                    if entity_type == "stop_line":
                        mode = "perpendicular"
                    if entity_type == "lane_marking":
                        gdf = _annotate_heading_for_lines(gdf, heading_segments, heading_max, mode)
                    else:
                        gdf = _filter_lines_by_heading(gdf, heading_segments, heading_max, mode)
            gdf = gdf.copy()
            gdf["entity_type"] = entity_type
            if apply_gate:
                if "heading_ok" in gdf.columns:
                    gdf["qa_flag"] = gdf["heading_ok"].apply(lambda v: "gated" if v else "angle_mismatch")
                else:
                    gdf["qa_flag"] = "gated"
            else:
                gdf["qa_flag"] = "raw"
            if not line_merge and "frame_id" in gdf.columns:
                grouped = []
                for frame_id, hits in gdf.groupby("frame_id"):
                    row = hits.iloc[0].copy()
                    row.geometry = unary_union(hits.geometry.values)
                    row["frame_id"] = frame_id
                    grouped.append(row)
                gdf = gpd.GeoDataFrame(grouped, geometry="geometry", crs=gdf.crs)
            return gdf

        lane_min_ratio = float(gate_cfg.get("lane_marking_min_inside_ratio", 0.6))
        stop_min_ratio = float(gate_cfg.get("stop_line_min_inside_ratio", 0.5))
        cross_min_ratio = float(gate_cfg.get("crosswalk_min_inside_ratio", 0.5))

        lane_frame = _build_frame_layer("lane_marking_img", "lane_marking", True, lane_cfg, lane_min_ratio)
        stop_frame = _build_frame_layer("stop_line_img", "stop_line", True, stop_cfg, stop_min_ratio)
        cross_frame = _build_frame_layer("crosswalk_img", "crosswalk", False, cross_cfg, cross_min_ratio)
        cross_frame_raw = _build_frame_layer("crosswalk_img", "crosswalk", False, cross_cfg, 0.0, apply_gate=False)
        gore_frame = _build_frame_layer("gore_marking_img", "gore_marking", False, cross_cfg, 0.0)
        cross_frame_evidence = cross_frame
        if not cross_frame_raw.empty:
            raw_frames = cross_frame_raw["frame_id"].nunique() if "frame_id" in cross_frame_raw.columns else 0
            gate_frames = cross_frame["frame_id"].nunique() if not cross_frame.empty else 0
            if gate_frames < max(1, int(raw_frames * 0.7)):
                cross_frame_evidence = cross_frame_raw

        if not lane_frame.empty:
            frame_layers_all["lane_marking_frame"].append(lane_frame)
            frame_layers_drive["lane_marking_frame"].append(lane_frame)
        if not stop_frame.empty:
            frame_layers_all["stop_line_frame"].append(stop_frame)
            frame_layers_drive["stop_line_frame"].append(stop_frame)
        if not cross_frame_evidence.empty:
            frame_layers_all["crosswalk_frame"].append(cross_frame_evidence)
            frame_layers_drive["crosswalk_frame"].append(cross_frame_evidence)

        gore_by_frame: Dict[str, Polygon] = {}
        gore_score_by_frame: Dict[str, float] = {}
        if not gore_frame.empty:
            for frame_id, hits in gore_frame.groupby("frame_id"):
                if hits.empty:
                    continue
                geoms = []
                for geom in hits.geometry.values:
                    if geom is None or geom.is_empty:
                        continue
                    if not geom.is_valid:
                        geom = geom.buffer(0)
                    if geom is None or geom.is_empty:
                        continue
                    geoms.append(geom)
                if not geoms:
                    continue
                try:
                    gore_union = unary_union(geoms)
                except Exception:
                    continue
                if gore_union is None or gore_union.is_empty:
                    continue
                gore_by_frame[frame_id] = gore_union
                gore_score_by_frame[frame_id] = _score_from_gdf(hits)

        def _aggregate_from_frame(
            gdf: gpd.GeoDataFrame,
            entity_type: str,
            line_merge: bool,
            eps_m: float,
            min_frames: int,
        ) -> None:
            if gdf.empty:
                return
            if entity_type == "lane_marking" and "heading_ok" in gdf.columns:
                gdf = gdf[gdf["heading_ok"]]
                if gdf.empty:
                    return
            clusters = _cluster_by_centroid(gdf, eps_m)
            entities = _build_entities_from_index_clusters(
                gdf,
                clusters,
                entity_type,
                drive_id,
                line_merge=line_merge,
                qa_flag="needs_review",
                evidence_sources={"image": True, "lidar": False, "aerial": False},
                created_from_run=str(run_dir),
            )
            filtered = []
            for ent in entities:
                frames_hit = int(ent["properties"].get("frames_hit", 0))
                if frames_hit < min_frames:
                    frames_hit = _frames_hit_by_proximity(ent["geometry"], data_root, drive_id, frame_rows, 10.0)
                    ent["properties"]["frames_hit"] = frames_hit
                if frames_hit >= min_frames:
                    filtered.append(ent)
            entity_layers[entity_type + ("_line" if line_merge else "_poly")].extend(filtered)

        _aggregate_from_frame(
            lane_frame,
            "lane_marking",
            True,
            float(cluster_cfg.get("lane_marking_eps_m", 5.0)),
            int(frames_cfg.get("lane_marking", 3)),
        )
        _aggregate_from_frame(
            stop_frame,
            "stop_line",
            True,
            float(cluster_cfg.get("stop_line_eps_m", 4.0)),
            int(frames_cfg.get("stop_line", 2)),
        )

        def _aggregate_crosswalk() -> None:
            if cross_frame.empty:
                return
            candidates = []
            for frame_id, hits in cross_frame.groupby("frame_id"):
                union_geom = unary_union(hits.geometry.values)
                if union_geom is None or union_geom.is_empty:
                    continue
                gore_overlap_ratio = 0.0
                gore_overlap = 0
                gore_geom = gore_by_frame.get(frame_id)
                if gore_geom is not None and not gore_geom.is_empty:
                    gore_overlap_ratio = union_geom.intersection(gore_geom).area / max(1e-6, union_geom.area)
                    cross_score = _score_from_gdf(hits)
                    gore_score = gore_score_by_frame.get(frame_id, 0.0)
                    if gore_overlap_ratio >= 0.30 and gore_score >= cross_score:
                        gore_overlap = 1
                if road_poly is not None and not road_poly.is_empty:
                    shrink_m = float(cross_final_cfg.get("road_shrink_m", 0.0))
                    road_ref = road_poly.buffer(-shrink_m) if shrink_m > 0 else road_poly
                    if road_ref.is_empty:
                        road_ref = road_poly
                    clipped = union_geom.intersection(road_ref)
                    if not clipped.is_empty:
                        union_geom = clipped
                rect_raw = union_geom.minimum_rotated_rectangle
                if rect_raw is None or rect_raw.is_empty:
                    continue
                metrics_raw = _rect_metrics(rect_raw, union_geom)
                buffer_m = float(cross_final_cfg.get("rect_buffer_m", 0.0))
                if buffer_m > 0:
                    union_geom = union_geom.buffer(buffer_m)
                rect = union_geom.minimum_rotated_rectangle
                if rect is None or rect.is_empty:
                    rect = rect_raw
                metrics = metrics_raw
                max_rect_l = float(cross_final_cfg.get("max_rect_l_m", 25.0))
                max_rect_w = float(cross_final_cfg.get("max_rect_w_m", 30.0))
                if metrics["rect_l_m"] > max_rect_l or metrics["rect_w_m"] > max_rect_w:
                    clip_r = max_rect_l / 2.0
                    clipped = union_geom.intersection(rect.centroid.buffer(clip_r))
                    if not clipped.is_empty:
                        union_geom = clipped
                        rect_raw = union_geom.minimum_rotated_rectangle
                        if rect_raw is None or rect_raw.is_empty:
                            continue
                        metrics_raw = _rect_metrics(rect_raw, union_geom)
                        rect = rect_raw
                        metrics = metrics_raw
                if metrics["rect_area_m2"] <= 0:
                    continue
                rect_heading = _rect_heading_deg(rect)
                road_heading = _road_poly_heading(
                    road_poly,
                    rect,
                    float(cross_final_cfg.get("road_heading_radius_m", 15.0)),
                )
                if road_heading is None:
                    road_heading = _trajectory_heading_near(data_root, drive_id, frame_rows, rect, 20.0)
                if road_heading is None:
                    road_heading = _nearest_yaw_heading(data_root, drive_id, frame_rows, rect)
                heading_diff = ""
                if rect_heading is not None and road_heading is not None:
                    diff = _angle_diff_deg(rect_heading, road_heading)
                    heading_diff = min(diff, abs(diff - 90.0))
                inside_ratio = 0.0
                if road_poly is not None and not road_poly.is_empty:
                    inside_ratio = rect.intersection(road_poly).area / max(1e-6, rect.area)
                reject = []
                if inside_ratio < float(cross_final_cfg.get("min_inside_ratio", 0.5)):
                    reject.append("off_road")
                if gore_overlap:
                    reject.append("gore_overlap")
                if heading_diff != "" and heading_diff > float(cross_final_cfg.get("max_heading_diff_deg", 25.0)):
                    reject.append("angle")
                if metrics["rectangularity"] < float(cross_final_cfg.get("min_rectangularity", 0.45)):
                    reject.append("rect")
                if metrics["rect_w_m"] < float(cross_final_cfg.get("min_rect_w_m", 1.5)) or metrics["rect_w_m"] > float(
                    cross_final_cfg.get("max_rect_w_m", 30.0)
                ):
                    reject.append("size")
                if metrics["rect_l_m"] < float(cross_final_cfg.get("min_rect_l_m", 3.0)) or metrics["rect_l_m"] > float(
                    cross_final_cfg.get("max_rect_l_m", 40.0)
                ):
                    reject.append("size")
                if metrics["aspect"] < float(cross_final_cfg.get("min_aspect", 1.3)) or metrics["aspect"] > float(
                    cross_final_cfg.get("max_aspect", 15.0)
                ):
                    reject.append("aspect")
                candidates.append(
                    {
                        "geometry": rect,
                        "properties": {
                            "candidate_id": _entity_id(f"{drive_id}_crosswalk_cand", len(candidates)),
                            "drive_id": drive_id,
                            "frame_id": frame_id,
                            "entity_type": "crosswalk",
                            "area_m2": metrics["area_m2"],
                            "rect_w_m": metrics["rect_w_m"],
                            "rect_l_m": metrics["rect_l_m"],
                            "aspect": metrics["aspect"],
                            "rectangularity": metrics["rectangularity"],
                            "heading_diff_to_perp_deg": heading_diff,
                            "inside_road_ratio": inside_ratio,
                            "gore_overlap": gore_overlap,
                            "gore_overlap_ratio": gore_overlap_ratio,
                            "reject_reasons": ",".join(reject),
                            "qa_flag": "ok" if not reject else reject[0],
                        },
                    }
                )
            if candidates:
                entity_layers["crosswalk_candidate_poly"].extend(candidates)

            candidate_map = {}
            for feat in entity_layers["crosswalk_candidate_poly"]:
                cand_id = feat["properties"].get("candidate_id")
                if cand_id:
                    candidate_map[cand_id] = feat

            candidate_gdf = _gdf_from_entities(entity_layers["crosswalk_candidate_poly"], "EPSG:32632")
            candidate_gdf = candidate_gdf[candidate_gdf["drive_id"] == drive_id]
            if candidate_gdf.empty:
                return
            candidate_gdf = candidate_gdf[~candidate_gdf["reject_reasons"].str.contains("off_road|giant", na=False)]
            if candidate_gdf.empty:
                return
            candidate_gdf_strict = candidate_gdf[candidate_gdf["reject_reasons"].fillna("") == ""]
            cross_eps = float(cluster_cfg.get("crosswalk_eps_m", 6.0))
            cross_buf = float(cross_cfg.get("cluster_buffer_m", 3.0))
            clusters = _cluster_by_centroid(candidate_gdf, max(cross_eps, cross_buf * 2.0))
            entities = []
            for idx, indices in enumerate(clusters):
                hits = candidate_gdf.iloc[indices]
                max_heading = float(cross_final_cfg.get("max_heading_diff_deg", 25.0))
                min_rect_w = float(cross_final_cfg.get("min_rect_w_m", 1.5))
                max_rect_w = float(cross_final_cfg.get("max_rect_w_m", 30.0))
                min_rect_l = float(cross_final_cfg.get("min_rect_l_m", 3.0))
                max_rect_l = float(cross_final_cfg.get("max_rect_l_m", 40.0))
                min_aspect = float(cross_final_cfg.get("min_aspect", 1.3))
                max_aspect = float(cross_final_cfg.get("max_aspect", 15.0))
                min_rectangularity = float(cross_final_cfg.get("min_rectangularity", 0.45))
                min_inside = float(cross_final_cfg.get("min_inside_ratio", 0.5))

                hits_geom = hits.copy()
                if "heading_diff_to_perp_deg" in hits_geom.columns:
                    hits_geom = hits_geom[
                        (hits_geom["heading_diff_to_perp_deg"].isna())
                        | (hits_geom["heading_diff_to_perp_deg"] <= max_heading)
                    ]
                if "rect_w_m" in hits_geom.columns:
                    hits_geom = hits_geom[(hits_geom["rect_w_m"] >= min_rect_w) & (hits_geom["rect_w_m"] <= max_rect_w)]
                if "rect_l_m" in hits_geom.columns:
                    hits_geom = hits_geom[(hits_geom["rect_l_m"] >= min_rect_l) & (hits_geom["rect_l_m"] <= max_rect_l)]
                if "aspect" in hits_geom.columns:
                    hits_geom = hits_geom[(hits_geom["aspect"] >= min_aspect) & (hits_geom["aspect"] <= max_aspect)]
                if "rectangularity" in hits_geom.columns:
                    hits_geom = hits_geom[hits_geom["rectangularity"] >= min_rectangularity]
                if "inside_road_ratio" in hits_geom.columns:
                    hits_geom = hits_geom[hits_geom["inside_road_ratio"] >= min_inside]
                if "gore_overlap" in hits_geom.columns:
                    hits_geom = hits_geom[hits_geom["gore_overlap"] != 1]

                strict_idx = candidate_gdf_strict.index.intersection(hits.index)
                hits_strict = candidate_gdf_strict.loc[strict_idx]
                geom_hits = hits_strict if not hits_strict.empty else hits_geom
                if geom_hits.empty:
                    geom_hits = hits
                rep_geom = None
                if not geom_hits.empty and "rect_w_m" in geom_hits.columns and "rect_l_m" in geom_hits.columns:
                    med_w = float(geom_hits["rect_w_m"].median())
                    med_l = float(geom_hits["rect_l_m"].median())
                    geom_hits = geom_hits.copy()
                    geom_hits["_size_dist"] = (geom_hits["rect_w_m"] - med_w).abs() + (geom_hits["rect_l_m"] - med_l).abs()
                    best = geom_hits.sort_values("_size_dist").iloc[0]
                    rep_geom = best.geometry
                union_geom = unary_union(geom_hits.geometry.values)
                if (union_geom is None or union_geom.is_empty) and rep_geom is not None:
                    union_geom = rep_geom
                if union_geom is None or union_geom.is_empty:
                    continue
                if road_poly is not None and not road_poly.is_empty:
                    shrink_m = float(cross_final_cfg.get("road_shrink_m", 0.0))
                    road_ref = road_poly.buffer(-shrink_m) if shrink_m > 0 else road_poly
                    if road_ref.is_empty:
                        road_ref = road_poly
                    clipped = union_geom.intersection(road_ref)
                    if not clipped.is_empty:
                        union_geom = clipped
                rect_raw = union_geom.minimum_rotated_rectangle
                if rect_raw is None or rect_raw.is_empty:
                    continue
                metrics_raw = _rect_metrics(rect_raw, union_geom)
                buffer_m = float(cross_final_cfg.get("rect_buffer_m", 0.0))
                if buffer_m > 0:
                    union_geom = union_geom.buffer(buffer_m)
                rect = union_geom.minimum_rotated_rectangle
                if rect is None or rect.is_empty:
                    rect = rect_raw
                metrics = metrics_raw
                max_rect_l = float(cross_final_cfg.get("max_rect_l_m", 25.0))
                max_rect_w = float(cross_final_cfg.get("max_rect_w_m", 8.0))
                if (metrics["rect_l_m"] > max_rect_l or metrics["rect_w_m"] > max_rect_w) and rep_geom is not None:
                    union_geom = rep_geom
                    rect_raw = union_geom.minimum_rotated_rectangle
                    if rect_raw is None or rect_raw.is_empty:
                        continue
                    metrics_raw = _rect_metrics(rect_raw, union_geom)
                    rect = rect_raw
                    metrics = metrics_raw
                if metrics["rect_l_m"] > max_rect_l or metrics["rect_w_m"] > max_rect_w:
                    clip_r = max_rect_l / 2.0
                    clipped = union_geom.intersection(rect.centroid.buffer(clip_r))
                    if not clipped.is_empty:
                        union_geom = clipped
                        rect_raw = union_geom.minimum_rotated_rectangle
                        if rect_raw is None or rect_raw.is_empty:
                            continue
                        metrics_raw = _rect_metrics(rect_raw, union_geom)
                        rect = rect_raw
                        metrics = metrics_raw
                if metrics["rect_area_m2"] <= 0:
                    continue
                rect_heading = _rect_heading_deg(rect)
                road_heading = _road_poly_heading(
                    road_poly,
                    rect,
                    float(cross_final_cfg.get("road_heading_radius_m", 15.0)),
                )
                if road_heading is None:
                    road_heading = _trajectory_heading_near(data_root, drive_id, frame_rows, rect, 20.0)
                if road_heading is None:
                    road_heading = _nearest_yaw_heading(data_root, drive_id, frame_rows, rect)
                heading_diff = ""
                if rect_heading is not None and road_heading is not None:
                    diff = _angle_diff_deg(rect_heading, road_heading)
                    heading_diff = min(diff, abs(diff - 90.0))
                inside_ratio = 0.0
                if road_poly is not None and not road_poly.is_empty:
                    inside_ratio = rect.intersection(road_poly).area / max(1e-6, rect.area)
                hits_support = hits.copy()
                if "gore_overlap" in hits_support.columns:
                    hits_support = hits_support[hits_support["gore_overlap"] != 1]
                frames = set(hits_support["frame_id"].tolist()) if not hits_support.empty else set(hits["frame_id"].tolist())
                frames_hit = len(frames)
                reject = []
                min_frames_hit = int(cross_final_cfg.get("min_frames_hit", frames_cfg.get("crosswalk", 2)))
                if metrics["area_m2"] > float(cross_final_cfg.get("giant_area_m2", 300.0)) or metrics["rect_l_m"] > float(
                    cross_final_cfg.get("giant_rect_l_m", 40.0)
                ):
                    reject.append("giant")
                if inside_ratio < float(cross_final_cfg.get("min_inside_ratio", 0.5)):
                    reject.append("off_road")
                if heading_diff != "" and heading_diff > float(cross_final_cfg.get("max_heading_diff_deg", 25.0)):
                    reject.append("angle")
                if metrics["rect_w_m"] < float(cross_final_cfg.get("min_rect_w_m", 1.5)) or metrics["rect_w_m"] > float(
                    cross_final_cfg.get("max_rect_w_m", 8.0)
                ):
                    reject.append("size")
                if metrics["rect_l_m"] < float(cross_final_cfg.get("min_rect_l_m", 3.0)) or metrics["rect_l_m"] > float(
                    cross_final_cfg.get("max_rect_l_m", 25.0)
                ):
                    reject.append("size")
                if metrics["aspect"] < float(cross_final_cfg.get("min_aspect", 1.3)) or metrics["aspect"] > float(
                    cross_final_cfg.get("max_aspect", 15.0)
                ):
                    reject.append("aspect")
                thin_long_l = float(cross_final_cfg.get("thin_long_rect_l_min", 0.0))
                thin_long_w = float(cross_final_cfg.get("thin_long_rect_w_max", 0.0))
                if thin_long_l > 0 and thin_long_w > 0:
                    if metrics["rect_l_m"] >= thin_long_l and metrics["rect_w_m"] <= thin_long_w:
                        reject.append("thin_long")
                if metrics["rectangularity"] < float(cross_final_cfg.get("min_rectangularity", 0.45)):
                    reject.append("rect")
                if frames_hit < min_frames_hit:
                    reject.append("low_hit")
                    for _, row in hits.iterrows():
                        cand_id = row.get("candidate_id")
                        if not cand_id:
                            continue
                        feat = candidate_map.get(cand_id)
                        if not feat:
                            continue
                        current = feat["properties"].get("reject_reasons", "")
                        feat["properties"]["reject_reasons"] = _append_reject_reason("low_hit", current)
                        feat["properties"]["qa_flag"] = "low_hit"

                support_geoms = hits_support if not hits_support.empty else hits
                jitter_p90 = 0.0
                angle_jitter_p90 = 0.0
                if not support_geoms.empty:
                    centroids = [geom.centroid for geom in support_geoms.geometry]
                    xs = [pt.x for pt in centroids]
                    ys = [pt.y for pt in centroids]
                    med_x = float(np.median(xs))
                    med_y = float(np.median(ys))
                    dists = [Point(med_x, med_y).distance(pt) for pt in centroids]
                    jitter_p90 = float(np.percentile(dists, 90)) if dists else 0.0
                    if "heading_diff_to_perp_deg" in support_geoms.columns:
                        diffs = [float(v) for v in support_geoms["heading_diff_to_perp_deg"].tolist() if v != ""]
                        angle_jitter_p90 = float(np.percentile(diffs, 90)) if diffs else 0.0
                    else:
                        headings = []
                        for geom in support_geoms.geometry:
                            heading = _rect_heading_deg(geom)
                            if heading is not None:
                                headings.append(heading)
                        if headings:
                            med_heading = float(np.median(headings))
                            diffs = [_angle_diff_deg(h, med_heading) for h in headings]
                            angle_jitter_p90 = float(np.percentile(diffs, 90)) if diffs else 0.0
                jitter_max = float(cross_final_cfg.get("jitter_p90_max", 8.0))
                angle_jitter_max = float(cross_final_cfg.get("angle_jitter_p90_max", 45.0))
                frame_span_max = int(cross_final_cfg.get("frame_span_max", 0))
                frame_span = 0
                if frames:
                    try:
                        frame_ids_num = [int(str(f)) for f in frames if str(f).isdigit()]
                        if frame_ids_num:
                            frame_span = max(frame_ids_num) - min(frame_ids_num)
                    except Exception:
                        frame_span = 0
                if jitter_p90 > jitter_max or angle_jitter_p90 > angle_jitter_max:
                    reject.append("unstable")
                if frame_span_max > 0 and frame_span > frame_span_max:
                    reject.append("span")

                score_total = min(1.0, frames_hit / max(1.0, float(min_frames_hit) * 2.0))
                if reject:
                    qa_flag = reject[0]
                else:
                    qa_flag = "ok"
                if not reject:
                    entities.append(
                        {
                            "geometry": rect,
                            "properties": {
                                "entity_id": _entity_id(f"{drive_id}_crosswalk", idx),
                                "drive_id": drive_id,
                                "entity_type": "crosswalk",
                                "confidence": score_total,
                                "evidence_sources": json.dumps({"image": True, "lidar": False, "aerial": False}, ensure_ascii=True),
                                "frames_hit": frames_hit,
                                "frame_ids": f"{min(frames)}..{max(frames)} ({len(frames)})" if frames else "",
                                "area_m2": metrics["area_m2"],
                                "rect_w_m": metrics["rect_w_m"],
                                "rect_l_m": metrics["rect_l_m"],
                                "aspect": metrics["aspect"],
                                "rectangularity": metrics["rectangularity"],
                                "heading_diff_to_perp_deg": heading_diff,
                                "inside_road_ratio": inside_ratio,
                                "jitter_p90": jitter_p90,
                                "angle_jitter_p90": angle_jitter_p90,
                                "frame_span": frame_span,
                                "support_frames": json.dumps(sorted(frames), ensure_ascii=True),
                                "score_total": score_total,
                                "reject_reasons": ",".join(reject),
                                "qa_flag": qa_flag,
                            },
                        }
                    )
            if entities:
                entity_layers["crosswalk_poly"].extend(entities)

        _aggregate_crosswalk()

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
                x, y, yaw = load_kitti360_pose(data_root, drive_id, frame_id)
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
            sanity_notes = []
            point = Point(x, y)
            crosswalk_candidate_ids = []
            crosswalk_final_ids = []
            for layer_name, feats in entity_layers.items():
                for feat in feats:
                    geom = feat["geometry"]
                    if geom is None or geom.is_empty:
                        continue
                    if geom.distance(point) <= args.qa_radius_m:
                        entity_ids.append(feat["properties"].get("entity_id", feat["properties"].get("candidate_id", "")))
                        if layer_name == "crosswalk_candidate_poly":
                            crosswalk_candidate_ids.append(feat["properties"].get("candidate_id", feat["properties"].get("entity_id", "")))
                        if layer_name == "crosswalk_poly":
                            crosswalk_final_ids.append(feat["properties"]["entity_id"])
                        rect = feat["properties"].get("rectangularity")
                        hd = feat["properties"].get("heading_diff_deg")
                        ent_id = feat["properties"].get("entity_id", feat["properties"].get("candidate_id", ""))
                        if rect:
                            sanity_notes.append(f"{ent_id}:rect={rect}")
                        if hd != "" and hd is not None:
                            sanity_notes.append(f"{ent_id}:hd={hd}")

            support_frames = set()
            reject_summary_by_frame: Dict[str, set] = {}
            for feat in entity_layers.get("crosswalk_poly", []):
                props = feat.get("properties", {})
                if props.get("drive_id") != drive_id:
                    continue
                frames_raw = props.get("support_frames", "")
                frames_list = []
                if isinstance(frames_raw, str) and frames_raw:
                    try:
                        frames_list = json.loads(frames_raw)
                    except Exception:
                        frames_list = []
                for frame in frames_list:
                    support_frames.add(str(frame))
            for feat in entity_layers.get("crosswalk_candidate_poly", []):
                props = feat.get("properties", {})
                if props.get("drive_id") != drive_id:
                    continue
                cand_frame_id = str(props.get("frame_id", ""))
                if not cand_frame_id:
                    continue
                reasons = str(props.get("reject_reasons", "")).split(",")
                reasons = [r for r in reasons if r]
                if not reasons:
                    continue
                reject_summary_by_frame.setdefault(cand_frame_id, set()).update(reasons)

            qa_images_dir = outputs_dir / "qa_images" / drive_id
            overlay_raw_path = ""
            overlay_gated_path = ""
            overlay_entities_path = ""
            if args.emit_qa_images:
                qa_images_dir.mkdir(parents=True, exist_ok=True)
                if overlay_path and Path(overlay_path).exists():
                    overlay_raw_path = str(qa_images_dir / f"{frame_id}_overlay_raw.png")
                    shutil.copy2(overlay_path, overlay_raw_path)
                base_path = ""
                if image_path and Path(image_path).exists():
                    base_path = image_path
                elif overlay_raw_path and Path(overlay_raw_path).exists():
                    base_path = overlay_raw_path
                elif overlay_path and Path(overlay_path).exists():
                    base_path = overlay_path
                if base_path:
                    try:
                        base_img = Image.open(base_path).convert("RGBA")
                        calib = load_kitti360_calib(data_root, cam_id)
                        pose = (x, y, yaw)
                        gated_overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
                        gated_draw = ImageDraw.Draw(gated_overlay, "RGBA")
                        for layer_name, frames_list in frame_layers_drive.items():
                            for gdf in frames_list:
                                matches = gdf[gdf["frame_id"] == frame_id]
                                for _, row in matches.iterrows():
                                    geom = row.geometry
                                    pts = _geom_to_image_points(geom, pose, calib)
                                    if len(pts) < 2:
                                        continue
                                    if layer_name == "crosswalk_frame":
                                        gated_draw.polygon(pts, outline=(0, 128, 255, 200))
                                    elif layer_name == "stop_line_frame":
                                        gated_draw.line(pts, fill=(255, 0, 0, 200), width=3)
                                    elif layer_name == "lane_marking_frame":
                                        heading_ok = bool(row.get("heading_ok", True))
                                        color = (255, 255, 0, 200) if heading_ok else (160, 160, 160, 200)
                                        gated_draw.line(pts, fill=color, width=2)
                        for feat in entity_layers.get("crosswalk_candidate_poly", []):
                            props = feat["properties"]
                            if props.get("drive_id") != drive_id or props.get("frame_id") != frame_id:
                                continue
                            geom = feat["geometry"]
                            pts = _geom_to_image_points(geom, pose, calib)
                            if len(pts) < 2:
                                continue
                            reject = props.get("reject_reasons", "")
                            color = (0, 255, 255, 200) if not reject else (160, 160, 160, 200)
                            gated_draw.polygon(pts, outline=color)
                            if "low_hit" in reject:
                                c = geom.centroid
                                pt = _geom_to_image_points(c, pose, calib)
                                if pt:
                                    gated_draw.text(pt[0], "low_hit", fill=(255, 64, 64, 220))
                            if "gore_overlap" in reject:
                                c = geom.centroid
                                pt = _geom_to_image_points(c, pose, calib)
                                if pt:
                                    gated_draw.text(pt[0], "gore", fill=(255, 140, 0, 220))
                        gated_out = Image.alpha_composite(base_img, gated_overlay)
                        overlay_gated_path = str(qa_images_dir / f"{frame_id}_overlay_gated.png")
                        gated_out.save(overlay_gated_path)

                        overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
                        draw = ImageDraw.Draw(overlay, "RGBA")
                        for layer_name, feats in entity_layers.items():
                            for feat in feats:
                                if feat["properties"].get("drive_id") != drive_id:
                                    continue
                                geom = feat["geometry"]
                                if geom is None or geom.is_empty:
                                    continue
                                pts = _geom_to_image_points(geom, pose, calib)
                                if len(pts) < 2:
                                    continue
                                if layer_name == "road_surface_poly":
                                    draw.polygon(pts, outline=(0, 255, 0, 180))
                                elif layer_name == "crosswalk_poly":
                                    draw.polygon(pts, fill=(0, 128, 255, 80), outline=(0, 128, 255, 200))
                                elif layer_name == "stop_line_line":
                                    draw.line(pts, fill=(255, 0, 0, 200), width=3)
                                elif layer_name == "lane_marking_line":
                                    draw.line(pts, fill=(255, 255, 0, 200), width=2)
                        out = Image.alpha_composite(base_img, overlay)
                        overlay_entities_path = str(qa_images_dir / f"{frame_id}_overlay_entities.png")
                        out.save(overlay_entities_path)
                    except Exception as exc:
                        log.warning("overlay qa failed: %s %s (%s)", drive_id, frame_id, exc)

            qa_rows.append(
                {
                    "drive_id": drive_id,
                    "frame_id": frame_id,
                    "timestamp": row.get("timestamp", ""),
                    "lon": lon,
                    "lat": lat,
                    "image_path": image_path,
                    "image_overlay_path": overlay_path,
                    "overlay_raw_path": overlay_raw_path,
                    "overlay_gated_path": overlay_gated_path,
                    "overlay_entities_path": overlay_entities_path,
                    "lidar_raster_path": str(lidar_raster) if lidar_raster.exists() else "",
                    "entity_ids": json.dumps(sorted(set(entity_ids)), ensure_ascii=True),
                    "crosswalk_candidate_ids_nearby": json.dumps(sorted(set(crosswalk_candidate_ids)), ensure_ascii=True),
                    "crosswalk_final_ids_nearby": json.dumps(sorted(set(crosswalk_final_ids)), ensure_ascii=True),
                    "entity_support_frames": "yes" if frame_id in support_frames else "no",
                    "reject_reason_summary": ",".join(sorted(reject_summary_by_frame.get(frame_id, []))),
                    "sanity_note": ";".join(sanity_notes),
                }
            )
    entity_layers_gdf = {
        "road_surface_poly": _gdf_from_entities(entity_layers["road_surface_poly"], "EPSG:32632"),
        "lane_marking_line": _gdf_from_entities(entity_layers["lane_marking_line"], "EPSG:32632"),
        "crosswalk_candidate_poly": _gdf_from_entities(entity_layers["crosswalk_candidate_poly"], "EPSG:32632"),
        "crosswalk_poly": _gdf_from_entities(entity_layers["crosswalk_poly"], "EPSG:32632"),
        "stop_line_line": _gdf_from_entities(entity_layers["stop_line_line"], "EPSG:32632"),
    }
    road_surface_gdf = entity_layers_gdf["road_surface_poly"]
    if not road_surface_gdf.empty and "drive_id" in road_surface_gdf.columns:
        missing_drive = road_surface_gdf["drive_id"].isna() | (road_surface_gdf["drive_id"] == "")
        if missing_drive.any():
            log.error("road_surface_poly missing drive_id")
            return 7
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

    frame_layers_out = {}
    for name, frames_list in frame_layers_all.items():
        if frames_list:
            frame_layers_out[name] = gpd.GeoDataFrame(pd.concat(frames_list, ignore_index=True), geometry="geometry", crs="EPSG:32632")
        else:
            frame_layers_out[name] = gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32632")
    _write_gpkg_layers(outputs_dir / "frame_evidence_utm32.gpkg", frame_layers_out, "EPSG:32632")

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
    report_lines.append("## Per-Drive Count Summary")
    for name, gdf in entity_layers_gdf.items():
        stats = _count_stats_per_drive(gdf)
        report_lines.append(f"- {name}: min={stats['min']} median={stats['median']} max={stats['max']}")
    report_lines.append("")
    report_lines.append("## QA Assets")
    report_lines.append(f"- qa_index: {qa_path}")
    report_lines.append(f"- image_evidence: {outputs_dir / 'image_evidence_utm32.gpkg'}")
    report_lines.append(f"- frame_evidence: {outputs_dir / 'frame_evidence_utm32.gpkg'}")
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
    report_lines.append("## Heading Diff Stats (After)")
    lane_angle = _angle_stats_by_drive(entity_layers_gdf["lane_marking_line"], heading_segments_by_drive, "parallel")
    stop_angle = _angle_stats_by_drive(entity_layers_gdf["stop_line_line"], heading_segments_by_drive, "perpendicular")
    report_lines.append("### lane_marking")
    for drive_id, row in lane_angle.items():
        report_lines.append(f"- {drive_id}: p50={row.get('p50')} p90={row.get('p90')}")
    report_lines.append("### stop_line")
    for drive_id, row in stop_angle.items():
        report_lines.append(f"- {drive_id}: p50={row.get('p50')} p90={row.get('p90')}")
    report_lines.append("")
    report_lines.append("## Frames Hit Stats (After)")
    for layer_name in ["lane_marking_line", "stop_line_line", "crosswalk_poly"]:
        stats = _frames_hit_stats(entity_layers_gdf[layer_name])
        report_lines.append(f"- {layer_name}: p50={stats['p50']} p90={stats['p90']}")
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

    if not entity_layers_gdf["crosswalk_candidate_poly"].empty:
        report_lines.append("## Crosswalk Candidate Summary")
        report_lines.append(f"- candidate_count: {len(entity_layers_gdf['crosswalk_candidate_poly'])}")
        reasons = entity_layers_gdf["crosswalk_candidate_poly"]["reject_reasons"].fillna("")
        reason_counts = {}
        for reason in reasons.tolist():
            for token in [r for r in reason.split(",") if r]:
                reason_counts[token] = reason_counts.get(token, 0) + 1
        for reason, count in sorted(reason_counts.items()):
            report_lines.append(f"- reject_{reason}: {count}")
        report_lines.append("")

    if not entity_layers_gdf["crosswalk_poly"].empty:
        report_lines.append("## Crosswalk Final Summary")
        report_lines.append(f"- final_count: {len(entity_layers_gdf['crosswalk_poly'])}")
        top_final = entity_layers_gdf["crosswalk_poly"].head(5)
        for _, row in top_final.iterrows():
            report_lines.append(
                f"- {row.get('entity_id')}: frames_hit={row.get('frames_hit')} area_m2={row.get('area_m2')} rect={row.get('rect_w_m')}x{row.get('rect_l_m')} heading_diff={row.get('heading_diff_to_perp_deg')} jitter_p90={row.get('jitter_p90')} angle_jitter_p90={row.get('angle_jitter_p90')}"
            )
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

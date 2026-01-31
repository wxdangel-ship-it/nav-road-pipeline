from __future__ import annotations

import datetime as dt
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _geokey_vlr_epsg(epsg: int) -> bytes:
    # Minimal GeoKeyDirectoryTag with EPSG projected CS.
    keys = [
        (1024, 0, 1, 1),  # GTModelTypeGeoKey: Projected
        (1025, 0, 1, 1),  # GTRasterTypeGeoKey: PixelIsArea
        (3072, 0, 1, int(epsg)),  # ProjectedCSTypeGeoKey
    ]
    header = [1, 1, 0, len(keys)]
    payload = header[:]
    for k in keys:
        payload.extend(list(k))
    arr = np.array(payload, dtype=np.uint16)
    return arr.tobytes()


def _vlr_record(user_id: str, record_id: int, description: str, data: bytes) -> bytes:
    user_id_b = user_id.encode("ascii", errors="ignore")[:16].ljust(16, b"\x00")
    desc_b = description.encode("ascii", errors="ignore")[:32].ljust(32, b"\x00")
    header = struct.pack("<H16sHH32s", 0, user_id_b, int(record_id), int(len(data)), desc_b)
    return header + data


def _las_header_bytes(
    point_count: int,
    point_format: int,
    point_record_length: int,
    scales: Tuple[float, float, float],
    offsets: Tuple[float, float, float],
    bounds: Tuple[float, float, float, float, float, float],
    vlr_bytes: bytes,
) -> bytes:
    today = dt.date.today()
    day_of_year = int(today.strftime("%j"))
    year = today.year
    header_size = 227
    offset_to_points = header_size + len(vlr_bytes)
    num_vlrs = 1 if vlr_bytes else 0

    sig = b"LASF"
    sys_id = b"Codex".ljust(32, b"\x00")
    gen_soft = b"nav-road-pipeline".ljust(32, b"\x00")
    project_id = b"\x00" * 16
    num_by_return = (0, 0, 0, 0, 0)

    minx, miny, minz, maxx, maxy, maxz = bounds
    sx, sy, sz = scales
    ox, oy, oz = offsets

    header = struct.pack(
        "<4sHH16sBB32s32sHHH I I B H I 5I ddd ddd dddddd",
        sig,
        0,
        0,
        project_id,
        1,
        2,
        sys_id,
        gen_soft,
        day_of_year,
        year,
        header_size,
        offset_to_points,
        num_vlrs,
        point_format,
        point_record_length,
        point_count,
        *num_by_return,
        sx,
        sy,
        sz,
        ox,
        oy,
        oz,
        maxx,
        minx,
        maxy,
        miny,
        maxz,
        minz,
    )
    return header


def _scale_offset(points_xyz: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if points_xyz.size == 0:
        return (0.001, 0.001, 0.001), (0.0, 0.0, 0.0)
    mins = points_xyz.min(axis=0)
    return (0.001, 0.001, 0.001), (float(mins[0]), float(mins[1]), float(mins[2]))


def _bounds(points_xyz: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    if points_xyz.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    return (float(mins[0]), float(mins[1]), float(mins[2]), float(maxs[0]), float(maxs[1]), float(maxs[2]))


def _points_to_int(points_xyz: np.ndarray, scales: Tuple[float, float, float], offsets: Tuple[float, float, float]) -> np.ndarray:
    sx, sy, sz = scales
    ox, oy, oz = offsets
    xi = np.round((points_xyz[:, 0] - ox) / sx).astype(np.int32)
    yi = np.round((points_xyz[:, 1] - oy) / sy).astype(np.int32)
    zi = np.round((points_xyz[:, 2] - oz) / sz).astype(np.int32)
    return np.stack([xi, yi, zi], axis=1)


def _map_intensity(intensity: np.ndarray) -> np.ndarray:
    if intensity.size == 0:
        return intensity.astype(np.uint16)
    if intensity.dtype.kind in {"f", "c"}:
        max_val = float(np.nanmax(intensity))
        if max_val <= 1.5:
            scaled = np.clip(intensity, 0.0, 1.0) * 65535.0
            return np.round(scaled).astype(np.uint16)
        if max_val <= 255.0:
            scaled = np.clip(intensity, 0.0, 255.0) * 256.0
            return np.round(scaled).astype(np.uint16)
    return np.clip(intensity, 0, 65535).astype(np.uint16)


def write_las(
    path: Path,
    points_xyz: np.ndarray,
    intensity: np.ndarray,
    classification: np.ndarray,
    epsg: int,
) -> List[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    scales, offsets = _scale_offset(points_xyz)
    bounds = _bounds(points_xyz)
    geo_vlr = _vlr_record("LASF_Projection", 34735, "GeoKeyDirectory", _geokey_vlr_epsg(int(epsg)))
    header = _las_header_bytes(
        point_count=int(points_xyz.shape[0]),
        point_format=0,
        point_record_length=20,
        scales=scales,
        offsets=offsets,
        bounds=bounds,
        vlr_bytes=geo_vlr,
    )
    ints_xyz = _points_to_int(points_xyz, scales, offsets)
    inten = _map_intensity(intensity)
    cls = np.clip(classification, 0, 255).astype(np.uint8)

    with path.open("wb") as f:
        f.write(header)
        f.write(geo_vlr)
        bitfield = 1
        scan_angle = 0
        user_data = 0
        point_src = 0
        for (x, y, z), it, cl in zip(ints_xyz, inten, cls):
            rec = struct.pack("<iiiHBBbBH", int(x), int(y), int(z), int(it), bitfield, int(cl), scan_angle, user_data, point_src)
            f.write(rec)

    outputs = [path]
    # Provide a .las twin if the requested suffix is .laz for better QGIS compatibility.
    if path.suffix.lower() == ".laz":
        las_path = path.with_suffix(".las")
        if las_path != path:
            las_path.write_bytes(path.read_bytes())
            outputs.append(las_path)
    return outputs


def classify_codes(road_mask: np.ndarray, road_points_mask: np.ndarray) -> np.ndarray:
    # LAS classification: 11=road surface, 1=unclassified/non-road.
    cls = np.ones((road_points_mask.shape[0],), dtype=np.uint8)
    cls[road_points_mask] = 11
    return cls


__all__ = ["write_las", "classify_codes"]

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from pipeline.calib.io_kitti360_calib import Kitti360Calib, Cam0PoseProvider, load_cam0_pose_provider, load_kitti360_calib_bundle
from pipeline.calib.kitti360_world import kitti_world_to_utm32
from pipeline.calib.kitti360_projection import project_world_to_image

try:
    import rasterio
except Exception:  # pragma: no cover - optional at runtime
    rasterio = None


@dataclass
class GroundModel:
    mode: str
    dtm_path: Optional[Path]
    z0: float


@dataclass
class BackprojectContext:
    data_root: Path
    drive_id: str
    cam_id: str
    calib: Kitti360Calib
    pose_provider: Cam0PoseProvider
    dtm_path: Optional[Path]
    dtm: Optional[object]
    dtm_nodata: Optional[float]


_CTX_CACHE: Dict[Tuple[str, str, str, str], BackprojectContext] = {}
_DEFAULT_CTX: Optional[BackprojectContext] = None


def _load_dtm(path: Optional[Path]) -> Tuple[Optional[object], Optional[float]]:
    if path is None or not path.exists() or rasterio is None:
        return None, None
    try:
        ds = rasterio.open(path)
    except Exception:
        return None, None
    nodata = ds.nodata
    return ds, nodata


def get_context(
    data_root: Path,
    drive_id: str,
    cam_id: str = "image_00",
    dtm_path: Optional[Path] = None,
    frame_id_for_size: Optional[str] = None,
) -> BackprojectContext:
    key = (str(data_root), str(drive_id), str(cam_id), str(dtm_path) if dtm_path else "")
    if key in _CTX_CACHE:
        return _CTX_CACHE[key]
    calib = load_kitti360_calib_bundle(data_root, drive_id, cam_id=cam_id, frame_id_for_size=frame_id_for_size)
    pose_provider = load_cam0_pose_provider(data_root, drive_id)
    dtm, nodata = _load_dtm(dtm_path)
    ctx = BackprojectContext(
        data_root=data_root,
        drive_id=drive_id,
        cam_id=cam_id,
        calib=calib,
        pose_provider=pose_provider,
        dtm_path=dtm_path,
        dtm=dtm,
        dtm_nodata=nodata,
    )
    _CTX_CACHE[key] = ctx
    return ctx


def configure_default_context(
    data_root: Path,
    drive_id: str,
    cam_id: str = "image_00",
    dtm_path: Optional[Path] = None,
    frame_id_for_size: Optional[str] = None,
) -> BackprojectContext:
    global _DEFAULT_CTX
    _DEFAULT_CTX = get_context(data_root, drive_id, cam_id=cam_id, dtm_path=dtm_path, frame_id_for_size=frame_id_for_size)
    return _DEFAULT_CTX


def _require_ctx(ctx: Optional[BackprojectContext]) -> BackprojectContext:
    if ctx is not None:
        return ctx
    if _DEFAULT_CTX is None:
        raise RuntimeError("backproject_context_not_configured")
    return _DEFAULT_CTX


def world_to_pixel_cam0(
    frame_id: str,
    xyz_world: np.ndarray,
    ctx: Optional[BackprojectContext] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctx = _require_ctx(ctx)
    pts = np.asarray(xyz_world, dtype=float)
    if pts.size == 0:
        empty = np.zeros((0,), dtype=float)
        return empty, empty, empty
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    proj = project_world_to_image(
        pts[:, :3],
        frame_id=str(frame_id),
        calib=ctx.calib,
        pose_provider=ctx.pose_provider,
        use_rect=True,
        y_flip_mode="fixed_true",
        sanity=False,
    )
    u = proj["u"].astype(float)
    v = proj["v"].astype(float)
    valid = proj["valid"].astype(bool)
    return u, v, valid


def _k_rect_from_p(p_rect: Optional[np.ndarray], k: np.ndarray) -> np.ndarray:
    if p_rect is None:
        return k
    return p_rect[:3, :3].copy()


def pixel_to_ray_c0rect(u: float, v: float, calib: Kitti360Calib) -> np.ndarray:
    k_rect = _k_rect_from_p(calib.p_rect_00, calib.k)
    fx, fy = float(k_rect[0, 0]), float(k_rect[1, 1])
    cx, cy = float(k_rect[0, 2]), float(k_rect[1, 2])
    if fx == 0 or fy == 0:
        raise ValueError("invalid_intrinsics")
    x = (u - cx) / fx
    y = (v - cy) / fy
    return np.array([x, y, 1.0], dtype=float)


def ray_c0rect_to_ray_c0(ray_c0rect: np.ndarray, calib: Kitti360Calib) -> np.ndarray:
    r_rect = calib.r_rect_00
    if r_rect is not None:
        ray_c0 = r_rect.T @ ray_c0rect
    else:
        ray_c0 = ray_c0rect
    norm = float(np.linalg.norm(ray_c0))
    if norm > 0:
        ray_c0 = ray_c0 / norm
    return ray_c0


def _sample_dtm_z(ctx: BackprojectContext, frame_id: str, x: float, y: float, z: float) -> Optional[float]:
    if ctx.dtm is None or rasterio is None:
        return None
    pts_wk = np.array([[float(x), float(y), float(z)]], dtype=np.float64)
    pts_wu = kitti_world_to_utm32(pts_wk, ctx.data_root, ctx.drive_id, frame_id)
    x_utm = float(pts_wu[0, 0])
    y_utm = float(pts_wu[0, 1])
    try:
        val = next(ctx.dtm.sample([(x_utm, y_utm)]))
    except Exception:
        return None
    if val is None or len(val) == 0:
        return None
    z = float(val[0])
    if ctx.dtm_nodata is not None and np.isfinite(ctx.dtm_nodata):
        if abs(z - float(ctx.dtm_nodata)) < 1e-6:
            return None
    if not np.isfinite(z):
        return None
    return z


def _intersect_plane(origin: np.ndarray, direction: np.ndarray, z0: float) -> Optional[np.ndarray]:
    if abs(direction[2]) < 1e-9:
        return None
    t = (float(z0) - float(origin[2])) / float(direction[2])
    if t <= 0:
        return None
    return origin + t * direction


def pixel_to_world_on_ground(
    frame_id: str,
    u: float,
    v: float,
    ground_model: GroundModel | Dict[str, object],
    ctx: Optional[BackprojectContext] = None,
) -> Optional[np.ndarray]:
    ctx = _require_ctx(ctx)
    if isinstance(ground_model, dict):
        mode = str(ground_model.get("mode", "fixed_plane"))
        z0 = float(ground_model.get("z0", 0.0))
        dtm_path = ground_model.get("dtm_path")
        dtm_path = Path(dtm_path) if dtm_path else None
    else:
        mode = str(ground_model.mode)
        z0 = float(ground_model.z0)
        dtm_path = ground_model.dtm_path

    ray_rect = pixel_to_ray_c0rect(float(u), float(v), ctx.calib)
    ray_c0 = ray_c0rect_to_ray_c0(ray_rect, ctx.calib)
    t_w_c0 = ctx.pose_provider.get_t_w_c0(str(frame_id))
    origin = t_w_c0[:3, 3]
    direction = t_w_c0[:3, :3] @ ray_c0
    direction = direction / max(1e-12, float(np.linalg.norm(direction)))

    if mode == "lidar_clean_dtm" and dtm_path is not None and ctx.dtm is not None:
        pt = _intersect_plane(origin, direction, z0)
        if pt is None:
            return None
        t = (pt[2] - origin[2]) / direction[2] if abs(direction[2]) > 1e-9 else None
        if t is None:
            return None
        for _ in range(3):
            candidate = origin + t * direction
            dtm_z = _sample_dtm_z(ctx, frame_id, candidate[0], candidate[1], candidate[2])
            if dtm_z is None:
                return pt
            if abs(direction[2]) < 1e-9:
                return None
            t = (dtm_z - origin[2]) / direction[2]
            if t <= 0:
                return None
        return origin + t * direction

    return _intersect_plane(origin, direction, z0)


__all__ = [
    "GroundModel",
    "BackprojectContext",
    "get_context",
    "configure_default_context",
    "world_to_pixel_cam0",
    "pixel_to_ray_c0rect",
    "ray_c0rect_to_ray_c0",
    "pixel_to_world_on_ground",
]

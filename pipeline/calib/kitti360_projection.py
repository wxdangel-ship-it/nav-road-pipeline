from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from pipeline.calib.io_kitti360_calib import Cam0PoseProvider, Kitti360Calib
from pipeline.calib.proj_sanity import (
    validate_depth,
    validate_in_image_ratio,
    validate_uv_spread,
)


def _project_with_p_rect(
    points_cam: np.ndarray,
    p_rect: np.ndarray,
    r_rect: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if r_rect is not None:
        cam_rect = (r_rect @ points_cam.T).T
    else:
        cam_rect = points_cam
    ones = np.ones((cam_rect.shape[0], 1), dtype=cam_rect.dtype)
    cam_h = np.hstack([cam_rect, ones])
    img = (p_rect @ cam_h.T).T
    z = img[:, 2]
    valid = z > 1e-6
    u = np.zeros_like(z, dtype=float)
    v = np.zeros_like(z, dtype=float)
    u[valid] = img[valid, 0] / z[valid]
    v[valid] = img[valid, 1] / z[valid]
    return u, v, cam_rect[:, 2]


def _project_with_k(
    points_cam: np.ndarray,
    k: np.ndarray,
    r_rect: Optional[np.ndarray],
    y_flip: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if r_rect is not None:
        cam_rect = (r_rect @ points_cam.T).T
    else:
        cam_rect = points_cam
    x = cam_rect[:, 0]
    y = cam_rect[:, 1]
    z = cam_rect[:, 2]
    if y_flip:
        y = -y
    valid = z > 1e-6
    u = np.zeros_like(z, dtype=float)
    v = np.zeros_like(z, dtype=float)
    u[valid] = (k[0, 0] * x[valid] / z[valid]) + k[0, 2]
    v[valid] = (k[1, 1] * y[valid] / z[valid]) + k[1, 2]
    return u, v, z


def project_cam0_to_image(
    points_cam0: np.ndarray,
    calib: Kitti360Calib,
    use_rect: bool = True,
    y_flip: bool = True,
    sanity: bool = False,
) -> Dict[str, np.ndarray]:
    p_rect = calib.p_rect_00 if use_rect else None
    r_rect = calib.r_rect_00 if use_rect else None
    if p_rect is not None:
        u, v, z_cam = _project_with_p_rect(points_cam0, p_rect, r_rect)
    else:
        u, v, z_cam = _project_with_k(points_cam0, calib.k, r_rect, y_flip=y_flip)
    h = int(calib.image_size[1])
    w = int(calib.image_size[0])
    valid = z_cam > 1e-6
    in_img = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if sanity:
        stats = {}
        stats.update(validate_depth(z_cam))
        stats.update(validate_uv_spread(u, v, in_img))
        stats.update(validate_in_image_ratio(in_img))
    return {"u": u, "v": v, "z_cam": z_cam, "valid": valid, "in_img": in_img}


def project_velo_to_image(
    points_velo: np.ndarray,
    calib: Kitti360Calib,
    use_rect: bool = True,
    y_flip_mode: str = "fixed_true",
    sanity: bool = False,
) -> Dict[str, np.ndarray]:
    if points_velo.size == 0:
        return {"u": np.zeros((0,)), "v": np.zeros((0,)), "z_cam": np.zeros((0,)), "valid": np.zeros((0,)), "in_img": np.zeros((0,))}
    pts_h = np.hstack([points_velo[:, :3], np.ones((points_velo.shape[0], 1), dtype=points_velo.dtype)])
    cam = (calib.t_c0_v @ pts_h.T).T
    y_flip = True
    if y_flip_mode == "fixed_false":
        y_flip = False
    elif y_flip_mode == "auto":
        y_flip = True
    return project_cam0_to_image(cam[:, :3], calib, use_rect=use_rect, y_flip=y_flip, sanity=sanity)


def project_world_to_image(
    points_world: np.ndarray,
    frame_id: str,
    calib: Kitti360Calib,
    pose_provider: Cam0PoseProvider,
    use_rect: bool = True,
    y_flip_mode: str = "fixed_true",
    sanity: bool = False,
) -> Dict[str, np.ndarray]:
    if points_world.size == 0:
        return {"u": np.zeros((0,)), "v": np.zeros((0,)), "z_cam": np.zeros((0,)), "valid": np.zeros((0,)), "in_img": np.zeros((0,))}
    t_w_c0 = pose_provider.get_t_w_c0(frame_id)
    t_c0_w = np.linalg.inv(t_w_c0)
    pts_h = np.hstack([points_world[:, :3], np.ones((points_world.shape[0], 1), dtype=points_world.dtype)])
    cam = (t_c0_w @ pts_h.T).T
    y_flip = True
    if y_flip_mode == "fixed_false":
        y_flip = False
    elif y_flip_mode == "auto":
        y_flip = True
    return project_cam0_to_image(cam[:, :3], calib, use_rect=use_rect, y_flip=y_flip, sanity=sanity)


def project_world_to_image_pose(
    points_world: np.ndarray,
    pose: Tuple[float, ...],
    calib: Kitti360Calib,
    use_rect: bool = True,
    y_flip_mode: str = "fixed_true",
    sanity: bool = False,
) -> Dict[str, np.ndarray]:
    if points_world.size == 0:
        return {"u": np.zeros((0,)), "v": np.zeros((0,)), "z_cam": np.zeros((0,)), "valid": np.zeros((0,)), "in_img": np.zeros((0,))}
    x0, y0, z0, roll, pitch, yaw = pose
    c1 = float(np.cos(yaw))
    s1 = float(np.sin(yaw))
    c2 = float(np.cos(pitch))
    s2 = float(np.sin(pitch))
    c3 = float(np.cos(roll))
    s3 = float(np.sin(roll))
    r_z = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    r_y = np.array([[c2, 0.0, s2], [0.0, 1.0, 0.0], [-s2, 0.0, c2]], dtype=float)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, c3, -s3], [0.0, s3, c3]], dtype=float)
    r_world_pose = r_z @ r_y @ r_x
    delta = points_world - np.array([x0, y0, z0], dtype=float)
    pose_xyz = (r_world_pose.T @ delta.T).T
    pts_h = np.hstack([pose_xyz, np.ones((pose_xyz.shape[0], 1), dtype=pose_xyz.dtype)])
    cam = (calib.t_c0_v @ pts_h.T).T
    y_flip = True
    if y_flip_mode == "fixed_false":
        y_flip = False
    return project_cam0_to_image(cam[:, :3], calib, use_rect=use_rect, y_flip=y_flip, sanity=sanity)

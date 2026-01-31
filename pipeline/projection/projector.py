from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Polygon

from pipeline.calib.io_kitti360_calib import Kitti360Calib
from pipeline.calib.kitti360_projection import (
    project_cam0_to_image,
    project_world_to_image_pose,
)


@dataclass
class RoundtripMetrics:
    reproj_iou_mask: Optional[float]
    reproj_iou_dilated: Optional[float]
    reproj_iou_bbox: Optional[float]
    reproj_center_err_px: Optional[float]
    reproj_area_ratio: Optional[float]


def world_geom_to_image(
    points: np.ndarray,
    pose: Tuple[float, ...],
    calib: Dict[str, np.ndarray],
    mode: str = "k_rrect",
) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=float)
    if len(pose) == 6:
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
        delta = points - np.array([x0, y0, z0], dtype=float)
        pose_xyz = (r_world_pose.T @ delta.T).T
        x_ego = pose_xyz[:, 0]
        y_ego = pose_xyz[:, 1]
        z_ego = pose_xyz[:, 2]
    else:
        x0, y0, yaw = pose
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

    if mode.startswith("p_rect"):
        proj = calib["p_rect"] @ np.vstack([cam[:3, :], np.ones((1, cam.shape[1]))])
        zs = proj[2, :]
        valid = zs > 1e-3
        us = np.zeros_like(zs)
        vs = np.zeros_like(zs)
        us[valid] = proj[0, valid] / zs[valid]
        vs[valid] = proj[1, valid] / zs[valid]
        return np.stack([us, vs, valid], axis=1)

    xyz = cam[:3, :].T
    if mode.endswith("rrect"):
        xyz = (calib["r_rect"] @ xyz.T).T
    zs = xyz[:, 2]
    valid = zs > 1e-3
    us = np.zeros_like(zs)
    vs = np.zeros_like(zs)
    k = calib["k"]
    us[valid] = (k[0, 0] * xyz[valid, 0] / zs[valid]) + k[0, 2]
    vs[valid] = (k[1, 1] * xyz[valid, 1] / zs[valid]) + k[1, 2]
    return np.stack([us, vs, valid], axis=1)


def project_points_cam0_to_image(
    points_cam0: np.ndarray,
    calib: Dict[str, np.ndarray],
    image_shape: Tuple[int, int],
    use_rect: bool = True,
    y_flip: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = int(image_shape[0]), int(image_shape[1])
    k = calib["k"]
    p_rect = calib.get("p_rect")
    r_rect = calib.get("r_rect")
    t_c0_v = calib.get("t_velo_to_cam")
    t_v_c0 = calib.get("t_cam_to_velo")
    calib_obj = Kitti360Calib(
        t_c0_v=t_c0_v,
        t_v_c0=t_v_c0,
        r_rect_00=r_rect,
        p_rect_00=p_rect,
        k=k,
        image_size=(w, h),
        warnings={},
    )
    proj = project_cam0_to_image(points_cam0, calib_obj, use_rect=use_rect, y_flip=y_flip, sanity=False)
    return proj["u"], proj["v"], proj["valid"], proj["in_img"]


def world_points_to_image(
    points_world: np.ndarray,
    pose: Tuple[float, ...],
    calib: Dict[str, np.ndarray],
    image_shape: Tuple[int, int],
    use_rect: bool = True,
    y_flip: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if points_world.size == 0:
        empty = np.zeros((0,), dtype=float)
        return empty, empty, empty, empty
    h, w = int(image_shape[0]), int(image_shape[1])
    k = calib["k"]
    p_rect = calib.get("p_rect")
    r_rect = calib.get("r_rect")
    t_c0_v = calib.get("t_velo_to_cam")
    t_v_c0 = calib.get("t_cam_to_velo")
    calib_obj = Kitti360Calib(
        t_c0_v=t_c0_v,
        t_v_c0=t_v_c0,
        r_rect_00=r_rect,
        p_rect_00=p_rect,
        k=k,
        image_size=(w, h),
        warnings={},
    )
    proj = project_world_to_image_pose(points_world, pose, calib_obj, use_rect=use_rect, y_flip_mode="fixed_true", sanity=False)
    return proj["u"], proj["v"], proj["valid"], proj["in_img"]


def image_mask_to_world_geom(
    geom: Polygon | LineString,
    calib: Dict[str, np.ndarray],
    pose: Tuple[float, ...],
    cam_to_pose: Optional[np.ndarray],
) -> Optional[Polygon | LineString]:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        coords = []
        for u, v in geom.coords:
            pt = _pixel_to_world(u, v, calib, pose, cam_to_pose)
            if pt is not None:
                coords.append(pt)
        if len(coords) < 2:
            return None
        return LineString(coords)
    if geom.geom_type == "Polygon":
        coords = []
        for u, v in geom.exterior.coords:
            pt = _pixel_to_world(u, v, calib, pose, cam_to_pose)
            if pt is not None:
                coords.append(pt)
        if len(coords) < 3:
            return None
        return Polygon(coords)
    return None


def _bbox_from_poly(poly: Optional[Polygon]) -> Optional[List[float]]:
    if poly is None or poly.is_empty:
        return None
    minx, miny, maxx, maxy = poly.bounds
    return [float(minx), float(miny), float(maxx), float(maxy)]


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_err(a: List[float], b: List[float]) -> Optional[float]:
    if not a or not b:
        return None
    cx1 = (a[0] + a[2]) / 2.0
    cy1 = (a[1] + a[3]) / 2.0
    cx2 = (b[0] + b[2]) / 2.0
    cy2 = (b[1] + b[3]) / 2.0
    return float(np.hypot(cx2 - cx1, cy2 - cy1))


def compute_roundtrip_metrics(
    raw_poly: Optional[Polygon],
    raw_bbox: Optional[List[float]],
    reproj_poly: Optional[Polygon],
    reproj_bbox: Optional[List[float]],
    iou_mask: Optional[float] = None,
    iou_dilated: Optional[float] = None,
) -> RoundtripMetrics:
    iou_bbox = None
    area_ratio = None
    if raw_poly is not None and reproj_poly is not None and not raw_poly.is_empty and not reproj_poly.is_empty:
        inter = raw_poly.intersection(reproj_poly).area
        union = raw_poly.union(reproj_poly).area
        if union > 0:
            if iou_mask is None:
                iou_mask = float(inter / union)
            area_ratio = float(reproj_poly.area / raw_poly.area) if raw_poly.area > 0 else None
    if raw_bbox and reproj_bbox:
        iou_bbox = float(_bbox_iou(raw_bbox, reproj_bbox))
    center_err = _center_err(raw_bbox or [], reproj_bbox or [])
    return RoundtripMetrics(
        reproj_iou_mask=iou_mask,
        reproj_iou_dilated=iou_dilated,
        reproj_iou_bbox=iou_bbox,
        reproj_center_err_px=center_err,
        reproj_area_ratio=area_ratio,
    )


def select_best_offset(
    offsets: Dict[int, float],
) -> Tuple[Optional[int], Optional[float]]:
    if not offsets:
        return None, None
    best = min(offsets.items(), key=lambda kv: kv[1])
    return int(best[0]), float(best[1])


def _pose_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    c1 = float(np.cos(yaw))
    s1 = float(np.sin(yaw))
    c2 = float(np.cos(pitch))
    s2 = float(np.sin(pitch))
    c3 = float(np.cos(roll))
    s3 = float(np.sin(roll))
    r_z = np.array([[c1, -s1, 0.0], [s1, c1, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    r_y = np.array([[c2, 0.0, s2], [0.0, 1.0, 0.0], [-s2, 0.0, c2]], dtype=float)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, c3, -s3], [0.0, s3, c3]], dtype=float)
    return r_z @ r_y @ r_x


def _pixel_to_world(
    u: float,
    v: float,
    calib: Dict[str, np.ndarray],
    pose: Tuple[float, ...],
    cam_to_pose: Optional[np.ndarray],
) -> Optional[Tuple[float, float]]:
    k = calib["k"]
    r_rect = calib["r_rect"]
    cam_to_velo = calib["t_cam_to_velo"]
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    if fx == 0 or fy == 0:
        return None
    dir_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)
    r_rect_inv = np.linalg.inv(r_rect)
    dir_cam = r_rect_inv.dot(dir_cam)
    if len(pose) == 6 and cam_to_pose is not None:
        x, y, z, roll, pitch, yaw = pose
        dir_pose = cam_to_pose[:3, :3].dot(dir_cam)
        r_world_pose = _pose_rotation_matrix(roll, pitch, yaw)
        dir_world = r_world_pose.dot(dir_pose)
        origin_pose = cam_to_pose[:3, 3]
        origin_world = np.array([x, y, z], dtype=float) + r_world_pose.dot(origin_pose)
        if dir_world[2] >= -1e-6:
            return None
        t = -origin_world[2] / dir_world[2]
        if t <= 0:
            return None
        hit = origin_world + t * dir_world
        return float(hit[0]), float(hit[1])
    if len(pose) != 3:
        return None
    pose_xy = (pose[0], pose[1])
    yaw = pose[2]
    dir_velo = cam_to_velo[:3, :3].dot(dir_cam)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    r_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    dir_world = r_yaw.dot(dir_velo)
    cam_offset = cam_to_velo[:3, 3]
    origin_z = float(abs(cam_offset[2]))
    origin = np.array(
        [
            pose_xy[0] + c * cam_offset[0] - s * cam_offset[1],
            pose_xy[1] + s * cam_offset[0] + c * cam_offset[1],
            origin_z,
        ],
        dtype=float,
    )
    if dir_world[2] >= -1e-6:
        return None
    t = -origin[2] / dir_world[2]
    if t <= 0:
        return None
    hit = origin + t * dir_world
    return float(hit[0]), float(hit[1])

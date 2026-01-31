from __future__ import annotations

from typing import Dict

import numpy as np


def validate_depth(z_cam: np.ndarray) -> Dict[str, float]:
    if z_cam.size == 0:
        raise ValueError("depth_empty")
    valid = z_cam[np.isfinite(z_cam) & (z_cam > 0)]
    if valid.size == 0:
        raise ValueError("depth_invalid")
    z_med = float(np.median(valid))
    z_max = float(np.max(valid))
    if z_med < 3.0 or z_med > 45.0 or z_max > 200.0:
        raise ValueError("depth_out_of_range")
    return {"z_cam_median": z_med, "z_cam_max": z_max}


def validate_uv_spread(u: np.ndarray, v: np.ndarray, in_mask: np.ndarray) -> Dict[str, float]:
    if np.sum(in_mask) == 0:
        raise ValueError("uv_in_image_empty")
    uu = u[in_mask]
    vv = v[in_mask]
    u_range = float(np.max(uu) - np.min(uu))
    v_range = float(np.max(vv) - np.min(vv))
    ui = np.round(uu).astype(np.int32)
    vi = np.round(vv).astype(np.int32)
    unique_pix = int(np.unique(vi * 100000 + ui).size)
    if u_range <= 50.0 or v_range <= 30.0 or unique_pix <= 1000:
        raise ValueError("uv_spread_insufficient")
    return {"u_range": u_range, "v_range": v_range, "unique_pixel_count": float(unique_pix)}


def validate_in_image_ratio(in_mask: np.ndarray) -> Dict[str, float]:
    ratio = float(np.sum(in_mask)) / max(1, int(in_mask.size))
    if ratio < 0.005:
        raise ValueError("in_image_ratio_low")
    return {"in_image_ratio": ratio}


def validate_matrix(t: np.ndarray) -> None:
    if t.shape != (4, 4):
        raise ValueError("matrix_shape")
    r = t[:3, :3]
    det = float(np.linalg.det(r))
    if abs(det - 1.0) > 1e-2:
        raise ValueError("matrix_det")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-2):
        raise ValueError("matrix_orthonormal")

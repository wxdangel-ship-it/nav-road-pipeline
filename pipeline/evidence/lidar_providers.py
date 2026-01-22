from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from shapely.geometry import MultiPoint

from pipeline.datasets.kitti360_io import load_kitti360_lidar_points_world
from pipeline.evidence.lidar_provider_registry import LidarProviderContext, register_lidar_provider


LOG = logging.getLogger("lidar_providers")


@dataclass
class LidarFrame:
    drive_id: str
    frame_id: str
    lidar_path: str


class BaseLidarProvider:
    def __init__(self, ctx: LidarProviderContext):
        self.ctx = ctx

    def load(self) -> None:
        return None

    def infer(self, frames: List[LidarFrame], out_dir, debug_dir) -> dict:
        raise NotImplementedError


@register_lidar_provider("dummy_lidar")
class DummyLidarProvider(BaseLidarProvider):
    def infer(self, frames: List[LidarFrame], out_dir, debug_dir) -> dict:
        return {
            "backend_status": "real",
            "fallback_used": False,
            "backend_reason": "",
            "counts": {},
            "features": [],
            "errors": [],
        }


@register_lidar_provider("pc_simple_ground")
class SimpleGroundProvider(BaseLidarProvider):
    def load(self) -> None:
        return None

    def infer(self, frames: List[LidarFrame], out_dir, debug_dir) -> dict:
        cfg = self.ctx.model_cfg or {}
        z_min = float(cfg.get("z_min", -2.0))
        z_max = float(cfg.get("z_max", 0.5))
        min_points = int(cfg.get("min_points", 500))
        max_points = int(cfg.get("max_points", 10000))
        label = str(cfg.get("label", "ground_surface"))
        score = float(cfg.get("score", 0.6))

        features = []
        errors = []
        counts = {}

        for frame in frames:
            try:
                pts = load_kitti360_lidar_points_world(
                    self.ctx.data_root, frame.drive_id, frame.frame_id
                )
            except Exception as exc:
                errors.append(f"{frame.drive_id}:{frame.frame_id}:{exc}")
                continue
            if pts.size == 0:
                continue
            mask = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
            ground = pts[mask]
            if ground.shape[0] < min_points:
                continue
            if ground.shape[0] > max_points:
                idx = np.random.choice(ground.shape[0], size=max_points, replace=False)
                ground = ground[idx]
            hull = MultiPoint(ground[:, :2]).convex_hull
            if hull.is_empty:
                continue
            props = {
                "provider_id": cfg.get("model_id", "pc_simple_ground_v1"),
                "model_id": cfg.get("model_id", "pc_simple_ground_v1"),
                "model_version": cfg.get("model_version", "v1"),
                "ckpt_hash": cfg.get("ckpt_hash", ""),
                "drive_id": frame.drive_id,
                "frame_id": frame.frame_id,
                "label": label,
                "score": score,
                "points_count": int(ground.shape[0]),
                "geometry_frame": "utm32",
                "evidence_strength": "strong",
                "backend_status": "real",
                "fallback_used": False,
                "fallback_from": "",
                "fallback_to": "",
                "backend_reason": "",
            }
            features.append({"geometry": hull, "properties": props, "class": label})
            counts[label] = counts.get(label, 0) + 1

        return {
            "backend_status": "real",
            "fallback_used": False,
            "backend_reason": "",
            "counts": counts,
            "features": features,
            "errors": errors,
        }


@register_lidar_provider("pc_open3dis")
class Open3DISProvider(BaseLidarProvider):
    def load(self) -> None:
        cfg = self.ctx.model_cfg or {}
        backend = str(cfg.get("backend", "auto"))
        if backend == "fallback":
            return None
        raise RuntimeError("missing_dependency:open3dis_runner")

    def infer(self, frames: List[LidarFrame], out_dir, debug_dir) -> dict:
        return {
            "backend_status": "unavailable",
            "fallback_used": False,
            "backend_reason": "missing_dependency",
            "counts": {},
            "features": [],
            "errors": [],
        }

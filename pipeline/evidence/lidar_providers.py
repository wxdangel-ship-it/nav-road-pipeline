from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
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
        python_path = str(cfg.get("open3dis_python", "")).strip()
        if not python_path:
            path_file = Path("cache") / "open3dis_python_path.txt"
            if path_file.exists():
                python_path = path_file.read_text(encoding="utf-8").strip()
        runner_path = str(cfg.get("open3dis_runner", "tools/open3dis_runner.py")).strip()

        self._python_path = python_path
        self._runner_path = runner_path
        self._backend = backend

        if backend == "fallback":
            return None
        if not python_path or not Path(python_path).exists():
            if backend == "real":
                raise RuntimeError("missing_dependency:open3dis_env")
            return None
        if not runner_path or not Path(runner_path).exists():
            if backend == "real":
                raise RuntimeError("missing_dependency:open3dis_runner")
            return None

    def infer(self, frames: List[LidarFrame], out_dir, debug_dir) -> dict:
        cfg = self.ctx.model_cfg or {}
        if not getattr(self, "_python_path", "") or not getattr(self, "_runner_path", ""):
            return {
                "backend_status": "unavailable",
                "fallback_used": False,
                "backend_reason": "missing_dependency",
                "counts": {},
                "features": [],
                "errors": [],
            }

        out_dir = Path(out_dir)
        debug_dir = Path(debug_dir)
        jobs_path = debug_dir / "open3dis_jobs.jsonl"
        out_path = debug_dir / "open3dis_results.jsonl"
        err_path = debug_dir / "open3dis_errors.txt"
        if out_path.exists():
            out_path.unlink()
        if err_path.exists():
            err_path.unlink()

        with jobs_path.open("w", encoding="utf-8") as f:
            for frame in frames:
                f.write(
                    json.dumps(
                        {
                            "drive_id": frame.drive_id,
                            "frame_id": frame.frame_id,
                            "lidar_path": frame.lidar_path,
                        }
                    )
                    + "\n"
                )

        cmd = [
            self._python_path,
            self._runner_path,
            "--data-root",
            self.ctx.data_root,
            "--jobs",
            str(jobs_path),
            "--out",
            str(out_path),
            "--errors",
            str(err_path),
            "--image-run-root",
            str(cfg.get("image_run_root", "")),
            "--image-provider",
            str(cfg.get("image_provider", "")),
            "--class-whitelist",
            ",".join(cfg.get("class_whitelist", []) or []),
            "--min-points",
            str(cfg.get("min_points", 60)),
            "--grid-size",
            str(cfg.get("grid_size_m", 4.0)),
            "--max-instances",
            str(cfg.get("max_instances_per_frame", 6)),
            "--min-area-m2",
            str(cfg.get("min_area_m2", 1.0)),
            "--z-min",
            str(cfg.get("z_min", -2.0)),
            "--z-max",
            str(cfg.get("z_max", 2.5)),
        ]
        try:
            subprocess.check_call(cmd, cwd=str(Path(__file__).resolve().parents[2]))
        except Exception as exc:
            return {
                "backend_status": "unavailable",
                "fallback_used": False,
                "backend_reason": f"runtime_error:{exc}",
                "counts": {},
                "features": [],
                "errors": [str(exc)],
            }

        features = []
        counts = {}
        errors = []
        if err_path.exists():
            errors = [line for line in err_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        if out_path.exists():
            import shapely.wkt

            for line in out_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                geom_wkt = row.get("geometry_wkt") or ""
                if not geom_wkt:
                    continue
                try:
                    geom = shapely.wkt.loads(geom_wkt)
                except Exception:
                    continue
                props = row.get("properties") or {}
                label = props.get("label", "unknown")
                props.update(
                    {
                        "provider_id": cfg.get("model_id", "pc_open3dis_v1"),
                        "model_id": cfg.get("model_id", "pc_open3dis_v1"),
                        "model_version": cfg.get("model_version", "v1"),
                        "ckpt_hash": cfg.get("ckpt_hash", ""),
                        "geometry_frame": "utm32",
                        "backend_status": "real",
                        "fallback_used": False,
                        "fallback_from": "",
                        "fallback_to": "",
                        "backend_reason": "",
                    }
                )
                features.append({"geometry": geom, "properties": props, "class": label})
                counts[label] = counts.get(label, 0) + 1

        return {
            "backend_status": "real",
            "fallback_used": False,
            "backend_reason": "",
            "counts": counts,
            "features": features,
            "errors": errors,
        }

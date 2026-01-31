from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml


REQUIRED_KEYS = [
    "KITTI_ROOT",
    "TARGET_EPSG",
    "ROI_BUFFER_M",
    "RASTER_RES_M",
    "VOXEL_SIZE_M",
    "MAX_FRAMES",
    "GROUND_BAND_DZ_M",
    "MIN_DENSITY",
    "ROAD_ROUGHNESS_MAX_M",
    "ROAD_CLOSE_RADIUS_M",
    "BG_WIN_RADIUS_M",
    "MARKING_AREA_RATIO_MIN",
    "MARKING_AREA_RATIO_MAX",
    "MARKING_CLOSE_RADIUS_M",
    "MARKING_SCORE_TH",
    "MARKING_SCORE_PCTL_CHOSEN",
    "CROSSWALK_MERGE_RADIUS_M",
    "CROSSWALK_W_MIN_M",
    "CROSSWALK_W_MAX_M",
    "CROSSWALK_L_MIN_M",
    "CROSSWALK_L_MAX_M",
    "CROSSWALK_AREA_MIN_M2",
    "CROSSWALK_AREA_MAX_M2",
    "CROSSWALK_MIN_COMPONENTS",
    "TIME_BUDGET_H",
    "TRAJ_STRIDE",
    "STRIDE",
    "LIDAR_WORLD_MODE",
    "OVERWRITE",
]


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return dict(data)


def _detect_kitti_root(cfg: Dict[str, Any]) -> str:
    env_root = str(cfg.get("KITTI_ROOT") or "").strip()
    if env_root:
        return env_root
    env_var = os.environ.get("POC_DATA_ROOT", "").strip()
    if env_var:
        return env_var
    for candidate in [
        r"E:\KITTI360\KITTI-360",
        r"D:\KITTI360\KITTI-360",
        r"C:\KITTI360\KITTI-360",
    ]:
        if Path(candidate).exists():
            return candidate
    return ""


def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _normalize(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    return obj


def get_params_hash(cfg: Dict[str, Any]) -> str:
    payload = _normalize(dict(cfg))
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _write_resolved(run_dir: Path, cfg: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "resolved_config.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
    params_hash = get_params_hash(cfg)
    (run_dir / "params_hash.txt").write_text(params_hash + "\n", encoding="utf-8")


def _assert_required(cfg: Dict[str, Any], required: Iterable[str]) -> None:
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys: {missing}")


def resolve_config(base_cfg: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    defaults = {
        "KITTI_ROOT": "",
        "TARGET_EPSG": 32632,
        "ROI_BUFFER_M": 20.0,
        "RASTER_RES_M": 0.20,
        "VOXEL_SIZE_M": 0.05,
        "MAX_FRAMES": 3000,
        "GROUND_BAND_DZ_M": 0.15,
        "MIN_DENSITY": 8,
        "ROAD_ROUGHNESS_MAX_M": 0.05,
        "ROAD_CLOSE_RADIUS_M": 0.6,
        "BG_WIN_RADIUS_M": 3.0,
        "MARKING_AREA_RATIO_MIN": 0.005,
        "MARKING_AREA_RATIO_MAX": 0.04,
        "MARKING_CLOSE_RADIUS_M": 0.6,
        "MARKING_SCORE_TH": -1.0,
        "MARKING_SCORE_PCTL_CHOSEN": -1.0,
        "CROSSWALK_MERGE_RADIUS_M": 1.0,
        "CROSSWALK_W_MIN_M": 2.5,
        "CROSSWALK_W_MAX_M": 8.0,
        "CROSSWALK_L_MIN_M": 3.0,
        "CROSSWALK_L_MAX_M": 20.0,
        "CROSSWALK_AREA_MIN_M2": 10.0,
        "CROSSWALK_AREA_MAX_M2": 200.0,
        "CROSSWALK_MIN_COMPONENTS": 4,
        "TIME_BUDGET_H": 6.0,
        "TRAJ_STRIDE": 5,
        "STRIDE": 1,
        "LIDAR_WORLD_MODE": "fullpose",
        "OVERWRITE": True,
        "GOLDEN_DRIVES_FILE": "configs/golden_drives.txt",
    }
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v

    if not str(cfg.get("KITTI_ROOT") or "").strip():
        cfg["KITTI_ROOT"] = _detect_kitti_root(cfg)

    _assert_required(cfg, REQUIRED_KEYS)
    _write_resolved(run_dir, cfg)
    return cfg


def update_resolved_config(cfg: Dict[str, Any], run_dir: Path) -> str:
    _assert_required(cfg, REQUIRED_KEYS)
    _write_resolved(run_dir, cfg)
    return get_params_hash(cfg)


__all__ = [
    "REQUIRED_KEYS",
    "get_params_hash",
    "load_yaml",
    "resolve_config",
    "update_resolved_config",
]

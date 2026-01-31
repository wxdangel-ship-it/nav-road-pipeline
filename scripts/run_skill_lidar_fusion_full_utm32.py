from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from pipeline.lidar_fusion.fuse_frames import collect_input_fingerprints, fuse_frames_to_las
from scripts.pipeline_common import (
    ensure_dir,
    ensure_overwrite,
    now_ts,
    setup_logging,
    write_csv,
    write_json,
    write_text,
)

LOG = logging.getLogger("lidar_fusion_skill")
REPO_ROOT = Path(__file__).resolve().parents[1]

# =========================
# 参数区（按需修改）
# =========================
MODE = "single"  # single | batch
JOB_FILE = r"configs\jobs\lidar_fusion\0010_f000_300.yaml"
BATCH_FILE = r"configs\jobs\lidar_fusion\golden8_full.yaml"
OVERWRITE = False
DRY_RUN = False
MAX_JOBS = 0  # 0 表示不限制


DEFAULTS = {
    "stride": 1,
    "output_mode": "utm32",
    "output_format": "laz",
    "require_laz": True,
    "use_r_rect_with_cam0_to_world": True,
    "require_cam0_to_world": True,
    "allow_poses_fallback": False,
    "enable_chunking": True,
    "max_parts": 8,
    "target_laz_mb_per_part": 1200,
    "fit_gate_pass_m": 1.0,
    "fit_gate_warn_m": 1.5,
    "banding_max_min_nonzero_dy_m": 0.01,
}


def _load_yaml(path: Path) -> Dict[str, object]:
    try:
        import yaml  # type: ignore
    except Exception:
        print("missing dependency: PyYAML. Install via: python -m pip install pyyaml")
        raise SystemExit(2)
    if not path.exists():
        raise FileNotFoundError(f"job_yaml_missing:{path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _expand_placeholders(text: str, ctx: Dict[str, str]) -> str:
    def repl_percent(match: re.Match[str]) -> str:
        var = match.group(1)
        return ctx.get(var, os.environ.get(var, ""))

    def repl_brace(match: re.Match[str]) -> str:
        var = match.group(1)
        return ctx.get(var, os.environ.get(var, ""))

    out = re.sub(r"%([^%]+)%", repl_percent, text)
    out = re.sub(r"\$\{([^}]+)\}", repl_brace, out)
    return out


def _resolve_path(raw: str, base_dir: Path, ctx: Dict[str, str]) -> Path:
    expanded = _expand_placeholders(raw, ctx)
    path = Path(expanded)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return Path(os.path.normpath(str(path)))


def _normalize_job(job: Dict[str, object], job_dir: Path) -> Dict[str, object]:
    out = dict(DEFAULTS)
    out.update(job)
    required = ["kitti_root", "drive_id", "frame_start", "frame_end"]
    missing = []
    for k in required:
        if k not in out:
            missing.append(k)
            continue
        val = out.get(k)
        if val is None:
            missing.append(k)
            continue
        if isinstance(val, str) and not val.strip():
            missing.append(k)
    if missing:
        raise ValueError(f"job_missing_required:{missing}")
    output_mode = str(out.get("output_mode", "utm32")).lower()
    transform_json = out.get("transform_json") or out.get("world_to_utm32_transform_json")
    transform = out.get("transform")
    if transform is None and transform_json:
        transform = {"mode": "use_file", "file": transform_json}
    if transform is None:
        if output_mode == "utm32":
            transform = {"mode": "auto_fit"}
        else:
            transform = {"mode": "none"}
    if not isinstance(transform, dict):
        raise ValueError("transform_invalid")
    transform.setdefault("mode", "auto_fit" if output_mode == "utm32" else "none")
    transform.setdefault("file", "")
    transform.setdefault("sample_max_frames", out.get("sample_max_frames", 300))
    transform.setdefault("gate_pass_m", out.get("fit_gate_pass_m", 1.0))
    transform.setdefault("gate_warn_m", out.get("fit_gate_warn_m", 1.5))
    out["transform"] = transform
    out["output_mode"] = output_mode

    ctx = {
        "REPO_ROOT": str(REPO_ROOT),
        "JOB_DIR": str(job_dir),
    }
    raw_kitti = str(out.get("kitti_root", "")).strip()
    if not raw_kitti:
        raise ValueError("kitti_root_missing")
    expanded_kitti = _expand_placeholders(raw_kitti, ctx)
    if not expanded_kitti.strip():
        raise ValueError("kitti_root_unresolved:set_env_or_update_job")
    if re.search(r"%[^%]+%", expanded_kitti) or re.search(r"\$\{[^}]+\}", expanded_kitti):
        raise ValueError("kitti_root_unresolved:set_env_or_update_job")
    kitti_path = _resolve_path(expanded_kitti, REPO_ROOT, ctx)
    if not kitti_path.exists():
        raise ValueError("kitti_root_not_found:set_env_or_update_job")
    out["kitti_root"] = str(kitti_path)
    return out


def _load_job_from_path(path: str) -> Tuple[Dict[str, object], Path]:
    job_path = Path(path)
    raw = _load_yaml(job_path)
    if isinstance(raw, dict) and "jobs" in raw:
        raise ValueError("batch_yaml_used_as_job")
    if not isinstance(raw, dict):
        raise ValueError("job_yaml_invalid")
    return _normalize_job(raw, job_path.parent), job_path.parent


def _load_batch_paths(path: str) -> Tuple[List[str], Path]:
    batch_path = Path(path)
    raw = _load_yaml(batch_path)
    jobs = raw.get("jobs", [])
    if not isinstance(jobs, list):
        raise ValueError("batch_yaml_invalid_jobs")
    paths: List[str] = []
    for item in jobs:
        if isinstance(item, str):
            paths.append(item)
        elif isinstance(item, dict) and "path" in item:
            paths.append(str(item["path"]))
    return paths, batch_path.parent


def _parse_part_range(name: str) -> Tuple[str, str]:
    stem = Path(name).stem
    parts = stem.split("_part_")
    if len(parts) != 2:
        return "", ""
    rng = parts[1]
    if "_" not in rng:
        return "", ""
    start, end = rng.split("_", 1)
    return start, end


def _write_bbox_geojson(path: Path, bbox: Tuple[float, float, float, float, float, float]) -> None:
    xmin, ymin, _zmin, xmax, ymax, _zmax = bbox
    coords = [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
        [xmin, ymin],
    ]
    geo = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "bbox_utm32"},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(geo, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_head(path: Path, n_bytes: int = 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()


def _banding_check(path: Path, max_points: int = 2_000_000) -> Dict[str, object]:
    import laspy
    import numpy as np

    with laspy.open(path) as reader:
        total = reader.header.point_count
        if total <= 0:
            return {"ok": False, "reason": "empty", "min_nonzero_dy": None, "unique_y_1mm": 0, "sample_n": 0}
        if total <= max_points:
            points = reader.read()
            y = np.asarray(points.y, dtype=np.float64)
        else:
            step = max(1, total // max_points)
            ys: List[float] = []
            count = 0
            for chunk in reader.chunk_iterator(1_000_000):
                y = np.asarray(chunk.y, dtype=np.float64)
                for j in range(0, y.size, step):
                    ys.append(float(y[j]))
                    count += 1
                    if count >= max_points:
                        break
                if count >= max_points:
                    break
            y = np.asarray(ys, dtype=np.float64)
    if y.size == 0:
        return {"ok": False, "reason": "empty", "min_nonzero_dy": None, "unique_y_1mm": 0, "sample_n": 0}
    y_round = np.round(y, 3)
    y_unique = np.unique(y_round)
    dy = np.diff(np.sort(y_unique))
    nonzero = dy[dy > 0]
    min_nonzero = float(np.min(nonzero)) if nonzero.size else None
    return {
        "ok": bool(min_nonzero is not None and min_nonzero <= 0.01),
        "min_nonzero_dy": min_nonzero,
        "unique_y_1mm": int(y_unique.size),
        "sample_n": int(y.size),
    }


def _select_parts(paths: List[str]) -> List[str]:
    if not paths:
        return []
    if len(paths) <= 2:
        return paths
    mid = len(paths) // 2
    return [paths[0], paths[mid], paths[-1]]


def _find_pose_file(data_root: Path, drive_id: str, names: List[str]) -> Path:
    base_dirs = [
        data_root / "data_poses" / drive_id,
        data_root / "data_poses" / drive_id / "poses",
        data_root / "data_poses" / drive_id / "pose",
    ]
    for base in base_dirs:
        if not base.exists():
            continue
        for name in names:
            cand = base / name
            if cand.exists():
                return cand
    raise FileNotFoundError(f"missing_pose_file:{names}")


def _find_oxts_dir(data_root: Path, drive_id: str) -> Path:
    candidates = [
        data_root / "data_poses_oxts" / drive_id / "oxts" / "data",
        data_root / "data_poses_oxts_extract" / drive_id / "oxts" / "data",
        data_root / "data_poses" / drive_id / "oxts" / "data",
        data_root / "data_poses" / "oxts" / drive_id / "oxts" / "data",
        data_root / "data_poses" / "oxts" / drive_id / "data",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"missing_oxts_dir:{drive_id}")


def _parse_pose_map(path: Path) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        parts = [p for p in line.strip().split() if p]
        if not parts:
            continue
        frame_id = None
        nums = parts
        if len(parts) in {13, 17}:
            frame_id = parts[0]
            nums = parts[1:]
        elif len(parts) in {12, 16}:
            frame_id = f"{idx:010d}"
        else:
            continue
        try:
            vals = [float(v) for v in nums]
        except ValueError:
            continue
        if len(vals) not in {12, 16}:
            continue
        key = str(frame_id)
        out[key] = vals
        if key.isdigit():
            out[f"{int(key):010d}"] = vals
    return out


def _read_oxts_frame(oxts_dir: Path, frame_id: str) -> Tuple[float, float, float]:
    path = oxts_dir / f"{frame_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"missing_oxts:{path}")
    parts = path.read_text(encoding="utf-8").strip().split()
    if len(parts) < 6:
        raise ValueError("parse_error:oxts")
    lat = float(parts[0])
    lon = float(parts[1])
    alt = float(parts[2])
    return lat, lon, alt


def _fit_similarity_2d(src: List[List[float]], dst: List[List[float]]) -> Tuple[float, float, float, float]:
    import numpy as np

    src_np = np.asarray(src, dtype=np.float64)
    dst_np = np.asarray(dst, dtype=np.float64)
    mu_src = src_np.mean(axis=0)
    mu_dst = dst_np.mean(axis=0)
    src_c = src_np - mu_src
    dst_c = dst_np - mu_dst
    cov = (dst_c.T @ src_c) / float(src_np.shape[0])
    u, s, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    var_src = float(np.mean(np.sum(src_c**2, axis=1)))
    scale = float(np.sum(s) / var_src) if var_src > 0 else 1.0
    t = mu_dst - scale * (r @ mu_src)
    yaw = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return float(t[0]), float(t[1]), yaw, scale


def _apply_similarity_2d(src: List[List[float]], dx: float, dy: float, yaw_deg: float, scale: float) -> List[List[float]]:
    import numpy as np

    src_np = np.asarray(src, dtype=np.float64)
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    x = src_np[:, 0]
    y = src_np[:, 1]
    x2 = scale * (c * x - s * y) + dx
    y2 = scale * (s * x + c * y) + dy
    return np.stack([x2, y2], axis=1).tolist()


def _auto_fit_world_to_utm32(
    data_root: Path,
    drive_id: str,
    frame_start: int,
    frame_end: int,
    sample_max_frames: int,
    gate_pass_m: float,
    gate_warn_m: float,
    report_dir: Path,
) -> Dict[str, object]:
    try:
        from pyproj import Transformer
    except Exception as exc:
        raise RuntimeError(f"pyproj_missing:{exc}") from exc

    cam0_to_world_path = _find_pose_file(data_root, drive_id, ["cam0_to_world.txt"])
    cam0_to_world = _parse_pose_map(cam0_to_world_path)
    oxts_dir = _find_oxts_dir(data_root, drive_id)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)

    full_ids = [f"{i:010d}" for i in range(int(frame_start), int(frame_end) + 1)]
    src_xy: List[List[float]] = []
    dst_xy: List[List[float]] = []
    dz_list: List[float] = []
    used_frames: List[str] = []

    for fid in full_ids:
        if fid not in cam0_to_world:
            continue
        try:
            lat, lon, alt = _read_oxts_frame(oxts_dir, fid)
        except Exception:
            continue
        utm_e, utm_n = transformer.transform(lon, lat)
        pose = cam0_to_world[fid]
        wx, wy, wz = float(pose[3]), float(pose[7]), float(pose[11])
        src_xy.append([wx, wy])
        dst_xy.append([utm_e, utm_n])
        dz_list.append(float(alt - wz))
        used_frames.append(fid)

    if len(src_xy) < 3:
        raise RuntimeError("insufficient_common_frames")

    total = len(src_xy)
    if total > sample_max_frames:
        idx = [int(i) for i in (list(range(0, total, max(1, total // sample_max_frames))))][:sample_max_frames]
        src_xy = [src_xy[i] for i in idx]
        dst_xy = [dst_xy[i] for i in idx]
        dz_list = [dz_list[i] for i in idx]
        used_frames = [used_frames[i] for i in idx]

    dx, dy, yaw_deg, scale = _fit_similarity_2d(src_xy, dst_xy)
    dz = float(sorted(dz_list)[len(dz_list) // 2]) if dz_list else 0.0
    pred = _apply_similarity_2d(src_xy, dx, dy, yaw_deg, scale)
    residual = []
    for i in range(len(pred)):
        dx_i = pred[i][0] - dst_xy[i][0]
        dy_i = pred[i][1] - dst_xy[i][1]
        residual.append(math.sqrt(dx_i * dx_i + dy_i * dy_i))
    rms = float(math.sqrt(sum(r * r for r in residual) / float(len(residual))))

    gate_status = "PASS"
    if rms > gate_warn_m:
        gate_status = "FAIL"
    elif rms > gate_pass_m:
        gate_status = "WARN"

    report = {
        "drive_id": drive_id,
        "frame_range": [frame_start, frame_end],
        "common_frames_used": int(len(src_xy)),
        "frame_span": [frame_start, frame_end],
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
        "yaw_deg": float(yaw_deg),
        "scale": float(scale),
        "rms_m": float(rms),
        "gate_status": gate_status,
        "gate_thresholds": {"pass": gate_pass_m, "warn": gate_warn_m},
        "paths": {"cam0_to_world": str(cam0_to_world_path), "oxts_dir": str(oxts_dir)},
    }

    pairs = []
    for i, fid in enumerate(used_frames):
        pairs.append(
            {
                "frame": fid,
                "world_x": float(src_xy[i][0]),
                "world_y": float(src_xy[i][1]),
                "utm_e": float(dst_xy[i][0]),
                "utm_n": float(dst_xy[i][1]),
                "residual": float(residual[i]),
            }
        )

    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "world_to_utm32_report.json", report)
    write_csv(
        report_dir / "world_to_utm32_pairs.csv",
        pairs,
        ["frame", "world_x", "world_y", "utm_e", "utm_n", "residual"],
    )
    summary = (
        f"world->utm32 fit: frames={len(src_xy)}, rms={rms:.3f}m, scale={scale:.4f}, yaw={yaw_deg:.3f}deg "
        f"=> {gate_status}"
    )
    write_text(report_dir / "world_to_utm32_summary.md", summary)
    return report


def _run_job(job: Dict[str, object], job_dir: Path) -> int:
    run_id = now_ts()
    drive_id = str(job["drive_id"])
    frame_start = int(job["frame_start"])
    frame_end = int(job["frame_end"])
    run_prefix = job.get("run_dir_prefix") or f"runs/lidar_fusion_{drive_id}_{frame_start}_{frame_end}"
    run_dir = Path(f"{run_prefix}_{run_id}")

    if OVERWRITE:
        ensure_overwrite(run_dir)
    ensure_dir(run_dir)
    setup_logging(run_dir / "logs" / "run.log")
    LOG.info("run_start")

    data_root = Path(str(job["kitti_root"]))
    if not data_root.exists():
        write_text(run_dir / "report" / "report.md", f"data_root_missing:{data_root}")
        write_json(run_dir / "report" / "metrics.json", {"status": "FAIL", "reason": "data_root_missing"})
        return 2

    coord_mode = str(job.get("output_mode", "utm32")).lower()
    out_dir = ensure_dir(run_dir / "outputs")
    transform_spec = job.get("transform", {})
    transform_mode = str(transform_spec.get("mode", "auto_fit")).lower()
    transform = None
    gate_status = ""

    ctx = {
        "REPO_ROOT": str(REPO_ROOT),
        "JOB_DIR": str(job_dir),
        "RUN_DIR": str(run_dir),
        "KITTI_ROOT": str(job["kitti_root"]),
    }

    if transform_mode == "none":
        coord_mode = "world"
        transform = None
    elif transform_mode == "use_file":
        raw_file = str(transform_spec.get("file", "")).strip()
        if not raw_file:
            raise ValueError("transform_file_missing")
        file_path = _resolve_path(raw_file, job_dir, ctx)
        if not file_path.exists():
            raise FileNotFoundError(f"transform_file_missing:{file_path}")
        transform = json.loads(file_path.read_text(encoding="utf-8"))
        gate_status = transform.get("gate_status", "")
    elif transform_mode == "auto_fit":
        report_dir = run_dir / "report"
        report = _auto_fit_world_to_utm32(
            data_root=data_root,
            drive_id=drive_id,
            frame_start=frame_start,
            frame_end=frame_end,
            sample_max_frames=int(transform_spec.get("sample_max_frames", 300)),
            gate_pass_m=float(transform_spec.get("gate_pass_m", 1.0)),
            gate_warn_m=float(transform_spec.get("gate_warn_m", 1.5)),
            report_dir=report_dir,
        )
        gate_status = str(report.get("gate_status", ""))
        if gate_status == "FAIL":
            write_json(report_dir / "metrics.json", {"status": "FAIL", "reason": "transform_gate_fail", "transform": report})
            write_text(report_dir / "report.md", "transform_gate_fail")
            print(
                f"job_done: drive_id={drive_id} frames={frame_start}-{frame_end} "
                f"run_dir={run_dir} gate_status=FAIL"
            )
            return 3
        transform = report
    else:
        raise ValueError(f"transform_mode_invalid:{transform_mode}")

    out_path = out_dir / f"fused_points_{coord_mode}.laz"

    result = fuse_frames_to_las(
        data_root=data_root,
        drive_id=drive_id,
        frame_start=frame_start,
        frame_end=frame_end,
        stride=int(job.get("stride", 1)),
        out_path=out_path,
        coord=coord_mode,
        cam_id="image_00",
        overwrite=bool(OVERWRITE),
        output_format=str(job.get("output_format", "laz")),
        require_laz=bool(job.get("require_laz", True)),
        require_cam0_to_world=bool(job.get("require_cam0_to_world", True)),
        allow_poses_fallback=bool(job.get("allow_poses_fallback", False)),
        use_r_rect_with_cam0_to_world=bool(job.get("use_r_rect_with_cam0_to_world", True)),
        world_to_utm32_transform=transform,
        enable_chunking=bool(job.get("enable_chunking", True)),
        target_laz_mb_per_part=float(job.get("target_laz_mb_per_part", 1200)),
        max_parts=int(job.get("max_parts", 8)),
    )

    frame_ids = list(range(int(frame_start), int(frame_end) + 1, max(1, int(job.get("stride", 1)))))
    missing_csv = run_dir / "outputs" / "missing_frames.csv"
    write_csv(missing_csv, result.missing_frames, ["frame_id", "reason"])
    write_json(run_dir / "outputs" / "missing_summary.json", result.missing_summary)
    if coord_mode == "utm32":
        _write_bbox_geojson(run_dir / "outputs" / "bbox_utm32.geojson", result.bbox)

    input_fingerprints = collect_input_fingerprints(data_root, drive_id, [f"{i:010d}" for i in frame_ids])
    output_list = []
    for p in result.output_paths:
        try:
            size = Path(p).stat().st_size
        except Exception:
            size = 0
        start_f, end_f = _parse_part_range(p)
        output_list.append(
            {
                "path": p,
                "size": size,
                "size_mb": size / (1024 * 1024) if size else 0.0,
                "frame_start": start_f,
                "frame_end": end_f,
                "sha256_head": _sha256_head(Path(p)) if size else "",
            }
        )

    if job.get("enable_chunking") and output_list:
        write_json(run_dir / "outputs" / "fused_points_utm32_index.json", {"parts": output_list})

    intensity = result.intensity_stats
    intensity_error = intensity.get("max", 0.0) <= 0.0
    if intensity_error:
        result.errors.append("intensity_all_zero")

    meta = {
        "status": "PASS" if not result.errors else "FAIL",
        "drive_id": drive_id,
        "frames": [frame_start, frame_end],
        "stride": int(job.get("stride", 1)),
        "frames_found_velodyne": result.frames_found_velodyne,
        "frames_processed": result.frames_processed,
        "points_read_total": result.points_read_total,
        "points_written_total": result.points_written_total,
        "coord": result.coord,
        "epsg": result.epsg,
        "pose_source": result.pose_source,
        "use_r_rect_with_cam0_to_world": result.use_r_rect_with_cam0_to_world,
        "total_frames": len(frame_ids),
        "success_frames": len(frame_ids) - len(result.missing_frames),
        "missing_frames": len(result.missing_frames),
        "missing_summary": result.missing_summary,
        "total_points": result.total_points,
        "written_points": result.written_points,
        "bbox": {
            "xmin": result.bbox[0],
            "ymin": result.bbox[1],
            "zmin": result.bbox[2],
            "xmax": result.bbox[3],
            "ymax": result.bbox[4],
            "zmax": result.bbox[5],
        },
        "bbox_check": result.bbox_check,
        "intensity": {
            "rule": result.intensity_rule,
            "min": intensity.get("min", 0.0),
            "mean": intensity.get("mean", 0.0),
            "max": intensity.get("max", 0.0),
            "nonzero_ratio": intensity.get("nonzero_ratio", 0.0),
            "error_all_zero": bool(intensity_error),
        },
        "output": {
            "path": str(result.output_path) if result.output_path else "",
            "format": result.output_format,
            "paths": result.output_paths,
        },
        "transform": transform or {},
        "transform_mode": transform_mode,
        "gate_status": gate_status or (transform or {}).get("gate_status", ""),
        "input_files": input_fingerprints,
        "warnings": result.warnings,
        "errors": result.errors,
    }
    write_json(run_dir / "outputs" / f"fused_points_{result.coord}.meta.json", meta)
    write_json(run_dir / "report" / "metrics.json", meta)
    if result.per_frame_points_sample:
        write_csv(
            run_dir / "report" / "per_frame_points_sample.csv",
            result.per_frame_points_sample,
            ["frame", "points_in", "points_written"],
        )

    banding_limit = float(job.get("banding_max_min_nonzero_dy_m", 0.01))
    banding_samples = _select_parts(result.output_paths or [])
    banding_report = {"parts_checked": [], "pass": True, "limit": banding_limit}
    for part in banding_samples:
        info = _banding_check(Path(part))
        min_nonzero = info.get("min_nonzero_dy")
        ok = bool(min_nonzero is not None and float(min_nonzero) <= banding_limit)
        entry = {"path": part, **info, "ok_by_limit": ok}
        banding_report["parts_checked"].append(entry)
        if not ok:
            banding_report["pass"] = False
    write_json(run_dir / "report" / "banding_audit_full.json", banding_report)
    summary_line = "PASS" if banding_report["pass"] else "FAIL"
    write_text(run_dir / "report" / "banding_summary_full.md", f"banding_check_full: {summary_line}")
    if not banding_report["pass"]:
        raise RuntimeError("banding_check_full_failed")

    report_lines = [
        "# LiDAR fusion full (skill runner)",
        "",
        f"- run_dir: {run_dir}",
        f"- drive_id: {drive_id}",
        f"- frames: {frame_start}-{frame_end} (stride={job.get('stride', 1)})",
        f"- coord: {result.coord} (epsg={result.epsg})",
        f"- pose_source: {result.pose_source}",
        f"- use_r_rect_with_cam0_to_world: {result.use_r_rect_with_cam0_to_world}",
        f"- output_paths: {len(result.output_paths)}",
        f"- total_points: {result.total_points}",
        f"- written_points: {result.written_points}",
        f"- bbox_check: {result.bbox_check.get('ok')} ({result.bbox_check.get('reason')})",
        f"- intensity_max: {intensity.get('max', 0.0):.1f}",
        f"- intensity_nonzero_ratio: {intensity.get('nonzero_ratio', 0.0):.4f}",
    ]
    if intensity_error:
        report_lines.append("- ERROR: intensity_max == 0")
    if result.warnings:
        report_lines.extend(["", "## Warnings"] + [f"- {w}" for w in result.warnings])
    if result.errors:
        report_lines.extend(["", "## Errors"] + [f"- {e}" for e in result.errors])
    write_text(run_dir / "report" / "report.md", "\n".join(report_lines))

    print(
        f"job_done: drive_id={drive_id} frames={frame_start}-{frame_end} "
        f"output_mode={coord_mode} transform_mode={transform_mode} gate_status={gate_status}"
        f" run_dir={run_dir}"
    )
    return 0


def main() -> int:
    if MODE not in {"single", "batch"}:
        raise ValueError("MODE must be single or batch")
    if MODE == "single":
        job, job_dir = _load_job_from_path(JOB_FILE)
        if DRY_RUN:
            print(f"dry_run: single job -> {JOB_FILE}")
            print(
                f"plan: drive_id={job['drive_id']} frames={job['frame_start']}-{job['frame_end']} "
                f"transform.mode={job['transform']['mode']}"
            )
            return 0
        return _run_job(job, job_dir)

    paths, batch_dir = _load_batch_paths(BATCH_FILE)
    if MAX_JOBS > 0:
        paths = paths[: int(MAX_JOBS)]
    if DRY_RUN:
        print(f"dry_run: batch jobs -> {BATCH_FILE}")
        for p in paths:
            ctx = {
                "REPO_ROOT": str(REPO_ROOT),
                "JOB_DIR": str(batch_dir),
            }
            job_path = _resolve_path(p, REPO_ROOT, ctx)
            if not job_path.exists():
                print(f"plan: {p} (missing job file)")
                continue
            job, _job_dir = _load_job_from_path(str(job_path))
            print(
                f"plan: {p} drive_id={job['drive_id']} frames={job['frame_start']}-{job['frame_end']} "
                f"transform.mode={job['transform']['mode']}"
            )
        return 0
    for p in paths:
        ctx = {
            "REPO_ROOT": str(REPO_ROOT),
            "JOB_DIR": str(batch_dir),
        }
        job_path = _resolve_path(p, REPO_ROOT, ctx)
        job, job_dir = _load_job_from_path(str(job_path))
        _run_job(job, job_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.evidence.image_provider_registry import ProviderContext, get_provider  # noqa: E402
from pipeline.evidence import image_providers  # noqa: F401,E402 - register providers


@dataclass
class ImageItem:
    drive_id: str
    frame_id: str
    path: Path
    camera: str
    scene_profile: str


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("run_image_providers")


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_jsonl(path: Path) -> List[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _device_auto() -> tuple[str, bool]:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", True
    except Exception:
        pass
    return "cpu", False


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _safe_unlink(path: Path) -> None:
    if path.exists():
        path.unlink()


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
        return out
    except Exception:
        return ""


def _write_meta(path: Path, img_size: tuple[int, int]) -> None:
    meta = {
        "image_w": int(img_size[0]),
        "image_h": int(img_size[1]),
        "model_input_w": int(img_size[0]),
        "model_input_h": int(img_size[1]),
        "resize_mode": "none",
        "scale": 1.0,
        "pad": [0.0, 0.0],
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _build_feature_store(
    drive_id: str,
    model_out_dir: Path,
    out_run_dir: Path,
    seg_schema: Path,
    feature_schema: Path,
    max_frames: int,
) -> int:
    cmd = [
        sys.executable,
        "tools/build_image_feature_store.py",
        "--drive",
        drive_id,
        "--model-out-dir",
        str(model_out_dir),
        "--out-run-dir",
        str(out_run_dir),
        "--seg-schema",
        str(seg_schema),
        "--feature-schema",
        str(feature_schema),
    ]
    if max_frames and max_frames > 0:
        cmd.extend(["--max-frames", str(max_frames)])
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return proc.returncode


def _build_feature_store_map(
    drive_id: str,
    feature_store: Path,
    out_store: Path,
    camera: str,
    max_frames: int,
) -> int:
    cmd = [
        sys.executable,
        "tools/project_feature_store_to_map.py",
        "--drive",
        drive_id,
        "--feature-store",
        str(feature_store),
        "--out-store",
        str(out_store),
        "--camera",
        camera,
    ]
    if max_frames and max_frames > 0:
        cmd.extend(["--max-frames", str(max_frames)])
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return proc.returncode


def _collect_dets(det_path: Path) -> List[dict]:
    if not det_path.exists():
        return []
    data = json.loads(det_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "detections" in data:
        return data["detections"]
    return []


def _write_evidence_records(
    out_path: Path,
    records: List[dict],
) -> None:
    _safe_unlink(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _backend_defaults(provider_id: str, report: dict) -> dict:
    backend_status = report.get("backend_status") or "real"
    fallback_used = bool(report.get("fallback_used", False))
    fallback_from = report.get("fallback_from") or ""
    fallback_to = report.get("fallback_to") or ""
    backend_reason = report.get("backend_reason") or ""
    if fallback_used and backend_status == "real":
        backend_status = "fallback"
    if fallback_used and not fallback_from:
        fallback_from = provider_id
    return {
        "backend_status": backend_status,
        "fallback_used": fallback_used,
        "fallback_from": fallback_from,
        "fallback_to": fallback_to,
        "backend_reason": backend_reason,
    }


def _env_flag(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default)
    return str(val).strip().lower() in {"1", "true", "yes", "y"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--providers", default="grounded_sam2_v1,gdino_sam2_v1")
    ap.add_argument("--out-run", default="")
    ap.add_argument("--zoo", default="configs/image_model_zoo.yaml")
    ap.add_argument("--seg-schema", default="configs/seg_schema.yaml")
    ap.add_argument("--feature-schema", default="configs/feature_schema.yaml")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--build-feature-store", type=int, default=1)
    ap.add_argument("--build-feature-store-map", type=int, default=1)
    ap.add_argument("--resume", type=int, default=0)
    args = ap.parse_args()

    log = _setup_logger()
    index_path = Path(args.index)
    if not index_path.exists():
        log.error("index not found: %s", index_path)
        return 2

    out_run = Path(args.out_run) if args.out_run else Path("runs") / f"image_evidence_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    if out_run.exists() and not args.resume:
        _safe_rmtree(out_run)
    out_run.mkdir(parents=True, exist_ok=True)

    sample_copy = out_run / "sample_index.jsonl"
    _safe_unlink(sample_copy)
    shutil.copy(index_path, sample_copy)

    zoo = _load_yaml(Path(args.zoo))
    models = zoo.get("models") or []
    model_by_id = {m.get("model_id"): m for m in models}
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    if not providers:
        log.error("no providers specified")
        return 3

    rows = _load_jsonl(index_path)
    items: List[ImageItem] = []
    for row in rows:
        path = Path(row.get("image_path", ""))
        if not path.exists():
            log.warning("missing image: %s", path)
            continue
        items.append(
            ImageItem(
                drive_id=str(row.get("drive_id") or ""),
                frame_id=str(row.get("frame_id") or ""),
                path=path,
                camera=str(row.get("camera") or "image_00"),
                scene_profile=str(row.get("scene_profile") or "car"),
            )
        )
    if not items:
        log.error("no valid frames found in index")
        return 4

    by_drive: Dict[str, List[ImageItem]] = {}
    for item in items:
        by_drive.setdefault(item.drive_id, []).append(item)
    for drive_id in by_drive:
        by_drive[drive_id] = sorted(by_drive[drive_id], key=lambda r: r.frame_id)
        if args.max_frames and args.max_frames > 0:
            by_drive[drive_id] = by_drive[drive_id][: args.max_frames]

    device, has_cuda = _device_auto()
    seg_schema = _load_yaml(Path(args.seg_schema))

    strict_backend = _env_flag("STRICT_BACKEND", "0")
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(Path("cache") / "hf")
    if not os.environ.get("HF_HUB_CACHE"):
        os.environ["HF_HUB_CACHE"] = str(Path("cache") / "hf" / "hub")
    if not os.environ.get("TRANSFORMERS_CACHE"):
        os.environ["TRANSFORMERS_CACHE"] = str(Path("cache") / "hf" / "transformers")

    run_manifest = {
        "run_id": out_run.name,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "providers": providers,
        "sample_index": str(sample_copy),
        "config_zoo": str(Path(args.zoo)),
        "seg_schema": str(Path(args.seg_schema)),
        "feature_schema": str(Path(args.feature_schema)),
        "device": device,
        "cuda_available": has_cuda,
        "git_commit": _git_commit(),
        "strict_backend": strict_backend,
        "providers_backend": {},
    }
    (out_run / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    for provider_id in providers:
        model_cfg = model_by_id.get(provider_id)
        if not model_cfg:
            log.error("model_id not found in zoo: %s", provider_id)
            return 5
        family = str(model_cfg.get("family", "")).lower()
        provider_cls = get_provider(family)
        if provider_cls is None:
            log.error("provider family not registered: %s", family)
            return 6

        log.info("provider=%s family=%s", provider_id, family)
        provider_out = out_run / "model_outputs" / provider_id
        debug_out = out_run / "debug" / provider_id
        evidence_path = out_run / "evidence" / f"{provider_id}.jsonl"
        if provider_out.exists() and not args.resume:
            _safe_rmtree(provider_out)
        provider_out.mkdir(parents=True, exist_ok=True)
        debug_out.mkdir(parents=True, exist_ok=True)

        ctx = ProviderContext(model_cfg=model_cfg, seg_schema=seg_schema, device=device)
        provider = provider_cls(ctx)
        try:
            provider.load()
        except Exception as exc:
            log.error("provider load failed: %s (%s)", provider_id, exc)
            provider_backend = {
                "backend_status": "unavailable",
                "fallback_used": False,
                "fallback_from": provider_id,
                "fallback_to": "",
                "backend_reason": "missing_dependency" if "No module" in str(exc) else "runtime_error",
            }
            run_manifest["providers_backend"][provider_id] = provider_backend
            (out_run / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
            return 12

        records: List[dict] = []
        counts: Dict[str, int] = {}
        per_drive_reports: Dict[str, dict] = {}
        provider_backend = {}
        for drive_id, frames in by_drive.items():
            drive_dir = provider_out / drive_id
            drive_dir.mkdir(parents=True, exist_ok=True)
            drive_debug = debug_out / drive_id
            drive_debug.mkdir(parents=True, exist_ok=True)
            if Image is not None and frames:
                img = Image.open(frames[0].path)
                _write_meta(drive_dir / "meta.json", img.size)
            report = provider.infer(frames, drive_dir, debug_dir=drive_debug)
            per_drive_reports[drive_id] = report
            for k, v in (report.get("counts") or {}).items():
                counts[k] = counts.get(k, 0) + int(v)
            provider_backend = _backend_defaults(provider_id, report)
            if provider_backend["backend_status"] == "unavailable":
                log.error("provider unavailable: %s (%s)", provider_id, provider_backend.get("backend_reason"))
                return 10
            if strict_backend and provider_backend["fallback_used"]:
                log.error("STRICT_BACKEND=1 fallback used by %s -> %s", provider_id, provider_backend.get("fallback_to"))
                (out_run / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
                return 11

            for item in frames:
                seg_path = drive_dir / "seg_masks" / f"{item.path.stem}_seg.png"
                det_path = drive_dir / "det_outputs" / f"{item.path.stem}_det.json"
                debug_path = drive_debug / f"{item.path.stem}_overlay.png"
                seg_record = {
                    "kind": "seg_map",
                    "provider_id": provider_id,
                    "model_id": provider_id,
                    "model_version": model_cfg.get("model_version") or "v1",
                    "ckpt_hash": model_cfg.get("ckpt_hash") or (model_cfg.get("download") or {}).get("sam2_checkpoint"),
                    "scene_profile": item.scene_profile,
                    "drive_id": item.drive_id,
                    "frame_id": item.frame_id,
                    "image_path": str(item.path),
                    "mask": {"format": "class_id_png", "path": str(seg_path)},
                    "geometry_frame": "image_px",
                    "debug_assets_path": str(debug_path) if debug_path.exists() else "",
                    "config_path": str(Path(args.zoo)),
                    "git_commit": run_manifest["git_commit"],
                    "generated_at": run_manifest["generated_at"],
                    **provider_backend,
                }
                records.append(seg_record)

                for det in _collect_dets(det_path):
                    det_record = {
                        "kind": "det",
                        "provider_id": provider_id,
                        "model_id": provider_id,
                        "model_version": model_cfg.get("model_version") or "v1",
                        "ckpt_hash": model_cfg.get("ckpt_hash") or (model_cfg.get("download") or {}).get("sam2_checkpoint"),
                        "scene_profile": item.scene_profile,
                        "drive_id": item.drive_id,
                        "frame_id": item.frame_id,
                        "image_path": str(item.path),
                        "prompt": det.get("class"),
                        "prompt_type": "text",
                        "score": det.get("conf"),
                        "bbox": det.get("bbox"),
                        "geometry_frame": "image_px",
                        "debug_assets_path": str(debug_path) if debug_path.exists() else "",
                        "config_path": str(Path(args.zoo)),
                        "git_commit": run_manifest["git_commit"],
                        "generated_at": run_manifest["generated_at"],
                        **provider_backend,
                    }
                    records.append(det_record)

        run_manifest["providers_backend"][provider_id] = provider_backend

        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        _write_evidence_records(evidence_path, records)

        infer_report = {
            "provider_id": provider_id,
            "family": family,
            "device": device,
            "counts": counts,
            "frames": {drive: len(frames) for drive, frames in by_drive.items()},
            "provider_report": per_drive_reports,
            "backend": provider_backend,
            "generated_at": run_manifest["generated_at"],
        }
        report_path = out_run / "reports" / f"{provider_id}_infer_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _safe_unlink(report_path)
        report_path.write_text(json.dumps(infer_report, indent=2), encoding="utf-8")

        if args.build_feature_store:
            fs_root = out_run / f"feature_store_{provider_id}"
            if fs_root.exists() and not args.resume:
                _safe_rmtree(fs_root)
            fs_root.mkdir(parents=True, exist_ok=True)
            for drive_id, frames in by_drive.items():
                code = _build_feature_store(
                    drive_id,
                    provider_out / drive_id,
                    fs_root,
                    Path(args.seg_schema),
                    Path(args.feature_schema),
                    args.max_frames,
                )
                if code != 0:
                    log.warning("feature_store failed: provider=%s drive=%s", provider_id, drive_id)

        if args.build_feature_store_map:
            fs_map_root = out_run / f"feature_store_map_{provider_id}"
            if fs_map_root.exists() and not args.resume:
                _safe_rmtree(fs_map_root)
            fs_map_root.mkdir(parents=True, exist_ok=True)
            for drive_id, frames in by_drive.items():
                camera = frames[0].camera if frames else "image_00"
                code = _build_feature_store_map(
                    drive_id,
                    out_run / f"feature_store_{provider_id}" / "feature_store",
                    fs_map_root,
                    camera,
                    args.max_frames,
                )
                if code != 0:
                    log.warning("feature_store_map failed: provider=%s drive=%s", provider_id, drive_id)

    (out_run / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    log.info("run completed: %s", out_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

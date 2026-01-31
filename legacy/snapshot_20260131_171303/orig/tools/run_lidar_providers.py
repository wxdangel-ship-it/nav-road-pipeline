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
from typing import Dict, List

import geopandas as gpd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.evidence.lidar_provider_registry import LidarProviderContext, get_lidar_provider  # noqa: E402
from pipeline.evidence import lidar_providers  # noqa: F401,E402 - register providers


@dataclass
class LidarItem:
    drive_id: str
    frame_id: str
    lidar_path: str


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("run_lidar_providers")


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


def _write_features_gpkg(path: Path, features: List[dict]) -> None:
    if path.exists():
        path.unlink()
    by_class: Dict[str, list] = {}
    for feat in features:
        cls = feat.get("class", "unknown")
        by_class.setdefault(cls, []).append(feat)
    for cls, feats in by_class.items():
        gdf = gpd.GeoDataFrame(
            [f["properties"] for f in feats],
            geometry=[f["geometry"] for f in feats],
            crs="EPSG:32632",
        )
        gdf.to_file(path, layer=cls, driver="GPKG")


def _write_frame_store(store_dir: Path, frame_id: str, features: List[dict]) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    out_path = store_dir / f"{frame_id}.jsonl"
    _safe_unlink(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for feat in features:
            record = {
                "properties": feat["properties"],
                "class": feat.get("class", "unknown"),
            }
            f.write(json.dumps(record) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--providers", default="pc_simple_ground_v1")
    ap.add_argument("--out-run", default="")
    ap.add_argument("--zoo", default="configs/lidar_model_zoo.yaml")
    ap.add_argument("--resume", type=int, default=0)
    args = ap.parse_args()

    log = _setup_logger()
    index_path = Path(args.index)
    if not index_path.exists():
        log.error("index not found: %s", index_path)
        return 2

    data_root = Path(os.environ.get("POC_DATA_ROOT", ""))
    if not data_root.exists():
        log.error("POC_DATA_ROOT not set or invalid.")
        return 3

    out_run = Path(args.out_run) if args.out_run else Path("runs") / f"lidar_evidence_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    if out_run.exists() and not args.resume:
        _safe_rmtree(out_run)
    out_run.mkdir(parents=True, exist_ok=True)

    zoo = _load_yaml(Path(args.zoo))
    models = zoo.get("models") or []
    model_by_id = {m.get("model_id"): m for m in models}
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    if not providers:
        log.error("no providers specified")
        return 4

    rows = _load_jsonl(index_path)
    items: List[LidarItem] = []
    for row in rows:
        drive_id = str(row.get("drive_id") or "")
        frame_id = str(row.get("frame_id") or "")
        lidar_path = str(row.get("lidar_path") or "")
        if not drive_id or not frame_id:
            continue
        items.append(LidarItem(drive_id=drive_id, frame_id=frame_id, lidar_path=lidar_path))
    if not items:
        log.error("no valid frames found in index")
        return 5

    by_drive: Dict[str, List[LidarItem]] = {}
    for item in items:
        by_drive.setdefault(item.drive_id, []).append(item)
    for drive_id in by_drive:
        by_drive[drive_id] = sorted(by_drive[drive_id], key=lambda r: r.frame_id)

    run_manifest = {
        "run_id": out_run.name,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "providers": providers,
        "sample_index": str(index_path),
        "config_zoo": str(Path(args.zoo)),
        "git_commit": _git_commit(),
        "providers_backend": {},
    }
    (out_run / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    for provider_id in providers:
        model_cfg = model_by_id.get(provider_id)
        if not model_cfg:
            log.error("model_id not found in zoo: %s", provider_id)
            return 6
        family = str(model_cfg.get("family", "")).lower()
        provider_cls = get_lidar_provider(family)
        if provider_cls is None:
            log.error("provider family not registered: %s", family)
            return 7

        log.info("provider=%s family=%s", provider_id, family)
        provider_out = out_run / "model_outputs" / provider_id
        debug_out = out_run / "debug" / provider_id
        if provider_out.exists() and not args.resume:
            _safe_rmtree(provider_out)
        provider_out.mkdir(parents=True, exist_ok=True)
        debug_out.mkdir(parents=True, exist_ok=True)

        ctx = LidarProviderContext(model_cfg=model_cfg, device=str(model_cfg.get("device", "cpu")), data_root=str(data_root))
        strict_backend = str(os.environ.get("STRICT_BACKEND", "0")).strip().lower() in {"1", "true", "yes", "y"}
        provider = provider_cls(ctx)
        try:
            provider.load()
        except Exception as exc:
            log.error("provider load failed: %s (%s)", provider_id, exc)
            run_manifest["providers_backend"][provider_id] = {
                "backend_status": "unavailable",
                "fallback_used": False,
                "backend_reason": str(exc),
            }
            (out_run / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
            return 8

        per_drive_counts: Dict[str, dict] = {}
        errors_summary = []

        for drive_id, frames in by_drive.items():
            drive_dir = provider_out / drive_id
            drive_dir.mkdir(parents=True, exist_ok=True)
            drive_debug = debug_out / drive_id
            drive_debug.mkdir(parents=True, exist_ok=True)

            report = provider.infer(frames, drive_dir, debug_dir=drive_debug)
            backend_status = report.get("backend_status", "real")
            fallback_used = bool(report.get("fallback_used", False))
            if strict_backend and (backend_status != "real" or fallback_used):
                log.error("STRICT_BACKEND=1 and provider not real: %s", provider_id)
                return 9
            features = report.get("features") or []
            errors = report.get("errors") or []
            counts = report.get("counts") or {}
            per_drive_counts[drive_id] = counts

            if errors:
                errors_summary.extend(errors)
                err_path = out_run / "lidar_map" / provider_id / drive_id / "errors.txt"
                err_path.parent.mkdir(parents=True, exist_ok=True)
                err_path.write_text("\n".join(errors), encoding="utf-8")

            store_dir = out_run / f"feature_store_lidar_{provider_id}" / drive_id
            for item in frames:
                frame_features = [f for f in features if f["properties"].get("frame_id") == item.frame_id]
                if frame_features:
                    _write_frame_store(store_dir, item.frame_id, frame_features)

            map_dir = out_run / f"lidar_map_{provider_id}" / drive_id
            map_dir.mkdir(parents=True, exist_ok=True)
            map_path = map_dir / "lidar_evidence_utm32.gpkg"
            _write_features_gpkg(map_path, features)

        run_manifest["providers_backend"][provider_id] = {
            "backend_status": report.get("backend_status", "real"),
            "fallback_used": bool(report.get("fallback_used", False)),
            "backend_reason": report.get("backend_reason", ""),
        }
        (out_run / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

        report_lines = [
            "# LiDAR Evidence Report",
            "",
            f"- provider: {provider_id}",
            f"- run_dir: {out_run}",
            "",
            "## Per-Drive Counts",
        ]
        for drive_id, counts in per_drive_counts.items():
            report_lines.append(f"- {drive_id}: {counts}")
        report_lines.append("")
        report_lines.append("## Errors")
        if errors_summary:
            report_lines.append(f"- total_errors: {len(errors_summary)}")
        else:
            report_lines.append("- none")
        (out_run / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    log.info("run completed: %s", out_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

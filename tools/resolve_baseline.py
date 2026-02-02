from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifact_location(baseline_dir: Path) -> Dict:
    path = baseline_dir / "artifact_location.json"
    if path.exists():
        return _load_json(path)
    return {}


def load_local_paths() -> Dict:
    path = REPO_ROOT / "configs" / "local_paths.json"
    if path.exists():
        return _load_json(path)
    return {}


def resolve_artifacts_base(baseline_dir: Path) -> Path | None:
    info = load_artifact_location(baseline_dir)
    base = info.get("artifacts_base_abs")
    if base:
        return Path(str(base))
    local_paths = load_local_paths()
    artifacts_root = local_paths.get("ARTIFACTS_ROOT")
    if artifacts_root:
        return Path(str(artifacts_root))
    return None


def load_large_files_manifest(baseline_dir: Path) -> Tuple[List[Dict], str]:
    primary = baseline_dir / "outputs" / "large_files_manifest.json"
    if primary.exists():
        return _load_json(primary), "outputs/large_files_manifest.json"
    legacy = baseline_dir / "manifests" / "laz_parts_manifest.json"
    if legacy.exists():
        return _load_json(legacy), "manifests/laz_parts_manifest.json"
    raise FileNotFoundError("baseline_large_files_manifest_missing")


def resolve_laz_paths(baseline_dir: Path) -> List[Path]:
    items, _ = load_large_files_manifest(baseline_dir)
    base = resolve_artifacts_base(baseline_dir)
    out: List[Path] = []
    for item in items:
        rel_path = item.get("rel_path") or ""
        abs_path = item.get("path") or ""
        if rel_path and base:
            out.append(Path(os.path.normpath(str(base / rel_path))))
        elif abs_path:
            out.append(Path(os.path.normpath(str(abs_path))))
        else:
            raise ValueError("manifest_entry_missing_path")
    return out

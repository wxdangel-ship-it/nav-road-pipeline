from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("setup_gdino_weights")


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        import shutil

        shutil.rmtree(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", default="configs/image_model_zoo.yaml")
    ap.add_argument("--model-id", default="gdino_sam2_v1")
    ap.add_argument("--clean", type=int, default=0)
    args = ap.parse_args()

    log = _setup_logger()
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(Path("cache") / "hf")
    if not os.environ.get("HF_HUB_CACHE"):
        os.environ["HF_HUB_CACHE"] = str(Path("cache") / "hf" / "hub")
    zoo = _load_yaml(Path(args.zoo))
    models = zoo.get("models") or []
    model_cfg = next((m for m in models if m.get("model_id") == args.model_id), None)
    if not model_cfg:
        log.error("model_id not found in zoo: %s", args.model_id)
        return 2

    repo_id = model_cfg.get("dino_repo_id") or (model_cfg.get("download") or {}).get("repo")
    revision = model_cfg.get("dino_revision") or "main"
    filename = model_cfg.get("dino_filename")
    local_dir = model_cfg.get("dino_weights_path")
    if not repo_id or not local_dir:
        log.error("dino_repo_id or dino_weights_path not configured")
        return 3

    local_dir = Path(local_dir)
    if args.clean:
        _safe_rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        log.error("missing huggingface_hub: %s", exc)
        return 4

    log.info("downloading grounding dino snapshot: %s (rev=%s)", repo_id, revision)
    snapshot_download(
        repo_id=str(repo_id),
        revision=str(revision),
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )

    if filename:
        weight_path = local_dir / str(filename)
        if not weight_path.exists():
            log.warning("weights file not found: %s", weight_path)
        else:
            log.info("weights ready: %s (bytes=%d)", weight_path, weight_path.stat().st_size)
    else:
        log.info("snapshot ready: %s", local_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

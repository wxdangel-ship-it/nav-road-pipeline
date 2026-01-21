from __future__ import annotations

import argparse
import logging
import os
import urllib.request
from pathlib import Path


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("setup_sam3_weights")


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _safe_unlink(path: Path) -> None:
    if path.exists():
        path.unlink()


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _safe_unlink(dst)
    urllib.request.urlretrieve(url, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", default="configs/image_model_zoo.yaml")
    ap.add_argument("--model-id", default="sam3_v1")
    args = ap.parse_args()

    log = _setup_logger()
    zoo = _load_yaml(Path(args.zoo))
    models = zoo.get("models") or []
    model_cfg = next((m for m in models if m.get("model_id") == args.model_id), None)
    if not model_cfg:
        log.error("model_id not found in zoo: %s", args.model_id)
        return 2

    weights_path = model_cfg.get("weights_path")
    weights_url = model_cfg.get("weights_url")
    if not weights_path:
        log.error("weights_path not configured in model zoo")
        return 3
    if not weights_url:
        log.error("weights_url not configured; please set a download URL")
        return 4

    dst = Path(weights_path)
    log.info("downloading sam3 weights to %s", dst)
    try:
        _download(str(weights_url), dst)
    except Exception as exc:
        log.error("download failed: %s", exc)
        return 5

    log.info("weights ready: %s", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import logging
import os
import urllib.request
import shutil
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


def _hf_download(repo_id: str, filename: str, revision: str, token: str | None, dst: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError
        from huggingface_hub import HfFolder
    except Exception as exc:
        raise RuntimeError(f"missing huggingface_hub: {exc}") from exc

    if token is None:
        token = HfFolder.get_token()
    if not token:
        raise RuntimeError("missing_token")

    cached = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision or "main", token=token)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _safe_unlink(dst)
    shutil.copy2(cached, dst)


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
    repo_id = model_cfg.get("weights_repo_id")
    filename = model_cfg.get("weights_filename")
    revision = model_cfg.get("weights_revision") or "main"
    if not weights_path:
        log.error("weights_path not configured in model zoo")
        return 3
    dst = Path(weights_path)
    if dst.exists():
        _safe_unlink(dst)

    if repo_id and filename:
        token = os.environ.get("HF_TOKEN")
        log.info("downloading sam3 weights via huggingface_hub: %s/%s", repo_id, filename)
        try:
            _hf_download(str(repo_id), str(filename), str(revision), token, dst)
        except RuntimeError as exc:
            if str(exc) == "missing_token":
                log.error("HF_TOKEN not set and no cached login found.")
                log.error("Please set HF_TOKEN or run huggingface-cli login, then retry.")
                return 4
            log.error("huggingface_hub download failed: %s", exc)
            return 5
        log.info("weights ready: %s (bytes=%d)", dst, dst.stat().st_size)
        return 0

    if not weights_url:
        log.error("weights_url not configured and no weights_repo_id/filename provided.")
        return 4

    log.info("downloading sam3 weights via URL to %s", dst)
    try:
        _download(str(weights_url), dst)
    except Exception as exc:
        if "403" in str(exc) or "401" in str(exc):
            log.error("download failed with 401/403; ensure HF_TOKEN or huggingface-cli login.")
            return 4
        log.error("download failed: %s", exc)
        return 5

    log.info("weights ready: %s (bytes=%d)", dst, dst.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

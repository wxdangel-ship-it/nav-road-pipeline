from __future__ import annotations
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import time
from copy import deepcopy
import yaml

from pipeline._io import load_yaml, ensure_dir, new_run_id


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, allow_unicode=False, sort_keys=False)
    path.write_text(text, encoding="utf-8")


def _extract_eval_run_dir(text: str, repo: Path) -> Path | None:
    m = re.search(r"\[EVAL\]\s+DONE\s+->\s+([^\r\n(]+)", text)
    if not m:
        return None
    raw = m.group(1).strip().strip("\"'")
    p = Path(raw)
    if not p.is_absolute():
        p = repo / p
    return p


def _latest_eval_run(repo: Path) -> Path | None:
    runs_dir = repo / "runs"
    if not runs_dir.exists():
        return None
    eval_dirs = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("eval_")]
    if not eval_dirs:
        return None
    eval_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return eval_dirs[0]


def _run_eval_cmd(
    repo: Path,
    config_path: Path,
    max_frames: int,
    index_path: Path,
    data_root: str | None,
    prior_root: str | None,
) -> Path:
    cmd = [
        "cmd.exe",
        "/c",
        str(repo / "scripts" / "eval.cmd"),
        "--config",
        str(config_path),
        "--max-frames",
        str(max_frames),
        "--index",
        str(index_path),
    ]
    if data_root:
        cmd += ["--data-root", data_root]
    if prior_root:
        cmd += ["--prior-root", prior_root]

    proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    run_dir = _extract_eval_run_dir(output, repo)
    if run_dir is None:
        run_dir = _latest_eval_run(repo)
    if proc.returncode != 0 or run_dir is None:
        raise SystemExit(f"ERROR: eval.cmd failed (code={proc.returncode}). Output:\n{output}")
    return run_dir


def _read_score_terms(run_dir: Path) -> dict:
    rc = run_dir / "RunCard_Arm0.json"
    data = json.loads(rc.read_text(encoding="utf-8"))
    return data.get("score_terms", {})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=500, help="max frames for eval")
    ap.add_argument("--index", default="cache/kitti360_index_500.json", help="index cache path")
    ap.add_argument("--data-root", default="", help="KITTI-360 root (optional, default=POC_DATA_ROOT)")
    ap.add_argument("--prior-root", default="", help="prior root (optional, default=POC_PRIOR_ROOT or data-root)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    data_root = args.data_root or os.environ.get("POC_DATA_ROOT", "")
    prior_root = args.prior_root or os.environ.get("POC_PRIOR_ROOT", "")
    index_path = repo / args.index
    if not data_root:
        raise SystemExit("ERROR: --data-root or POC_DATA_ROOT is required.")
    if not index_path.exists():
        raise SystemExit(f"ERROR: index not found: {index_path}")

    cfg_active = load_yaml(repo / "configs" / "active.yaml")
    run_id = new_run_id("surrogate_check")
    run_dir = ensure_dir(repo / "runs" / run_id)

    cfg_a = deepcopy(cfg_active)
    cfg_a["config_id"] = f"{cfg_active.get('config_id')}_SUR_A"

    cfg_b = deepcopy(cfg_active)
    cfg_b["config_id"] = f"{cfg_active.get('config_id')}_SUR_B"
    modules_b = deepcopy(cfg_b.get("modules", {}))
    if "M2" in modules_b:
        m2 = modules_b["M2"]
        m2["params"] = dict(m2.get("params", {}) or {})
        m2["params"]["dummy_thr"] = 0.95
    if "M6a" in modules_b:
        m6a = modules_b["M6a"]
        m6a["impl_id"] = "m6a_stub2"
        m6a["params"] = {"smooth_lambda": 1.9, "max_shift_m": 2.9}
    cfg_b["modules"] = modules_b

    cfg_a_path = run_dir / "A.yaml"
    cfg_b_path = run_dir / "B.yaml"
    _write_yaml(cfg_a_path, cfg_a)
    _write_yaml(cfg_b_path, cfg_b)

    run_a = _run_eval_cmd(repo, cfg_a_path, args.max_frames, index_path, data_root, prior_root or data_root)
    time.sleep(1.1)
    run_b = _run_eval_cmd(repo, cfg_b_path, args.max_frames, index_path, data_root, prior_root or data_root)

    terms_a = _read_score_terms(run_a)
    terms_b = _read_score_terms(run_b)

    print("A score_terms:")
    print(json.dumps(terms_a, ensure_ascii=False, indent=2))
    print("B score_terms:")
    print(json.dumps(terms_b, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

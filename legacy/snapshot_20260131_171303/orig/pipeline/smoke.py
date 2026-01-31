from __future__ import annotations
from pathlib import Path
from pipeline._io import load_yaml, ensure_dir, new_run_id, write_text, RUNTIME_TARGET

def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_yaml(repo / "configs" / "active.yaml")
    run_id = new_run_id("smoke")
    run_dir = ensure_dir(repo / "runs" / run_id)

    text = "\n".join([
        "# StateSnapshot",
        "",
        f"- run_id: {run_id}",
        f"- runtime_target: {RUNTIME_TARGET}",
        f"- config_id: {cfg.get('config_id')}",
        "",
        "Smoke PASS",
        ""
    ])
    write_text(run_dir / "StateSnapshot.md", text)
    print(f"[SMOKE] OK -> {run_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

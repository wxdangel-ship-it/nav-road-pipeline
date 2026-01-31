import argparse
import importlib
import json
import os
from datetime import datetime
from pathlib import Path

from core.runmeta import write_runmeta
from core.reporting import write_metrics, write_skill_report


def _new_run_id(prefix: str = "smoke") -> str:
    return f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"


def run_skill(skill_id: str, run_dir: Path) -> None:
    """按约定加载 skills/<skill_id>/run.py 并执行 main(run_dir=...)。"""
    mod = importlib.import_module(f"skills.{skill_id}.run")
    if not hasattr(mod, "main"):
        raise RuntimeError(f"{skill_id} 缺少 main(run_dir=...) 入口")
    mod.main(run_dir=run_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", default="S0_env_doctor", help="要运行的 skill_id（默认 S0_env_doctor）")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = repo_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    run_id = _new_run_id("smoke")
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) 执行 Skill
    run_skill(args.skill, run_dir)

    # 2) 生成 RunMeta（最小输入/参数示例）
    params = {"skill": args.skill}
    inputs = {"note": "smoke has no external inputs"}
    write_runmeta(run_dir, repo_root, run_id, params=params, inputs=inputs)

    # 3) 写一个总的 smoke report（可选）
    write_metrics(run_dir, {"smoke": True, "skill": args.skill})
    write_skill_report(run_dir, [
        "# Smoke Report",
        "",
        f"- run_id: `{run_id}`",
        f"- skill: `{args.skill}`",
        "",
        "产物应至少包含：`metrics.json`、`SkillReport.md`、`debug_layers.gpkg`、`RunMeta.json`。",
    ])

    print(f"[OK] smoke finished: {run_dir}")


if __name__ == "__main__":
    main()

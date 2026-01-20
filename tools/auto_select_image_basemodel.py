from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _run(cmd: List[str], env: Optional[dict] = None) -> tuple[int, str]:
    if cmd and cmd[0].lower().endswith(".cmd"):
        cmd = ["cmd", "/c"] + cmd
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr)


def _newest_run(prefix: str) -> Optional[Path]:
    runs = sorted(Path("runs").glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--candidates", default="")
    ap.add_argument("--out-run", default="")
    ap.add_argument("--zoo", default="configs/image_model_zoo.yaml")
    args = ap.parse_args()

    out_run = Path(args.out_run) if args.out_run else Path("runs") / f"basemodel_select_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    out_run.mkdir(parents=True, exist_ok=True)

    zoo = _load_yaml(Path(args.zoo))
    models = zoo.get("models") or []
    if args.candidates:
        wanted = {c.strip() for c in args.candidates.split(",") if c.strip()}
        models = [m for m in models if m.get("model_id") in wanted]
    if not models:
        raise SystemExit("ERROR: no candidates found")

    report_rows = []
    for model in models:
        model_id = model.get("model_id")
        print(f"[SELECT] candidate: {model_id}")

        cmd = [
            sys.executable,
            "tools/run_image_basemodel.py",
            "--drive",
            args.drive,
            "--max-frames",
            str(args.max_frames),
            "--model-id",
            model_id,
            "--out-run",
            str(out_run),
            "--seg-schema",
            "configs/seg_schema.yaml",
        ]
        code, out = _run(cmd, env=os.environ.copy())
        infer_report = _read_json(out_run / "infer_report.json")
        status = "ok" if code == 0 else "fail"

        model_out_dir = Path(out_run) / "model_outputs" / model_id / args.drive
        feat_run = out_run / f"feature_store_{model_id}"
        feat_run.mkdir(parents=True, exist_ok=True)
        feat_cmd = [
            "scripts/image_features.cmd",
            "--drive",
            args.drive,
            "--max-frames",
            str(args.max_frames),
            "--model-out-dir",
            str(model_out_dir),
            "--out-run-dir",
            str(feat_run),
            "--seg-schema",
            "configs/seg_schema.yaml",
            "--feature-schema",
            "configs/feature_schema.yaml",
        ]
        feat_code, feat_out = _run(feat_cmd, env=os.environ.copy())
        feat_index = _read_json(feat_run / "feature_store" / "index.json")

        env = os.environ.copy()
        env["FEATURE_STORE_DIR"] = str(feat_run / "feature_store")
        center_cmd = [
            "scripts/centerlines_v2.cmd",
            "--drive",
            args.drive,
            "--max-frames",
            str(args.max_frames),
            "--debug-dividers",
            "true",
        ]
        before = _newest_run("geom")
        center_code, center_out = _run(center_cmd, env=env)
        after = _newest_run("geom")
        geom_run = after if after and after != before else after
        qc = _read_json(geom_run / "outputs" / "qc.json") if geom_run else {}

        # intersections v2 (markings)
        temp_index = out_run / f"postopt_index_{model_id}.jsonl"
        temp_index.write_text(
            json.dumps(
                {
                    "drive": args.drive,
                    "drive_id": args.drive,
                    "tile_id": args.drive,
                    "stage": "full",
                    "status": "PASS",
                    "outputs_dir": str(geom_run / "outputs") if geom_run else "",
                    "candidate_id": "auto",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        temp_cfg = out_run / f"intersections_v2_{model_id}.yaml"
        cfg = _load_yaml(Path("configs/intersections_v2.yaml"))
        cfg.setdefault("refine", {})
        cfg["refine"]["markings_enabled"] = True
        cfg["refine"]["markings_feature_store_dir"] = str(feat_run / "feature_store")
        import yaml

        temp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        inter_cmd = [
            "scripts/intersections_v2.cmd",
            "--index",
            str(temp_index),
            "--stage",
            "full",
            "--config",
            str(temp_cfg),
            "--out-dir",
            str(out_run / f"intersections_{model_id}"),
            "--resume",
        ]
        inter_code, inter_out = _run(inter_cmd, env=os.environ.copy())
        report_path = out_run / f"intersections_{model_id}" / "full_report_per_drive.json"
        report_rows = _read_json(report_path) if report_path.exists() else []
        hit_rate = None
        if report_rows:
            hit_rate = report_rows[0].get("refine_markings_hit_rate")

        row = {
            "model_id": model_id,
            "status": status,
            "infer_report": infer_report,
            "feature_store_counts": feat_index.get("counts"),
            "feature_store_frames": feat_index.get("frames_with_class"),
            "divider_found": qc.get("divider_found"),
            "split_success": qc.get("split_success"),
            "dual_ratio": qc.get("dual_ratio"),
            "dual_len_m": qc.get("centerlines_dual_len_m"),
            "refine_markings_hit_rate": hit_rate,
            "geom_run": str(geom_run) if geom_run else None,
            "stdout": out,
            "feature_store_log": feat_out,
            "centerlines_log": center_out,
            "intersections_log": inter_out,
        }
        report_rows.append(row)

    report_path = out_run / "report.json"
    report_path.write_text(json.dumps(report_rows, indent=2), encoding="utf-8")

    lines = ["# BaseModel Selection Report", ""]
    for row in report_rows:
        lines.extend(
            [
                f"## {row['model_id']}",
                f"- status: {row['status']}",
                f"- divider_found: {row.get('divider_found')}",
                f"- split_success: {row.get('split_success')}",
                f"- dual_ratio: {row.get('dual_ratio')}",
                f"- dual_len_m: {row.get('dual_len_m')}",
                f"- refine_markings_hit_rate: {row.get('refine_markings_hit_rate')}",
                f"- feature_store_counts: {row.get('feature_store_counts')}",
                f"- feature_store_frames: {row.get('feature_store_frames')}",
                "",
            ]
        )
    (out_run / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("[SELECT] report:", report_path)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())

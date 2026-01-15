from __future__ import annotations
from pathlib import Path
import argparse
import json
from pipeline._io import load_yaml, ensure_dir, new_run_id, RUNTIME_TARGET
from pipeline._report import write_run_card, write_sync_pack
from pipeline.adapters.kitti360_adapter import summarize_dataset, infer_priors


def _gate(metrics: dict, gates: dict) -> tuple[bool, str]:
    g = gates.get("gate", {})
    if metrics["C"] < g.get("C_min", 0.8):
        return False, "C below threshold"
    if metrics["B_roughness"] > g.get("B_max_roughness", 0.35):
        return False, "B roughness above threshold"
    if metrics["A_dangling_per_km"] > g.get("A_max_dangling_per_km", 5.0):
        return False, "A dangling above threshold"
    return True, "PASS"


def _calc_metrics_from_data(ds: dict, use_osm: bool, use_sat: bool, priors: dict) -> dict:
    # 当前阶段：点云+视频优先，因此 C 先用 image_coverage（pose 作为单独指标展示）
    image_cov = float(ds.get("image_coverage", 0.0))
    pose_cov = float(ds.get("pose_coverage", 0.0))

    osm_avail = priors.get("osm_layers") is not None
    sat_avail = priors.get("sat_tiles") is not None

    # 先验只做小幅“拉一把”，且如果目录不存在则视为不可用
    bonus = 0.0
    if use_osm and osm_avail:
        bonus += 0.005
    if use_sat and sat_avail:
        bonus += 0.005

    C = min(0.99, image_cov + bonus)

    prior_used = "NONE"
    if use_osm and use_sat:
        prior_used = "BOTH"
    elif use_osm:
        prior_used = "OSM"
    elif use_sat:
        prior_used = "SAT"

    # 先验冲突率：这里先用“启用且可用”给一个小值，后续接入真实对齐/冲突计算再替换
    conflict = 0.02
    if (use_osm and osm_avail) or (use_sat and sat_avail):
        conflict = 0.03

    # B/A 先给可解释的占位：C越高越好；冲突越高越差
    B = 0.22 + 0.20 * (1.0 - C) + 0.50 * conflict
    # A 先用 pose 覆盖率反推：pose 越少，拓扑更可能出问题（占位）
    A = 1.5 + 8.0 * (1.0 - pose_cov) + 10.0 * conflict

    prior_conf = 0.0
    align_res = 999.0
    if prior_used != "NONE":
        # 有任何先验启用时，如果对应目录存在，给一个默认可信度
        if (use_osm and osm_avail) or (use_sat and sat_avail):
            prior_conf = 0.8
            align_res = 1.0

    return {
        "C": round(C, 4),
        "B_roughness": round(B, 4),
        "A_dangling_per_km": round(A, 3),
        "prior_used": prior_used,
        "prior_confidence_p50": round(prior_conf, 3),
        "alignment_residual_p50": round(align_res, 3),
        "conflict_rate": round(conflict, 3),
        "image_coverage": round(image_cov, 4),
        "pose_coverage": round(pose_cov, 4),
        "prior_osm_available": bool(osm_avail),
        "prior_sat_available": bool(sat_avail),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/active.yaml", help="config yaml path")
    ap.add_argument("--data-root", default="", help="POC data root (KITTI-360 root)")
    ap.add_argument("--prior-root", default="", help="prior root (optional, default=data-root)")
    ap.add_argument("--drives", default="", help="comma separated drive list (optional)")
    ap.add_argument("--max-frames", type=int, default=0, help="limit frames per drive for speed (0=all)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = load_yaml(repo / args.config)
    arms = load_yaml(repo / "configs" / "arms.yaml").get("arms", {})
    gates = load_yaml(repo / "configs" / "gates.yaml")

    run_id = new_run_id("eval")
    run_dir = ensure_dir(repo / "runs" / run_id)

    # 写 StateSnapshot（包含 config 与运行参数）
    snap = {
        "run_id": run_id,
        "runtime_target": RUNTIME_TARGET,
        "config_id": cfg.get("config_id"),
        "config_path": args.config,
        "data_root": args.data_root,
        "prior_root": args.prior_root,
        "drives": args.drives,
        "max_frames": args.max_frames,
        "modules": cfg.get("modules", {}),
    }
    (run_dir / "StateSnapshot.md").write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")

    # --------- 数据模式：必须提供 data_root ----------
    if not args.data_root:
        raise SystemExit("ERROR: --data-root is required now (e.g. --data-root E:\\\\KITTI360\\\\KITTI-360)")

    data_root = Path(args.data_root)
    prior_root = Path(args.prior_root) if args.prior_root else data_root
    if not data_root.exists():
        raise SystemExit(f"ERROR: data_root not exists: {data_root}")

    drive_list = [x.strip() for x in args.drives.split(",") if x.strip()] if args.drives else None
    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else None

    ds = summarize_dataset(data_root, drives=drive_list, max_frames=max_frames)
    priors = infer_priors(data_root, prior_root=prior_root)

    # 将先验路径写入 ds_summary 方便排查
    ds_summary = {
        "drive_count": ds.get("drive_count"),
        "total_lidar": ds.get("total_lidar"),
        "total_img_any": ds.get("total_img_any"),
        "total_pose": ds.get("total_pose"),
        "image_coverage": ds.get("image_coverage"),
        "pose_coverage": ds.get("pose_coverage"),
        "missing_pose_drives": ds.get("missing_pose_drives"),
        "osm_layers": str(priors.get("osm_layers")) if priors.get("osm_layers") else None,
        "sat_tiles": str(priors.get("sat_tiles")) if priors.get("sat_tiles") else None,
    }

    for arm_name, a in arms.items():
        use_osm = bool(a.get("use_osm", False))
        use_sat = bool(a.get("use_sat", False))
        m = _calc_metrics_from_data(ds, use_osm, use_sat, priors)
        ok, reason = _gate(m, gates)

        run_card = {
            "run_id": run_id,
            "arm": arm_name,
            "config_id": cfg.get("config_id"),
            "runtime_target": RUNTIME_TARGET,
            "gate": "PASS" if ok else "FAIL",
            "gate_reason": reason,
            "metrics": m,
            "data_summary": ds_summary,
        }
        write_run_card(run_dir / f"RunCard_{arm_name}.md", run_card)
        write_sync_pack(
            run_dir / f"SyncPack_{arm_name}.md",
            diff={"config_id": cfg.get("config_id"), "arm": arm_name, "config_path": args.config},
            evidence=run_card,
            ask="If FAIL, propose <=3 fixes. If PASS, suggest next autotune actions."
        )

    print(f"[EVAL] DONE -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

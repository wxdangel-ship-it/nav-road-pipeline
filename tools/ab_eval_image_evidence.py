from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None


def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("ab_eval_image_evidence")


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


def _latest_run(prefix: str) -> Optional[Path]:
    runs = sorted(Path("runs").glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _safe_unlink(path: Path) -> None:
    if path.exists():
        path.unlink()


def _read_mask(path: Path) -> np.ndarray:
    if Image is None:
        raise RuntimeError("PIL is required to read masks.")
    img = Image.open(path)
    return np.array(img)


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=float), q))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--index", default="")
    ap.add_argument("--providers", default="grounded_sam2_v1,gdino_sam2_v1,sam3_v1")
    ap.add_argument("--seg-schema", default="configs/seg_schema.yaml")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    log = _setup_logger()
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run("image_evidence")
    if not run_dir or not run_dir.exists():
        log.error("run dir not found")
        return 2

    index_path = Path(args.index) if args.index else run_dir / "sample_index.jsonl"
    if not index_path.exists():
        log.error("sample index not found: %s", index_path)
        return 3

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "ab_eval"
    if out_dir.exists():
        _safe_rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_schema = _load_yaml(Path(args.seg_schema))
    id_to_class = {int(k): str(v) for k, v in (seg_schema.get("id_to_class") or {}).items()}
    class_to_id = {str(v).lower(): int(k) for k, v in (seg_schema.get("id_to_class") or {}).items()}

    rows = _load_jsonl(index_path)
    by_drive: Dict[str, List[dict]] = {}
    for row in rows:
        drive_id = str(row.get("drive_id") or "")
        by_drive.setdefault(drive_id, []).append(row)
    for drive_id in by_drive:
        by_drive[drive_id] = sorted(by_drive[drive_id], key=lambda r: str(r.get("frame_id") or ""))

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    if not providers:
        log.error("no providers specified")
        return 4

    per_provider: Dict[str, dict] = {}
    for provider in providers:
        det_counts: Dict[str, int] = {}
        det_scores: Dict[str, List[float]] = {}
        mask_areas: Dict[str, List[float]] = {}
        frames_with_class: Dict[str, int] = {}

        for drive_id, frames in by_drive.items():
            base = run_dir / "model_outputs" / provider / drive_id
            for row in frames:
                frame_id = str(row.get("frame_id") or "")
                img_path = Path(row.get("image_path") or "")
                stem = img_path.stem
                det_path = base / "det_outputs" / f"{stem}_det.json"
                if det_path.exists():
                    dets = json.loads(det_path.read_text(encoding="utf-8"))
                    for det in dets:
                        cls = str(det.get("class") or "unknown").lower()
                        det_counts[cls] = det_counts.get(cls, 0) + 1
                        score = det.get("conf")
                        if score is not None:
                            det_scores.setdefault(cls, []).append(float(score))

                seg_path = base / "seg_masks" / f"{stem}_seg.png"
                if seg_path.exists():
                    mask = _read_mask(seg_path)
                    ids = np.unique(mask.astype(np.int32)).tolist()
                    for cid in ids:
                        if cid == 0:
                            continue
                        cls = id_to_class.get(int(cid), str(cid))
                        area = float(np.sum(mask == cid))
                        mask_areas.setdefault(cls, []).append(area)
                        frames_with_class[cls] = frames_with_class.get(cls, 0) + 1

        per_provider[provider] = {
            "det_counts": det_counts,
            "det_score_p50": {cls: _percentile(scores, 50) for cls, scores in det_scores.items()},
            "mask_area_p50": {cls: _percentile(areas, 50) for cls, areas in mask_areas.items()},
            "mask_area_p90": {cls: _percentile(areas, 90) for cls, areas in mask_areas.items()},
            "frames_with_class": frames_with_class,
        }

    consistency: Dict[str, dict] = {}
    disagreements: Dict[str, dict] = {}
    for i in range(len(providers)):
        for j in range(i + 1, len(providers)):
            a_id = providers[i]
            b_id = providers[j]
            key = f"{a_id}__vs__{b_id}"
            per_class_ious: Dict[str, List[float]] = {}
            worst: Dict[str, List[Tuple[float, str, str]]] = {}
            for drive_id, frames in by_drive.items():
                for row in frames:
                    img_path = Path(row.get("image_path") or "")
                    stem = img_path.stem
                    mask_a = run_dir / "model_outputs" / a_id / drive_id / "seg_masks" / f"{stem}_seg.png"
                    mask_b = run_dir / "model_outputs" / b_id / drive_id / "seg_masks" / f"{stem}_seg.png"
                    if not mask_a.exists() or not mask_b.exists():
                        continue
                    arr_a = _read_mask(mask_a)
                    arr_b = _read_mask(mask_b)
                    for cls, cid in class_to_id.items():
                        bin_a = arr_a == cid
                        bin_b = arr_b == cid
                        if not bin_a.any() and not bin_b.any():
                            continue
                        iou = _mask_iou(bin_a, bin_b)
                        per_class_ious.setdefault(cls, []).append(iou)
                        worst.setdefault(cls, []).append((iou, drive_id, stem))

            consistency[key] = {cls: _percentile(vals, 50) for cls, vals in per_class_ious.items()}
            disagreements[key] = {}
            for cls, vals in worst.items():
                vals_sorted = sorted(vals, key=lambda x: x[0])[: args.top_k]
                disagreements[key][cls] = [
                    {
                        "iou": v[0],
                        "drive_id": v[1],
                        "frame": v[2],
                        "debug_a": str(run_dir / "debug" / a_id / v[1] / f"{v[2]}_overlay.png"),
                        "debug_b": str(run_dir / "debug" / b_id / v[1] / f"{v[2]}_overlay.png"),
                    }
                    for v in vals_sorted
                ]

    stability: Dict[str, dict] = {}
    for provider in providers:
        per_class: Dict[str, List[float]] = {}
        for drive_id, frames in by_drive.items():
            ordered = frames
            for idx in range(1, len(ordered)):
                prev = ordered[idx - 1]
                curr = ordered[idx]
                stem_prev = Path(prev.get("image_path") or "").stem
                stem_curr = Path(curr.get("image_path") or "").stem
                mask_prev = run_dir / "model_outputs" / provider / drive_id / "seg_masks" / f"{stem_prev}_seg.png"
                mask_curr = run_dir / "model_outputs" / provider / drive_id / "seg_masks" / f"{stem_curr}_seg.png"
                if not mask_prev.exists() or not mask_curr.exists():
                    continue
                arr_prev = _read_mask(mask_prev)
                arr_curr = _read_mask(mask_curr)
                for cls, cid in class_to_id.items():
                    bin_prev = arr_prev == cid
                    bin_curr = arr_curr == cid
                    if not bin_prev.any() and not bin_curr.any():
                        continue
                    per_class.setdefault(cls, []).append(_mask_iou(bin_prev, bin_curr))
        stability[provider] = {cls: _percentile(vals, 50) for cls, vals in per_class.items()}

    report = {
        "run_dir": str(run_dir),
        "sample_index": str(index_path),
        "providers": providers,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "per_provider": per_provider,
        "consistency_p50": consistency,
        "stability_p50": stability,
        "disagreements": disagreements,
    }
    report_json = out_dir / "report.json"
    _safe_unlink(report_json)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Image Evidence AB Report",
        "",
        f"- run_dir: {run_dir}",
        f"- sample_index: {index_path}",
        f"- providers: {', '.join(providers)}",
        f"- generated_at: {report['generated_at']}",
        "",
        "## Per-Provider Summary",
    ]
    for provider in providers:
        stats = per_provider.get(provider, {})
        lines.extend(
            [
                f"### {provider}",
                f"- det_counts: {stats.get('det_counts')}",
                f"- det_score_p50: {stats.get('det_score_p50')}",
                f"- mask_area_p50: {stats.get('mask_area_p50')}",
                f"- mask_area_p90: {stats.get('mask_area_p90')}",
                f"- frames_with_class: {stats.get('frames_with_class')}",
                "",
            ]
        )
    lines.append("## Cross-Model Consistency (IoU p50)")
    for key, vals in consistency.items():
        lines.append(f"- {key}: {vals}")
    lines.append("")
    lines.append("## Stability (IoU p50)")
    for provider, vals in stability.items():
        lines.append(f"- {provider}: {vals}")
    lines.append("")
    lines.append("## Disagreements (Top-K Lowest IoU)")
    for key, per_cls in disagreements.items():
        lines.append(f"### {key}")
        for cls, items in per_cls.items():
            lines.append(f"- {cls}: {items}")
        lines.append("")
    report_md = out_dir / "report.md"
    _safe_unlink(report_md)
    report_md.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote report: %s", report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

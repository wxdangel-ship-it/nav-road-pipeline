from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from pipeline.lidar_semantic.export_pointcloud import write_las
from scripts.pipeline_common import ensure_dir, now_ts, setup_logging, write_json, write_text


LOG = logging.getLogger("vis_lidar_intensity_highlights")


def _read_las_header(path: Path) -> Tuple[dict, int]:
    import struct

    header_size = 227
    with path.open("rb") as f:
        header = f.read(header_size)
    fmt = "<4sHH16sBB32s32sHHH I I B H I 5I ddd ddd dddddd"
    values = struct.unpack(fmt, header[: struct.calcsize(fmt)])
    (
        sig,
        _file_src,
        _global_enc,
        _proj_id,
        ver_major,
        ver_minor,
        _sys_id,
        _gen_soft,
        _doy,
        _year,
        header_size,
        offset_to_points,
        _num_vlrs,
        point_format,
        point_record_length,
        point_count,
        *_num_by_return,
        sx,
        sy,
        sz,
        ox,
        oy,
        oz,
        maxx,
        minx,
        maxy,
        miny,
        maxz,
        minz,
    ) = values
    if sig != b"LASF":
        raise RuntimeError("invalid_las_signature")
    meta = {
        "version": f"{ver_major}.{ver_minor}",
        "header_size": int(header_size),
        "offset_to_points": int(offset_to_points),
        "point_format": int(point_format),
        "point_record_length": int(point_record_length),
        "point_count": int(point_count),
        "scales": (float(sx), float(sy), float(sz)),
        "offsets": (float(ox), float(oy), float(oz)),
        "bounds": (float(minx), float(miny), float(minz), float(maxx), float(maxy), float(maxz)),
    }
    return meta, header_size


def _read_las_points(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    meta, _ = _read_las_header(path)
    point_count = int(meta["point_count"])
    point_record_length = int(meta["point_record_length"])
    offset = int(meta["offset_to_points"])
    if point_record_length < 20:
        raise RuntimeError("unsupported_point_record_length")
    with path.open("rb") as f:
        f.seek(offset)
        raw = f.read(point_count * point_record_length)
    pad = point_record_length - 20
    dtype = np.dtype(
        [
            ("x", "<i4"),
            ("y", "<i4"),
            ("z", "<i4"),
            ("intensity", "<u2"),
            ("bitfield", "u1"),
            ("classification", "u1"),
            ("scan_angle", "i1"),
            ("user_data", "u1"),
            ("point_src", "<u2"),
            ("pad", f"V{pad}") if pad > 0 else ("pad", "V0"),
        ]
    )
    arr = np.frombuffer(raw, dtype=dtype, count=point_count)
    sx, sy, sz = meta["scales"]
    ox, oy, oz = meta["offsets"]
    x = arr["x"].astype(np.float64) * sx + ox
    y = arr["y"].astype(np.float64) * sy + oy
    z = arr["z"].astype(np.float64) * sz + oz
    intensity = arr["intensity"].astype(np.uint16)
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    return points, intensity, meta


def _intensity_stats(inten: np.ndarray) -> Dict[str, float]:
    if inten.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "nonzero_ratio": 0.0,
        }
    vals = inten.astype(np.float64)
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "p99": float(np.percentile(vals, 99)),
        "nonzero_ratio": float(np.mean(vals > 0.0)),
    }


def _plot_hist(intensity: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.hist(intensity.astype(np.float64), bins=256, range=(0, 65535), color="steelblue", alpha=0.85)
    ax.set_title("Intensity Histogram")
    ax.set_xlabel("intensity (uint16)")
    ax.set_ylabel("count")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_bev(points: np.ndarray, intensity: np.ndarray, out_path: Path, max_points: int = 300000) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if points.size == 0:
        return
    rng = np.random.default_rng(0)
    idx = np.arange(points.shape[0])
    if idx.size > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    pts = points[idx]
    inten = intensity[idx]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    sc = ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c=inten, cmap="viridis", alpha=0.6)
    ax.set_title("BEV Intensity")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_highlights(
    points: np.ndarray,
    intensity: np.ndarray,
    threshold: float,
    out_path: Path,
    max_points: int = 300000,
) -> Tuple[int, int]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if points.size == 0:
        return 0, 0
    rng = np.random.default_rng(0)
    idx = np.arange(points.shape[0])
    if idx.size > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    pts = points[idx]
    inten = intensity[idx]
    mask = inten >= threshold
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.scatter(pts[:, 0], pts[:, 1], s=0.25, c="lightgray", alpha=0.35)
    if np.any(mask):
        ax.scatter(pts[mask, 0], pts[mask, 1], s=0.6, c="red", alpha=0.85, label="high intensity")
        ax.legend(loc="upper right")
    ax.set_title(f"BEV Highlights (>= {threshold:.0f})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return int(mask.sum()), int(pts.shape[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input LAS/LAZ path (prefer .las)")
    ap.add_argument("--out", default="", help="Output directory (default runs/lidar_fuse_0010_f280_300_viz_<ts>)")
    ap.add_argument("--percentile", type=float, default=99.0, help="High-intensity percentile (default 99)")
    ap.add_argument("--max-points-plot", type=int, default=300000, help="Max points sampled for plots")
    ap.add_argument("--write-subset", action="store_true", help="Write high-intensity subset LAS")
    args = ap.parse_args()

    in_path = Path(args.input)
    if in_path.suffix.lower() == ".laz":
        las_path = in_path.with_suffix(".las")
        if las_path.exists():
            in_path = las_path
        else:
            raise SystemExit("input_laz_not_supported_without_las")
    if not in_path.exists():
        raise SystemExit(f"input_missing:{in_path}")

    out_dir = Path(args.out) if args.out else Path("runs") / f"lidar_fuse_0010_f280_300_viz_{now_ts()}"
    ensure_dir(out_dir)
    setup_logging(out_dir / "run.log")
    LOG.info("load_points: %s", in_path)

    points, intensity, meta = _read_las_points(in_path)
    stats = _intensity_stats(intensity)
    threshold = float(np.percentile(intensity.astype(np.float64), float(args.percentile))) if intensity.size else 0.0
    high_mask = intensity >= threshold
    high_count = int(np.sum(high_mask))

    img_dir = ensure_dir(out_dir / "images")
    _plot_hist(intensity, img_dir / "intensity_hist.png")
    _plot_bev(points, intensity, img_dir / "bev_intensity.png", max_points=int(args.max_points_plot))
    sample_high, sample_total = _plot_highlights(
        points,
        intensity,
        threshold,
        img_dir / "bev_high_intensity.png",
        max_points=int(args.max_points_plot),
    )

    subset_path = None
    if args.write_subset and high_count > 0:
        subset_dir = ensure_dir(out_dir / "subset")
        subset_path = subset_dir / "high_intensity_points_utm32.laz"
        write_las(
            subset_path,
            points[high_mask].astype(np.float32),
            intensity[high_mask].astype(np.uint16),
            np.ones((high_count,), dtype=np.uint8),
            32632,
        )

    stats_payload = {
        "input": str(in_path),
        "point_count": int(points.shape[0]),
        "intensity_stats": stats,
        "percentile": float(args.percentile),
        "threshold": float(threshold),
        "high_count": high_count,
        "high_ratio": float(high_count / max(points.shape[0], 1)),
        "sample_high": sample_high,
        "sample_total": sample_total,
        "las_meta": meta,
    }
    write_json(out_dir / "stats.json", stats_payload)

    report = [
        "# LiDAR intensity visualization",
        "",
        f"- input: {in_path}",
        f"- total_points: {int(points.shape[0])}",
        f"- percentile: {float(args.percentile):.2f}",
        f"- threshold: {float(threshold):.1f}",
        f"- high_count: {high_count}",
        f"- high_ratio: {high_count / max(points.shape[0], 1):.6f}",
        "",
        "## Outputs",
        f"- images/intensity_hist.png",
        f"- images/bev_intensity.png",
        f"- images/bev_high_intensity.png",
    ]
    if subset_path is not None:
        report.append(f"- {subset_path.relative_to(out_dir)}")
    write_text(out_dir / "report.md", "\n".join(report))
    LOG.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

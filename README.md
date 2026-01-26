# nav-road-pipeline
Windows-native road production pipeline (POC -> Prod).

## Project Goals and Three-Stage Outputs
- Primitive Evidence: source-level evidence in pixel/world space (image masks, lidar projections, SAT priors).
- World Candidates: aggregated, projected entities in map space (UTM32/WGS84) with QA metrics.
- Map Features: final curated map features (not part of this phase).

## Data Sources and Modules
- Sources: Image / LiDAR / SAT / Traj. Evidence priority is strong -> weak:
  - Strong: LiDAR + in-vehicle video
  - Medium: trajectories
  - Soft prior: OSM/SAT (never override strong evidence; record conflict_rate)
- Providers (pluggable): image/lidar/sat providers emit Primitive Evidence; see `configs/*_model_zoo.yaml`.
- Fusion/Resolve: modular stages (M0-M9) combine evidence and produce World Candidates.
- Schema contracts: `docs/evidence_schema.md`.

## Quick Start (Windows)

### Dependencies (stable, current)
- `numpy==1.26.4`
- `shapely==2.0.3`
- `pyproj==3.6.1`
- `pyyaml`
- `rasterio` (required for SAT tools and alignment diagnostics; install separately)

Install:
```cmd
scripts\setup.cmd
```

### One-click scripts (cmd)
- `scripts\setup.cmd`: venv + requirements
- `scripts\smoke.cmd`: smoke tests
- `scripts\index.cmd --max-frames 2000`: build `cache\kitti360_index.json`
- `scripts\eval.cmd --max-frames 2000`: eval entry (Arm0..3)
- `scripts\autotune.cmd`: autotune entry
- `scripts\run_image_providers.cmd`: run image providers
- `scripts\run_lidar_providers.cmd`: run lidar providers
- `scripts\run_crosswalk_strict_250_500.cmd`: strict frame-range regression (Stage2)
- `scripts\run_crosswalk_golden8_full.cmd`: Golden8 full evaluation

### Required env vars
```cmd
set POC_DATA_ROOT=E:\KITTI360\KITTI-360
set POC_PRIOR_ROOT=E:\KITTI360\_priors   (optional)
```

## QA Workflow
- raw: per-frame detections from providers before projection/gates.
- gated: candidates after projection + basic gates (IoU/support/shape).
- entities: final aggregated world candidates (clustered/refined).

Key QA outputs (per run):
- `runs\<run_id>\outputs\qa_index_wgs84.geojson` (open in QGIS)
- `runs\<run_id>\outputs\qa_images\` (raw/gated/entities overlays)
- `runs\<run_id>\outputs\crosswalk_trace.csv` (per-frame metrics)

QA index usage:
- QGIS: load `qa_index_wgs84.geojson`, filter by `qa_flag`, `reject_reasons`, `proj_method`.
- File Explorer: use `overlay_*` paths from the QA index to inspect frames.

## Output Directories and No-Commit Rules
- Runs: `runs\<run_id>\outputs\...`
- Eval protocol outputs (Arm0..3):
  - `runs\<run_id>\StateSnapshot.md`
  - `runs\<run_id>\RunCard_Arm0..3.md`
  - `runs\<run_id>\SyncPack_Arm0..3.md`
- Do not commit: `runs/`, `cache/`, `data/`, or any weights/large media.

## FAQ
- Why is IoU ~ 0?
  - Projection alignment likely failed. Check `projection_alignment_report.md` and `crosswalk_trace.csv`
    for `proj_method`, `reproj_iou_bbox`, `reproj_iou_dilated`.
- Why is `points_support = 0`?
  - Lidar projection did not land inside the mask. Verify `LIDAR_CALIB_MISMATCH`,
    `proj_in_image_ratio`, and `points_support_accum` in `crosswalk_trace.csv`.
- Why does `frame_id` look overwritten?
  - Evidence inputs may lack stable `frame_id`. Run alignment diagnostics to verify
    frame_id diversity and CRS consistency (see `docs/RUNBOOK.md`).
- How to locate `LIDAR_CALIB_MISMATCH` / `proj_fail`?
  - Search `crosswalk_trace.csv` for `drop_reason_code` and `qa_flag=proj_fail`,
    then open `proj_debug_failpack` from the same run.

## Notes
- Any file ending with `_wgs84` must be true EPSG:4326 (range-checked), otherwise rename to
  `_utm32` or reproject.
- See `docs/ARCHITECTURE.md` and `docs/RUNBOOK.md` for detailed architecture and reproducible workflows.

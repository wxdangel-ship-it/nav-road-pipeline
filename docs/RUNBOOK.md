# RUNBOOK (Reproducibility + Diagnostics)

This runbook focuses on minimal commands to reproduce key workflows on Windows.

## 0) Setup (once per machine)
```cmd
scripts\setup.cmd
```

## 1) Strict Regression (250-500, range asserted)
```cmd
scripts\run_crosswalk_strict_250_500.cmd
```
Outputs:
- `runs\<run_id>\outputs\crosswalk_stage2_report.md`
- `runs\<run_id>\outputs\frame_range_assert.txt`
- `runs\<run_id>\outputs\qa_index_wgs84.geojson`

Override range/drive (optional):
```cmd
set DRIVE_ID=2013_05_28_drive_0010_sync
set FRAME_START=250
set FRAME_END=500
set KITTI_ROOT=E:\KITTI360\KITTI-360
scripts\run_crosswalk_strict_250_500.cmd
```

## 2) Quick Regression (280-300)
```cmd
set MONITOR_START=280
set MONITOR_END=300
scripts\run_crosswalk_monitor.cmd
```
Outputs:
- `runs\<run_id>\outputs\crosswalk_monitor_report.md`
- `runs\<run_id>\outputs\qa_index_wgs84.geojson`

## 3) Fullpose Raster + Trajectory Alignment Check

### 3.1 Build fullpose raster and lidar center deltas
```cmd
.venv\Scripts\python.exe tools\analyze_lidar_traj_alignment.py ^
  --kitti-root E:\KITTI360\KITTI-360 ^
  --drive 2013_05_28_drive_0010_sync ^
  --traj-points runs\<run_id>\outputs\qa_index_wgs84.geojson ^
  --out-dir runs\<run_id>\outputs\alignment_fullpose
```

### 3.2 Alignment diagnostics (CRS + frame_id + offsets)
```cmd
.venv\Scripts\python.exe tools\check_alignment_diagnostics.py ^
  --qa-index runs\<run_id>\outputs\qa_index_wgs84.geojson ^
  --lidar-raster runs\<run_id>\outputs\alignment_fullpose\lidar_intensity_utm32_fullpose_2013_05_28_drive_0010_sync.tif ^
  --frame-evidence runs\<run_id>\outputs\frame_candidates_utm32.gpkg ^
  --lidar-evidence runs\<run_id>\outputs\crosswalk_entities_utm32.gpkg ^
  --drive 2013_05_28_drive_0010_sync ^
  --out-dir runs\<run_id>\outputs\alignment_check
```

## 4) Roundtrip v2 Report (subset stats + failpack index)
```cmd
scripts\run_roundtrip_report_v2.cmd
scripts\run_failpack_summary.cmd
```
Outputs:
- `projection_alignment_report_v2.md`
- `proj_debug_failpack\index.md`

## 5) near_final_diagnose
```cmd
scripts\run_near_final_diagnose.cmd
```
Interpretation:
- `near_final_report.md`: per-cluster drift or propagation issues
- `overview.csv`: quick scan for unstable clusters

## 6) Autotune
```cmd
scripts\run_crosswalk_autotune_0010_280_300.cmd
scripts\run_crosswalk_autotune_0010_250_500.cmd
```
Outputs: `runs\crosswalk_autotune_*\leaderboard.csv`, `autotune_report.md`, `best\outputs`

## 7) Golden8 Full Evaluation
```cmd
scripts\run_crosswalk_golden8_full.cmd
```
Outputs: `runs\crosswalk_golden8_full_*\outputs`, `runs\crosswalk_golden8_full_*\summary`

## 8) Eval Protocol (Arm0..3)
```cmd
scripts\eval.cmd --max-frames 2000
```
Outputs (per run):
- `StateSnapshot.md`
- `RunCard_Arm0..3.md`
- `SyncPack_Arm0..3.md`

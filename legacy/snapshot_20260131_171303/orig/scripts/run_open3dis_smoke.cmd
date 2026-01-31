@echo off
setlocal
cd /d %~dp0\..

if "%OPEN3DIS_SKIP_SETUP%"=="" (
  call .\scripts\setup_open3dis_env.cmd
)

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

if "%STRICT_BACKEND%"=="" (
  set "STRICT_BACKEND=1"
)

set "INDEX=runs\lidar_samples_open3dis_smoke\sample_index.jsonl"
if not exist "%INDEX%" (
  .venv\Scripts\python.exe tools\build_lidar_sample_index.py --index runs\sweep_geom_postopt_20260119_061421\postopt_index.jsonl --out "%INDEX%" --drive 2013_05_28_drive_0000_sync --frames-per-drive 30 --stride 5
)

.venv\Scripts\python.exe tools\run_lidar_providers.py --index "%INDEX%" --providers pc_open3dis_v1

endlocal

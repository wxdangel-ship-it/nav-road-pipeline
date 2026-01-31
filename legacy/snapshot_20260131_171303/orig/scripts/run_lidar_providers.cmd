@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "ARGS=%*"
set "INDEX=runs\lidar_samples_golden8\sample_index.jsonl"
if "%ARGS%"=="" (
  if not exist "%INDEX%" (
    .venv\Scripts\python.exe tools\build_lidar_sample_index.py --index runs\sweep_geom_postopt_20260119_061421\postopt_index.jsonl --out "%INDEX%" --frames-per-drive 20 --stride 5
  )
  .venv\Scripts\python.exe tools\run_lidar_providers.py --index "%INDEX%" --providers pc_simple_ground_v1
) else (
  .venv\Scripts\python.exe tools\run_lidar_providers.py %ARGS%
)

endlocal

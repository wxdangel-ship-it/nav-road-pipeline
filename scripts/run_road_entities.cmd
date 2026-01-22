@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "INDEX=runs\lidar_samples_open3dis_golden8_30\sample_index.jsonl"
if not exist "%INDEX%" (
  .venv\Scripts\python.exe tools\build_lidar_sample_index.py --index runs\sweep_geom_postopt_20260119_061421\postopt_index.jsonl --out "%INDEX%" --frames-per-drive 30 --stride 5
)

set "IMAGE_RUN=runs\image_evidence_20260122_114726"
set "IMAGE_PROVIDER=grounded_sam2_v1"
set "ROAD_ROOT=runs\full_stack_20260121_000548"
set "IMAGE_EVIDENCE_GPKG=runs\full_stack_20260121_000548\evidence_clean\evidence_clean_golden8.gpkg"

if "%*"=="" (
  .venv\Scripts\python.exe tools\build_road_entities.py --index "%INDEX%" --image-run "%IMAGE_RUN%" --image-provider "%IMAGE_PROVIDER%" --road-root "%ROAD_ROOT%" --image-evidence-gpkg "%IMAGE_EVIDENCE_GPKG%"
) else (
  .venv\Scripts\python.exe tools\build_road_entities.py %*
)

endlocal

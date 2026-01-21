@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "ARGS=%*"
if "%ARGS%"=="" (
  set "INDEX=runs\image_samples_golden8\sample_index.jsonl"
  if not exist "%INDEX%" (
    .venv\Scripts\python.exe tools\build_image_sample_index.py --index runs\sweep_geom_postopt_20260119_061421\postopt_index.jsonl --out "%INDEX%" --frames-per-drive 30
  )
  .venv\Scripts\python.exe tools\run_image_providers.py --index "%INDEX%"
) else (
  .venv\Scripts\python.exe tools\run_image_providers.py %ARGS%
)

endlocal

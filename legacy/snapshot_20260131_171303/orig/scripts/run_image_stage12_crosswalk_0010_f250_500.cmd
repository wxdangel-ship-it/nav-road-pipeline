@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "CFG=configs\image_stage12_crosswalk_0010_f250_500.yaml"

.venv\Scripts\python.exe scripts\run_image_stage12_crosswalk_0010_f250_500.py --config "%CFG%"

endlocal

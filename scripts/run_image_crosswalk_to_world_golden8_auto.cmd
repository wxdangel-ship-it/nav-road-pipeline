@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "CFG=configs\image_crosswalk_to_world_golden8_auto.yaml"

.venv\Scripts\python.exe scripts\run_image_crosswalk_to_world_golden8_auto.py --config "%CFG%"

endlocal

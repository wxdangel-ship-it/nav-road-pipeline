@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "CFG=configs\world_crosswalk_candidates_0010_f250_500.yaml"

.venv\Scripts\python.exe scripts\run_world_crosswalk_candidates_0010_f250_500.py --config "%CFG%"

endlocal

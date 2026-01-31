@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "CFG=configs\\crosswalk_stage2_autotune_280_300.yaml"
set "OUT_DIR="

.venv\Scripts\python.exe tools\\autotune_crosswalk_stage2.py --config "%CFG%"

for /f %%i in ('dir /b /ad /o-d runs\\crosswalk_autotune_0010_280_300_* 2^>nul') do (
  set "OUT_DIR=runs\\%%i"
  goto :found
)

:found
if "%OUT_DIR%"=="" (
  echo [AUTOTUNE_280_300] outputs: not found
) else (
  echo [AUTOTUNE_280_300] outputs: %OUT_DIR%
  echo [AUTOTUNE_280_300] leaderboard: %OUT_DIR%\\leaderboard.csv
  echo [AUTOTUNE_280_300] report: %OUT_DIR%\\autotune_report.md
  echo [AUTOTUNE_280_300] best: %OUT_DIR%\\best\\outputs
)

endlocal

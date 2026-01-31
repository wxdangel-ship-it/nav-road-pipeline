@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "CFG=configs\crosswalk_golden8_full.yaml"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
set "OUT_RUN=runs\crosswalk_golden8_full_%TS%"

.venv\Scripts\python.exe tools\run_crosswalk_golden8_full.py --config "%CFG%" --out-run "%OUT_RUN%"

echo [GOLDEN8] outputs: %OUT_RUN%\outputs
echo [GOLDEN8] summary: %OUT_RUN%\summary

endlocal

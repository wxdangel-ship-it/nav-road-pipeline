@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

call .\scripts\run_crosswalk_strict_250_500.cmd

set "RUN_DIR="
set "DRIVE_ID=%DRIVE_ID%"
set "FRAME_START=%FRAME_START%"
set "FRAME_END=%FRAME_END%"

for /f %%i in ('.venv\Scripts\python.exe -c "d='%DRIVE_ID%';print(d.split('_')[-2] if '_' in d else d)"') do set "DRIVE_TAG=%%i"
if "%FRAME_START%"=="" set "FRAME_START=250"
if "%FRAME_END%"=="" set "FRAME_END=500"

for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-ChildItem runs -Directory -Filter 'crosswalk_strict_%DRIVE_TAG%_%FRAME_START%_%FRAME_END%_*' | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName"`) do set "RUN_DIR=%%i"

if "%RUN_DIR%"=="" (
  echo [PROJ_SMOKE] strict run dir not found for %DRIVE_TAG%_%FRAME_START%_%FRAME_END%
  exit /b 1
)

set "OUT_DIR=%RUN_DIR%\outputs"
.venv\Scripts\python.exe tools\summarize_roundtrip_subsets.py --run-dir "%OUT_DIR%" --out-md projection_alignment_report_v2.md --topn 20 --iou-thr 0.05 --make-failpack
.venv\Scripts\python.exe tools\summarize_fail_types.py --run-dir "%OUT_DIR%" --topn-per-type 5
.venv\Scripts\python.exe tools\build_failpack_index.py --run-dir "%OUT_DIR%" --format md

echo [PROJ_SMOKE] outputs: %OUT_DIR%
echo [PROJ_SMOKE] stats: %OUT_DIR%\roundtrip_stats_subsets.csv
echo [PROJ_SMOKE] fail_summary: %OUT_DIR%\fail_type_summary.md
echo [PROJ_SMOKE] failpack: %OUT_DIR%\proj_debug_failpack\index.md

endlocal

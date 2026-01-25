@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "RUN_DIR=%RUN_DIR%"

if "%RUN_DIR%"=="" (
  for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-ChildItem runs -Directory -Filter 'crosswalk_strict_0010_250_500_*' | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName"`) do set "RUN_DIR=%%i"
)

if "%RUN_DIR%"=="" (
  echo [REPORT_V2] no strict run dir found
  exit /b 1
)

set "OUT_DIR=%RUN_DIR%\\outputs"

.venv\Scripts\python.exe tools\summarize_roundtrip_subsets.py --run-dir "%OUT_DIR%" --out-md projection_alignment_report_v2.md --topn 20 --iou-thr 0.05 --make-failpack

echo [REPORT_V2] outputs: %OUT_DIR%
echo [REPORT_V2] report: %OUT_DIR%\projection_alignment_report_v2.md
echo [REPORT_V2] stats: %OUT_DIR%\roundtrip_stats_subsets.csv
echo [REPORT_V2] fail: %OUT_DIR%\roundtrip_fail_frames_top20.csv
echo [REPORT_V2] failpack: %OUT_DIR%\proj_debug_failpack

endlocal

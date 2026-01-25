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
  echo [FAILPACK] no strict run dir found
  exit /b 1
)

set "OUT_DIR=%RUN_DIR%\\outputs"

.venv\Scripts\python.exe tools\summarize_fail_types.py --run-dir "%OUT_DIR%" --topn-per-type 5
.venv\Scripts\python.exe tools\build_failpack_index.py --run-dir "%OUT_DIR%" --format md

echo [FAILPACK] outputs: %OUT_DIR%
echo [FAILPACK] fail_type_summary: %OUT_DIR%\fail_type_summary.md
echo [FAILPACK] fail_type_csv: %OUT_DIR%\fail_type_summary.csv
echo [FAILPACK] index: %OUT_DIR%\proj_debug_failpack\index.md

endlocal

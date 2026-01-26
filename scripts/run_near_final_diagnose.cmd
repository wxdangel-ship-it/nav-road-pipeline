@echo off
setlocal

if "%RUN_DIR%"=="" set RUN_DIR=runs\crosswalk_sam2video_0010_250_500_20260124_230129\outputs
if "%OUT_DIR%"=="" set OUT_DIR=%RUN_DIR%\near_final_diagnose
if "%CLUSTERS%"=="" set CLUSTERS=cluster_0000,cluster_0002,cluster_0004

set PYTHON=.venv\Scripts\python.exe
if not exist %PYTHON% set PYTHON=python

%PYTHON% tools\near_final_diagnose.py --run-dir %RUN_DIR% --clusters %CLUSTERS% --out-dir %OUT_DIR%
if errorlevel 1 exit /b 1

echo [NEAR_FINAL] out: %OUT_DIR%
echo [NEAR_FINAL] report: %OUT_DIR%\near_final_report.md
echo [NEAR_FINAL] overview: %OUT_DIR%\overview.csv
endlocal

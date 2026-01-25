@echo off
setlocal

set "CFG=configs\crosswalk_stage2_autotune_250_500_v2.yaml"

.venv\Scripts\python.exe tools\autotune_crosswalk_stage2.py --config "%CFG%"
if errorlevel 1 exit /b 1

echo [AUTOTUNE] done
endlocal

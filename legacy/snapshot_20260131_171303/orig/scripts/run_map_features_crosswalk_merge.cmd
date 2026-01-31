@echo off
setlocal
set "PYTHONUTF8=1"
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."

.venv\Scripts\python.exe -m scripts.run_map_features_crosswalk_merge %*
if errorlevel 1 exit /b 1
endlocal

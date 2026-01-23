@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "CFG=configs\crosswalk_monitor.yaml"
set "MONITOR_DRIVE=%MONITOR_DRIVE%"
set "MONITOR_START=%MONITOR_START%"
set "MONITOR_END=%MONITOR_END%"
set "KITTI_ROOT=%KITTI_ROOT%"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
if "%MONITOR_DRIVE%"=="" for /f %%i in ('.venv\Scripts\python.exe -c "import yaml;print(yaml.safe_load(open(r\"configs\\crosswalk_monitor.yaml\",encoding=\"utf-8\"))[\"drive_id\"])\"') do set "MONITOR_DRIVE=%%i"
if "%MONITOR_START%"=="" for /f %%i in ('.venv\Scripts\python.exe -c "import yaml;print(yaml.safe_load(open(r\"configs\\crosswalk_monitor.yaml\",encoding=\"utf-8\"))[\"frame_start\"])\"') do set "MONITOR_START=%%i"
if "%MONITOR_END%"=="" for /f %%i in ('.venv\Scripts\python.exe -c "import yaml;print(yaml.safe_load(open(r\"configs\\crosswalk_monitor.yaml\",encoding=\"utf-8\"))[\"frame_end\"])\"') do set "MONITOR_END=%%i"

set "OUT_RUN=runs\crosswalk_monitor_%MONITOR_DRIVE%_%MONITOR_START%_%MONITOR_END%_%TS%"

set "ARGS=--config "%CFG%" --out-run "%OUT_RUN%""
if not "%MONITOR_DRIVE%"=="" set "ARGS=%ARGS% --drive "%MONITOR_DRIVE%""
if not "%MONITOR_START%"=="" set "ARGS=%ARGS% --frame-start %MONITOR_START%"
if not "%MONITOR_END%"=="" set "ARGS=%ARGS% --frame-end %MONITOR_END%"
if not "%KITTI_ROOT%"=="" set "ARGS=%ARGS% --kitti-root "%KITTI_ROOT%""

.venv\Scripts\python.exe tools\run_crosswalk_monitor_range.py %ARGS%

echo [MONITOR] outputs: %OUT_RUN%\outputs
echo [MONITOR] report: %OUT_RUN%\outputs\crosswalk_monitor_report.md
echo [MONITOR] qa_index: %OUT_RUN%\outputs\qa_index_wgs84.geojson
echo [MONITOR] trace: %OUT_RUN%\outputs\crosswalk_trace.csv

endlocal

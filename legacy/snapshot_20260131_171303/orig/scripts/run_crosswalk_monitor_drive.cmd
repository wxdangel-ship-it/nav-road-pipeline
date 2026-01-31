@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "CFG=configs\crosswalk_monitor_drive.yaml"
set "MONITOR_DRIVE=%MONITOR_DRIVE%"
set "KITTI_ROOT=%KITTI_ROOT%"

if "%MONITOR_DRIVE%"=="" for /f %%i in ('.venv\Scripts\python.exe -c "import yaml;print(yaml.safe_load(open('configs/crosswalk_monitor_drive.yaml',encoding='utf-8'))['drive_id'])"') do set "MONITOR_DRIVE=%%i"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
set "OUT_RUN=runs\crosswalk_monitor_%MONITOR_DRIVE%_%TS%"

set "EXTRA="
if not "%MONITOR_DRIVE%"=="" set "EXTRA=%EXTRA% --drive %MONITOR_DRIVE%"
if not "%KITTI_ROOT%"=="" set "EXTRA=%EXTRA% --kitti-root %KITTI_ROOT%"

.venv\Scripts\python.exe tools\run_crosswalk_monitor_drive.py --config "%CFG%" --out-run "%OUT_RUN%" %EXTRA%

echo [MONITOR] outputs: %OUT_RUN%\outputs
echo [MONITOR] report: %OUT_RUN%\outputs\crosswalk_monitor_report.md
echo [MONITOR] qa_index: %OUT_RUN%\outputs\qa_index_wgs84.geojson
echo [MONITOR] trace: %OUT_RUN%\outputs\crosswalk_trace.csv

endlocal

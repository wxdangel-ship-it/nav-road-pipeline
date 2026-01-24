@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "CFG=configs\crosswalk_drive_full.yaml"
set "DRIVE_ID=%DRIVE_ID%"
set "KITTI_ROOT=%KITTI_ROOT%"
set "EXPORT_ALL_FRAMES=%EXPORT_ALL_FRAMES%"

if "%DRIVE_ID%"=="" for /f %%i in ('.venv\Scripts\python.exe -c "import yaml;print(yaml.safe_load(open('configs/crosswalk_drive_full.yaml',encoding='utf-8'))['drive_id'])"') do set "DRIVE_ID=%%i"
for /f %%i in ('.venv\Scripts\python.exe -c "d='%DRIVE_ID%';print(d.split('_')[-2] if '_' in d else d)"') do set "DRIVE_TAG=%%i"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
set "OUT_RUN=runs\crosswalk_drive%DRIVE_TAG%_full_%TS%"

set "EXTRA="
if not "%DRIVE_ID%"=="" set "EXTRA=%EXTRA% --drive %DRIVE_ID%"
if not "%KITTI_ROOT%"=="" set "EXTRA=%EXTRA% --kitti-root %KITTI_ROOT%"
if not "%EXPORT_ALL_FRAMES%"=="" set "EXTRA=%EXTRA% --export-all-frames %EXPORT_ALL_FRAMES%"

.venv\Scripts\python.exe tools\run_crosswalk_drive_full.py --config "%CFG%" --out-run "%OUT_RUN%" %EXTRA%

echo [FULL] outputs: %OUT_RUN%\outputs
echo [FULL] report: %OUT_RUN%\outputs\crosswalk_full_report.md
echo [FULL] trace: %OUT_RUN%\outputs\crosswalk_trace.csv
echo [FULL] qa_index: %OUT_RUN%\outputs\qa_index_wgs84.geojson

endlocal

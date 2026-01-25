@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "CFG=configs\crosswalk_range_250_500_strict.yaml"
set "DRIVE_ID=%DRIVE_ID%"
set "FRAME_START=%FRAME_START%"
set "FRAME_END=%FRAME_END%"
set "KITTI_ROOT=%KITTI_ROOT%"

if "%DRIVE_ID%"=="" for /f %%i in ('.venv\Scripts\python.exe -c "import yaml;print(yaml.safe_load(open('configs/crosswalk_range_250_500_strict.yaml',encoding='utf-8'))['drive_id'])"') do set "DRIVE_ID=%%i"
if "%FRAME_START%"=="" for /f %%i in ('.venv\Scripts\python.exe -c "import yaml;print(yaml.safe_load(open('configs/crosswalk_range_250_500_strict.yaml',encoding='utf-8'))['frame_start'])"') do set "FRAME_START=%%i"
if "%FRAME_END%"=="" for /f %%i in ('.venv\Scripts\python.exe -c "import yaml;print(yaml.safe_load(open('configs/crosswalk_range_250_500_strict.yaml',encoding='utf-8'))['frame_end'])"') do set "FRAME_END=%%i"

for /f %%i in ('.venv\Scripts\python.exe -c "d='%DRIVE_ID%';print(d.split('_')[-2] if '_' in d else d)"') do set "DRIVE_TAG=%%i"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
set "OUT_RUN=runs\crosswalk_strict_%DRIVE_TAG%_%FRAME_START%_%FRAME_END%_%TS%"

set "EXTRA="
if not "%DRIVE_ID%"=="" set "EXTRA=%EXTRA% --drive %DRIVE_ID%"
if not "%FRAME_START%"=="" set "EXTRA=%EXTRA% --frame-start %FRAME_START%"
if not "%FRAME_END%"=="" set "EXTRA=%EXTRA% --frame-end %FRAME_END%"
if not "%KITTI_ROOT%"=="" set "EXTRA=%EXTRA% --kitti-root %KITTI_ROOT%"

.venv\Scripts\python.exe tools\run_crosswalk_monitor_range.py --config "%CFG%" --out-run "%OUT_RUN%" %EXTRA%

echo [STRICT] outputs: %OUT_RUN%\outputs
echo [STRICT] report: %OUT_RUN%\outputs\crosswalk_stage2_report.md
echo [STRICT] trace: %OUT_RUN%\outputs\crosswalk_trace.csv
echo [STRICT] qa_index: %OUT_RUN%\outputs\qa_index_wgs84.geojson
echo [STRICT] range_assert: %OUT_RUN%\outputs\frame_range_assert.txt

endlocal

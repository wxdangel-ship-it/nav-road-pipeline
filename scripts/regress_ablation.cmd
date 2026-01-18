@echo off
setlocal enabledelayedexpansion
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

if "%POC_DATA_ROOT%"=="" (
  if exist "E:\KITTI360\KITTI-360" set "POC_DATA_ROOT=E:\KITTI360\KITTI-360"
)
if not "%POC_DATA_ROOT%"=="" echo [ABLATION] POC_DATA_ROOT=%POC_DATA_ROOT%

if "%MAX_FRAMES%"=="" set "MAX_FRAMES=2000"
if "%MAX_DRIVES%"=="" set "MAX_DRIVES=6"
if "%REQUIRE_OXTS%"=="" set "REQUIRE_OXTS=1"
if "%REQUIRE_VELODYNE%"=="" set "REQUIRE_VELODYNE=1"
if "%ALLOW_SKIP_INVALID%"=="" set "ALLOW_SKIP_INVALID=0"

set "ARMS=ArmA ArmB ArmC ArmD"

set "DRIVES_LIST=%DRIVES%"
if not "%DRIVES_LIST%"=="" set "DRIVES_LIST=%DRIVES_LIST:,= %"
set "DISCOVERY_MODE=auto"

if "%DRIVES_LIST%"=="" (
  if "%DRIVES_FILE%"=="" set "DRIVES_FILE=configs\drives.txt"
  if exist "%DRIVES_FILE%" (
    for /f "usebackq delims=" %%d in ("%DRIVES_FILE%") do (
      set "LINE=%%d"
      if not "!LINE!"=="" if not "!LINE:~0,1!"=="#" set "DRIVES_LIST=!DRIVES_LIST! !LINE!"
    )
  )
)

if not "%DRIVES_LIST%"=="" (
  set "DISCOVERY_MODE=explicit"
)

for /f "usebackq delims=" %%t in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "RUN_TAG=%%t"
set "REGRESS_DIR=runs\regress_ablation_%RUN_TAG%"
if not exist "%REGRESS_DIR%" mkdir "%REGRESS_DIR%"
set "INDEX_FILE=%REGRESS_DIR%\ablation_index.jsonl"
if exist "%INDEX_FILE%" del /f /q "%INDEX_FILE%" 2>nul

if "%DRIVES_LIST%"=="" (
  if "%POC_DATA_ROOT%"=="" (
    echo [ABLATION] ERROR: POC_DATA_ROOT not set and default path not found.
    exit /b 1
  )
  set "DISCOVERY_MODE=auto"
  set "SEEN_DRIVES= "
  set "DISCOVERED_COUNT=0"
  set "FOUND_VALID=0"
  set "SELECTED_COUNT=0"
  for %%S in ("%POC_DATA_ROOT%" "%POC_DATA_ROOT%\data_3d_raw" "%POC_DATA_ROOT%\data_2d_raw" "%POC_DATA_ROOT%\data_poses" "%POC_DATA_ROOT%\data_poses_oxts" "%POC_DATA_ROOT%\data_poses_oxts_extract") do (
    for /d %%d in ("%%~S\*_drive_*_sync") do (
      set "DRIVE=%%~nxd"
      echo !SEEN_DRIVES! | findstr /i /c:" !DRIVE! " >nul
      if errorlevel 1 (
        set "SEEN_DRIVES=!SEEN_DRIVES!!DRIVE! "
        set /a DISCOVERED_COUNT+=1
        call :check_drive_data
        if "!DRIVE_OK!"=="1" (
          set /a FOUND_VALID+=1
          set "DRIVES_LIST=!DRIVES_LIST! !DRIVE!"
          set /a SELECTED_COUNT+=1
          if "!SELECTED_COUNT!" GEQ "%MAX_DRIVES%" goto :discovery_done
        ) else (
          echo [ABLATION] WARN: drive !DRIVE! skipped ^(!DRIVE_BAD_REASON!^)
        )
      )
    )
  )
)

:discovery_done
if "%DISCOVERY_MODE%"=="auto" (
  echo [ABLATION] auto-discovery checked=%DISCOVERED_COUNT% valid=%FOUND_VALID% selected=%SELECTED_COUNT% max_drives=%MAX_DRIVES%
  echo [ABLATION] selected drives:%DRIVES_LIST%
)

if "%DRIVES_LIST%"=="" (
  echo [ABLATION] ERROR: no drives found. Set DRIVES or DRIVES_FILE or POC_DATA_ROOT/MAX_DRIVES.
  exit /b 1
)

set "PASS_ARM_A=0"
set "PASS_ARM_B=0"

for %%d in (%DRIVES_LIST%) do (
  set "DRIVE=%%d"
  echo [ABLATION] drive=%%d

  call :check_drive_data
  if "!DRIVE_OK!"=="0" (
    for %%a in (%ARMS%) do (
      set "ARM=%%a"
      call :write_index SKIPPED "!DRIVE_BAD_REASON!" "" ""
    )
    echo [ABLATION] WARN: drive %%d skipped ^(!DRIVE_BAD_REASON!^)
  ) else (
    for %%a in (%ARMS%) do (
      set "ARM=%%a"
      call :run_arm
    )
  )
)

.venv\Scripts\python.exe tools\collect_ablation_report.py --regress-dir "%REGRESS_DIR%"

set "RC=0"
if %PASS_ARM_A% EQU 0 set "RC=1"
if %PASS_ARM_B% EQU 0 set "RC=1"

echo [ABLATION] DONE -^> %REGRESS_DIR%
endlocal & exit /b %RC%

:run_arm
set "ARM_STATUS=PASS"
set "ARM_REASON="
if /I "%ARM%"=="ArmA" (
  set "GEOM_BACKEND=algo"
) else (
  set "GEOM_BACKEND=nn"
)
echo [ABLATION] arm=%ARM% backend=%GEOM_BACKEND%

call .\scripts\build_geom.cmd --drive %DRIVE% --max-frames %MAX_FRAMES%
if errorlevel 1 (
  set "ARM_STATUS=FAIL"
  set "ARM_REASON=build_geom_failed"
  call :write_index !ARM_STATUS! "!ARM_REASON!" "" ""
  goto :eof
)

for /f "usebackq delims=" %%p in (`
  .venv\Scripts\python.exe -c "from pathlib import Path; runs=Path('runs'); dirs=sorted([p for p in runs.iterdir() if p.is_dir() and p.name.startswith('geom_')], key=lambda p: p.stat().st_mtime); print(dirs[-1].name if dirs else '')"
`) do set "GEOM_RUN=%%p"
if "!GEOM_RUN!"=="" (
  set "ARM_STATUS=FAIL"
  set "ARM_REASON=geom_run_not_found"
  call :write_index !ARM_STATUS! "!ARM_REASON!" "" ""
  goto :eof
)
set "GEOM_OUT=runs\!GEOM_RUN!\outputs"

if /I "%ARM%"=="ArmC" (
  .venv\Scripts\python.exe tools\osm_ref_extract.py --outputs-dir "!GEOM_OUT!"
  if errorlevel 1 (
    set "ARM_STATUS=FAIL"
    set "ARM_REASON=osm_ref_failed"
  )
)

if /I "%ARM%"=="ArmD" (
  .venv\Scripts\python.exe tools\make_qgis_package.py --outputs-dir "!GEOM_OUT!"
  if errorlevel 1 (
    set "ARM_STATUS=FAIL"
    set "ARM_REASON=qgis_package_failed"
  )
)

call :write_index !ARM_STATUS! "!ARM_REASON!" "!GEOM_RUN!" "!GEOM_OUT!"

if "!ARM_STATUS!"=="PASS" (
  if /I "%ARM%"=="ArmA" set /a PASS_ARM_A+=1
  if /I "%ARM%"=="ArmB" set /a PASS_ARM_B+=1
)
goto :eof

:check_drive_data
set "DRIVE_OK=1"
set "DRIVE_BAD_REASON="
if "!POC_DATA_ROOT!"=="" (
  set "DRIVE_OK=0"
  set "DRIVE_BAD_REASON=missing_drive_dir"
  goto :eof
)
set "DRIVE_DIR_OK=0"
if exist "!POC_DATA_ROOT!\!DRIVE!\" set "DRIVE_DIR_OK=1"
if exist "!POC_DATA_ROOT!\data_poses\!DRIVE!\" set "DRIVE_DIR_OK=1"
if exist "!POC_DATA_ROOT!\data_poses\oxts\!DRIVE!\" set "DRIVE_DIR_OK=1"
if exist "!POC_DATA_ROOT!\data_poses_oxts\!DRIVE!\" set "DRIVE_DIR_OK=1"
if exist "!POC_DATA_ROOT!\data_poses_oxts_extract\!DRIVE!\" set "DRIVE_DIR_OK=1"
if exist "!POC_DATA_ROOT!\data_3d_raw\!DRIVE!\" set "DRIVE_DIR_OK=1"
if "!DRIVE_DIR_OK!"=="0" (
  set "DRIVE_OK=0"
  set "DRIVE_BAD_REASON=missing_drive_dir"
  goto :eof
)

if "%REQUIRE_OXTS%"=="1" (
  set "OXTS_OK=0"
  if exist "!POC_DATA_ROOT!\!DRIVE!\oxts\data\" set "OXTS_OK=1"
  if exist "!POC_DATA_ROOT!\data_poses\!DRIVE!\oxts\data\" set "OXTS_OK=1"
  if exist "!POC_DATA_ROOT!\data_poses\oxts\!DRIVE!\oxts\data\" set "OXTS_OK=1"
  if exist "!POC_DATA_ROOT!\data_poses\!DRIVE!\data\" set "OXTS_OK=1"
  if exist "!POC_DATA_ROOT!\data_poses_oxts\!DRIVE!\oxts\data\" set "OXTS_OK=1"
  if exist "!POC_DATA_ROOT!\data_poses_oxts_extract\!DRIVE!\oxts\data\" set "OXTS_OK=1"
  if "!OXTS_OK!"=="0" (
    set "DRIVE_OK=0"
    set "DRIVE_BAD_REASON=missing_oxts"
    goto :eof
  )
)

if "%REQUIRE_VELODYNE%"=="1" (
  set "VELO_OK=0"
  if exist "!POC_DATA_ROOT!\!DRIVE!\velodyne_points\data\" set "VELO_OK=1"
  if exist "!POC_DATA_ROOT!\data_3d_raw\!DRIVE!\velodyne_points\data\" set "VELO_OK=1"
  if exist "!POC_DATA_ROOT!\data_3d_raw\!DRIVE!\velodyne_points\data\1\" set "VELO_OK=1"
  if "!VELO_OK!"=="0" (
    set "DRIVE_OK=0"
    set "DRIVE_BAD_REASON=missing_velodyne"
    goto :eof
  )
)
goto :eof

:write_index
set "IDX_STATUS=%~1"
set "IDX_REASON=%~2"
set "IDX_RUN=%~3"
set "IDX_OUT=%~4"
for /f "usebackq delims=" %%t in (`powershell -NoProfile -Command "Get-Date -Format yyyy-MM-ddTHH:mm:ss"`) do set "IDX_TS=%%t"
.venv\Scripts\python.exe -c "import json,os; out=os.environ.get('IDX_OUT') or None; print(json.dumps({'arm':os.environ.get('ARM'), 'drive':os.environ.get('DRIVE'), 'status':os.environ.get('IDX_STATUS'), 'reason':os.environ.get('IDX_REASON') or None, 'geom_run_id':os.environ.get('IDX_RUN') or None, 'outputs_dir': out, 'timestamp':os.environ.get('IDX_TS')}))" >> "%INDEX_FILE%"
set "IDX_STATUS="
set "IDX_REASON="
set "IDX_RUN="
set "IDX_OUT="
set "IDX_TS="
goto :eof

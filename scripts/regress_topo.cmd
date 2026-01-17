@echo off
setlocal enabledelayedexpansion
cd /d %~dp0\..

rem validate_topo_outputs.py supports --summary/--issues/--actions or env TOPO_*.

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

if "%MAX_FRAMES%"=="" set "MAX_FRAMES=2000"
if "%MAX_DRIVES%"=="" set "MAX_DRIVES=3"
if "%REQUIRE_OXTS%"=="" set "REQUIRE_OXTS=1"
if "%REQUIRE_VELODYNE%"=="" set "REQUIRE_VELODYNE=1"
if "%ALLOW_SKIP_INVALID%"=="" set "ALLOW_SKIP_INVALID=0"

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

if "%DRIVES_LIST%"=="" (
  for /f "usebackq delims=" %%d in (`
    .venv\Scripts\python.exe -c "from pathlib import Path; import os; from pipeline.adapters.kitti360_adapter import discover_drives; root=os.environ.get('POC_DATA_ROOT',''); drives=discover_drives(Path(root)) if root else []; max_drives=int(os.environ.get('MAX_DRIVES','3')); print('\n'.join(drives[:max_drives]))"
  `) do (
    if not "%%d"=="" set "DRIVES_LIST=!DRIVES_LIST! %%d"
  )
)

if "%DRIVES_LIST%"=="" (
  echo [REGRESS] ERROR: no drives found. Set DRIVES or DRIVES_FILE or POC_DATA_ROOT/MAX_DRIVES.
  exit /b 1
)

for /f "usebackq delims=" %%t in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "RUN_TAG=%%t"
set "REGRESS_DIR=runs\regress_%RUN_TAG%"
if not exist "%REGRESS_DIR%" mkdir "%REGRESS_DIR%"
set "INDEX_FILE=%REGRESS_DIR%\regress_index.jsonl"
if exist "%INDEX_FILE%" del /f /q "%INDEX_FILE%" 2>nul

set "RUNNABLE_COUNT=0"

for %%d in (%DRIVES_LIST%) do (
  set "DRIVE=%%d"
  echo [REGRESS] drive=%%d

  call :check_drive_data
  if "!DRIVE_OK!"=="0" (
    set "ALLOW_SKIP=1"
    if "!DISCOVERY_MODE!"=="explicit" if "%ALLOW_SKIP_INVALID%"=="0" set "ALLOW_SKIP=0"
    if "!ALLOW_SKIP!"=="0" (
      call :write_index FAIL "!DRIVE_BAD_REASON!" "" ""
      exit /b 1
    )
    call :write_index SKIPPED "!DRIVE_BAD_REASON!" "" ""
    echo [REGRESS] WARN: drive %%d skipped ^(!DRIVE_BAD_REASON!^)
  )

  if "!DRIVE_OK!"=="1" (
    set /a RUNNABLE_COUNT+=1
    call .\scripts\build_geom.cmd --drive %%d --max-frames %MAX_FRAMES%
    if errorlevel 1 (
      call :write_index FAIL "build_geom_failed" "" ""
      exit /b 1
    )

    call .\scripts\build_topo.cmd --drive %%d --max-frames %MAX_FRAMES%
    if errorlevel 1 (
      call :write_index FAIL "build_topo_failed" "" ""
      exit /b 1
    )

    for /f "usebackq delims=" %%p in (`
      .venv\Scripts\python.exe -c "from pathlib import Path; runs=Path('runs'); dirs=sorted([p for p in runs.iterdir() if p.is_dir() and p.name.startswith('topo_')], key=lambda p: p.stat().st_mtime); print(dirs[-1].name if dirs else '')"
    `) do set "TOPO_RUN=%%p"
    if "!TOPO_RUN!"=="" (
      call :write_index FAIL "topo_run_not_found" "" ""
      echo [REGRESS] ERROR: could not find topo run for drive %%d
      exit /b 1
    )
    set "TOPO_OUT=runs\!TOPO_RUN!\outputs"

    call .\scripts\eval.cmd --max-frames %MAX_FRAMES%
    if errorlevel 1 (
      call :write_index FAIL "eval_failed" "!TOPO_RUN!" "!TOPO_OUT!"
      exit /b 1
    )

    .venv\Scripts\python.exe tools\validate_topo_outputs.py --summary "!TOPO_OUT!\TopoSummary.md" --issues "!TOPO_OUT!\TopoIssues.jsonl" --actions "!TOPO_OUT!\TopoActions.jsonl"
    if errorlevel 1 (
      call :write_index FAIL "validate_failed" "!TOPO_RUN!" "!TOPO_OUT!"
      exit /b 1
    )

    call :write_index PASS "" "!TOPO_RUN!" "!TOPO_OUT!"
  )
)

if %RUNNABLE_COUNT% EQU 0 (
  call :write_index FAIL "no_valid_drives" "" ""
  exit /b 1
)

.venv\Scripts\python.exe tools\collect_topo_regress.py --regress-dir "%REGRESS_DIR%"
if errorlevel 1 exit /b 1

echo [REGRESS] DONE -> %REGRESS_DIR%
endlocal
exit /b 0

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
.venv\Scripts\python.exe -c "import json,os; out=os.environ.get('IDX_OUT') or None; print(json.dumps({'drive':os.environ.get('DRIVE'), 'status':os.environ.get('IDX_STATUS'), 'reason':os.environ.get('IDX_REASON') or None, 'run_id':os.environ.get('IDX_RUN') or None, 'outputs_dir': out, 'topo_outputs': out, 'summary_path': (os.path.join(out, 'TopoSummary.md') if out else None), 'issues_path': (os.path.join(out, 'TopoIssues.jsonl') if out else None), 'actions_path': (os.path.join(out, 'TopoActions.jsonl') if out else None), 'timestamp':os.environ.get('IDX_TS')}))" >> "%INDEX_FILE%"
set "IDX_STATUS="
set "IDX_REASON="
set "IDX_RUN="
set "IDX_OUT="
set "IDX_TS="
goto :eof

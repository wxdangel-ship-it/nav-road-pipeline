@echo off
setlocal enabledelayedexpansion
cd /d %~dp0\..

rem validate_topo_outputs.py supports --summary/--issues/--actions or env TOPO_*.

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

if "%POC_DATA_ROOT%"=="" (
  if exist "E:\KITTI360\KITTI-360" set "POC_DATA_ROOT=E:\KITTI360\KITTI-360"
)
if not "%POC_DATA_ROOT%"=="" echo [REGRESS] POC_DATA_ROOT=%POC_DATA_ROOT%

if "%MAX_FRAMES%"=="" set "MAX_FRAMES=2000"
if "%MAX_DRIVES%"=="" set "MAX_DRIVES=6"
if "%REQUIRE_OXTS%"=="" set "REQUIRE_OXTS=1"
if "%REQUIRE_VELODYNE%"=="" set "REQUIRE_VELODYNE=1"
if "%ALLOW_SKIP_INVALID%"=="" set "ALLOW_SKIP_INVALID=0"
if "%ENABLE_GATE%"=="" set "ENABLE_GATE=1"
if "%BASELINE_PATH%"=="" set "BASELINE_PATH=configs\topo_regress_baseline.yaml"
if "%BASELINE_MODE%"=="" (
  if exist "%BASELINE_PATH%" (
    set "BASELINE_MODE=compare"
  ) else (
    set "BASELINE_MODE=off"
  )
)

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
set "REGRESS_DIR=runs\regress_%RUN_TAG%"
if not exist "%REGRESS_DIR%" mkdir "%REGRESS_DIR%"
set "INDEX_FILE=%REGRESS_DIR%\regress_index.jsonl"
if exist "%INDEX_FILE%" del /f /q "%INDEX_FILE%" 2>nul

if "%DRIVES_LIST%"=="" (
  if "%POC_DATA_ROOT%"=="" (
    echo [REGRESS] ERROR: POC_DATA_ROOT not set and default path not found.
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
          call :write_index SKIPPED "!DRIVE_BAD_REASON!" "" ""
          echo [REGRESS] WARN: drive !DRIVE! skipped ^(!DRIVE_BAD_REASON!^)
        )
      )
    )
  )
)

:discovery_done
if "%DISCOVERY_MODE%"=="auto" (
  echo [REGRESS] auto-discovery checked=%DISCOVERED_COUNT% valid=%FOUND_VALID% selected=%SELECTED_COUNT% max_drives=%MAX_DRIVES%
  echo [REGRESS] selected drives:%DRIVES_LIST%
)

if "%DRIVES_LIST%"=="" (
  echo [REGRESS] ERROR: no drives found. Set DRIVES or DRIVES_FILE or POC_DATA_ROOT/MAX_DRIVES.
  exit /b 1
)

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

set "RC=0"

.venv\Scripts\python.exe tools\collect_topo_regress.py --regress-dir "%REGRESS_DIR%" --gate-config "configs\topo_regress_gate.yaml" --baseline "%BASELINE_PATH%"
if errorlevel 1 (
  set "RC=1"
  goto :done
)

if "%ENABLE_GATE%"=="1" (
  .venv\Scripts\python.exe tools\check_topo_regress_gate.py --index "%INDEX_FILE%" --config "configs\topo_regress_gate.yaml" --baseline "%BASELINE_PATH%" --baseline-mode "%BASELINE_MODE%"
  if errorlevel 1 set "RC=1"
) else (
  call :check_min_pass
  if errorlevel 1 set "RC=1"
)

:done
echo [REGRESS] DONE -^> %REGRESS_DIR%
endlocal & exit /b %RC%

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

:check_min_pass
.venv\Scripts\python.exe -c "import json, os, sys, yaml; idx=os.environ.get('INDEX_FILE'); cfg=yaml.safe_load(open('configs/topo_regress_gate.yaml','r',encoding='utf-8')); min_pass=int(cfg.get('min_pass_drives',1)); cnt=0; f=open(idx,'r',encoding='utf-8');\nfor line in f:\n line=line.strip();\n if not line: continue\n if json.loads(line).get('status')=='PASS': cnt+=1\nsys.exit(0 if cnt>=min_pass else 1)"
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

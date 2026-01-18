@echo off
setlocal enabledelayedexpansion
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

if "%POC_DATA_ROOT%"=="" (
  if exist "E:\KITTI360\KITTI-360" set "POC_DATA_ROOT=E:\KITTI360\KITTI-360"
)
if not "%POC_DATA_ROOT%"=="" echo [EVIDENCE] POC_DATA_ROOT=%POC_DATA_ROOT%

if "%MAX_FRAMES%"=="" set "MAX_FRAMES=2000"
if "%MAX_DRIVES%"=="" set "MAX_DRIVES=6"
if "%REQUIRE_OXTS%"=="" set "REQUIRE_OXTS=1"
if "%REQUIRE_VELODYNE%"=="" set "REQUIRE_VELODYNE=1"
if "%ALLOW_SKIP_INVALID%"=="" set "ALLOW_SKIP_INVALID=0"

set "ARMS=Arm0 Arm1 Arm2 Arm3"

set "DRIVES_LIST=%DRIVES%"
if not "%DRIVES_LIST%"=="" set "DRIVES_LIST=%DRIVES_LIST:,= %"
set "GOLDEN_FILE="
if "%DRIVES_LIST%"=="" (
  if "%DRIVES_FILE%"=="" (
    if exist "configs\golden_drives.txt" (
      set "DRIVES_FILE=configs\\golden_drives.txt"
    )
  )
  if not "%DRIVES_FILE%"=="" if exist "%DRIVES_FILE%" (
    if /I "%DRIVES_FILE%"=="configs\\golden_drives.txt" set "GOLDEN_FILE=%DRIVES_FILE%"
    for /f "usebackq delims=" %%d in ("%DRIVES_FILE%") do (
      set "LINE=%%d"
      if not "!LINE!"=="" if not "!LINE:~0,1!"=="#" set "DRIVES_LIST=!DRIVES_LIST! !LINE!"
    )
  )
)

for /f "usebackq delims=" %%t in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "RUN_TAG=%%t"
set "REGRESS_DIR=runs\regress_evidence_%RUN_TAG%"
if not exist "%REGRESS_DIR%" mkdir "%REGRESS_DIR%"
set "INDEX_FILE=%REGRESS_DIR%\evidence_index.jsonl"
if exist "%INDEX_FILE%" del /f /q "%INDEX_FILE%" 2>nul

for %%a in (%ARMS%) do (
  if not exist "%REGRESS_DIR%\%%a" mkdir "%REGRESS_DIR%\%%a"
  if not exist "%REGRESS_DIR%\%%a\SyncPack" mkdir "%REGRESS_DIR%\%%a\SyncPack"
)

if "%DRIVES_LIST%"=="" (
  if "%POC_DATA_ROOT%"=="" (
    echo [EVIDENCE] ERROR: POC_DATA_ROOT not set and golden_drives.txt not found.
    exit /b 1
  )
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
        )
      )
    )
  )
)

:discovery_done
if "%DRIVES_LIST%"=="" (
  echo [EVIDENCE] ERROR: no drives found.
  exit /b 1
)
echo [EVIDENCE] drives:%DRIVES_LIST%

set "PASS_ARM0=0"

for %%d in (%DRIVES_LIST%) do (
  set "DRIVE=%%d"
  echo [EVIDENCE] drive=%%d
  set "BASE_OK=1"

  call :check_drive_data
  if "!DRIVE_OK!"=="0" (
    for %%a in (%ARMS%) do (
      set "ARM=%%a"
      call :write_index SKIPPED "!DRIVE_BAD_REASON!" "" "" "" ""
    )
    echo [EVIDENCE] WARN: drive %%d skipped ^(!DRIVE_BAD_REASON!^)
    set "BASE_OK=0"
  )
  if "!BASE_OK!"=="1" (
    set "GEOM_BACKEND=nn"
    call .\scripts\build_geom.cmd --drive %%d --max-frames %MAX_FRAMES%
    if errorlevel 1 (
      set "ARM=Arm0"
      call :write_index FAIL "build_geom_failed" "" "" "" ""
      for %%a in (Arm1 Arm2 Arm3) do (
        set "ARM=%%a"
        call :write_index SKIPPED "base_geom_failed" "" "" "" ""
      )
      set "BASE_OK=0"
    )
  )

  if "!BASE_OK!"=="1" (
    for /f "usebackq delims=" %%p in (`
      .venv\Scripts\python.exe -c "from pathlib import Path; runs=Path('runs'); dirs=sorted([p for p in runs.iterdir() if p.is_dir() and p.name.startswith('geom_')], key=lambda p: p.stat().st_mtime); print(dirs[-1].name if dirs else '')"
    `) do set "BASE_RUN=%%p"
    if "!BASE_RUN!"=="" (
      set "ARM=Arm0"
      call :write_index FAIL "geom_run_not_found" "" "" "" ""
      for %%a in (Arm1 Arm2 Arm3) do (
        set "ARM=%%a"
        call :write_index SKIPPED "base_geom_failed" "" "" "" ""
      )
      set "BASE_OK=0"
    )
  )

  if "!BASE_OK!"=="1" (
    set "BASE_OUT=runs\!BASE_RUN!\outputs"

    call :ensure_runcard Arm0 "!BASE_OUT!"
    call :sync_pack Arm0 "!BASE_OUT!"
    set "ARM=Arm0"
    call :write_index PASS "" "!BASE_RUN!" "!BASE_OUT!" "" ""
    set /a PASS_ARM0+=1

    call :run_osm Arm1 "!BASE_OUT!"
    call :run_dop20 Arm2 "!BASE_OUT!"
    call :run_osm_dop20 Arm3 "!BASE_OUT!"
  )
)

.venv\Scripts\python.exe tools\collect_evidence_report.py --regress-dir "%REGRESS_DIR%"

set "RC=0"
if %PASS_ARM0% EQU 0 set "RC=1"

echo [EVIDENCE] DONE -^> %REGRESS_DIR%
endlocal & exit /b %RC%

:run_osm
set "ARM=%~1"
set "BASE_OUT=%~2"
set "ARM_STATUS=PASS"
set "ARM_REASON="
if exist "!BASE_OUT!\osm_ref_metrics.json" (
  rem reuse cached osm metrics
) else (
  .venv\Scripts\python.exe tools\osm_ref_extract.py --outputs-dir "!BASE_OUT!"
)
if errorlevel 1 (
  set "ARM_STATUS=FAIL"
  set "ARM_REASON=osm_ref_failed"
)
call :ensure_runcard !ARM! "!BASE_OUT!"
call :sync_pack !ARM! "!BASE_OUT!"
call :write_index !ARM_STATUS! "!ARM_REASON!" "!BASE_RUN!" "!BASE_OUT!" "!BASE_OUT!\osm_ref_metrics.json" ""
goto :eof

:run_dop20
set "ARM=%~1"
set "BASE_OUT=%~2"
set "ARM_STATUS=PASS"
set "ARM_REASON="
if exist "!BASE_OUT!\qgis_package\layers.json" (
  rem reuse cached qgis package
) else (
  .venv\Scripts\python.exe tools\make_qgis_package.py --outputs-dir "!BASE_OUT!"
)
if errorlevel 1 (
  set "ARM_STATUS=FAIL"
  set "ARM_REASON=qgis_package_failed"
)
call :ensure_runcard !ARM! "!BASE_OUT!"
call :sync_pack !ARM! "!BASE_OUT!"
call :write_index !ARM_STATUS! "!ARM_REASON!" "!BASE_RUN!" "!BASE_OUT!" "" "!BASE_OUT!\qgis_package\layers.json"
goto :eof

:run_osm_dop20
set "ARM=%~1"
set "BASE_OUT=%~2"
set "ARM_STATUS=PASS"
set "ARM_REASON="
if exist "!BASE_OUT!\osm_ref_metrics.json" (
  rem reuse cached osm metrics
) else (
  .venv\Scripts\python.exe tools\osm_ref_extract.py --outputs-dir "!BASE_OUT!"
)
if errorlevel 1 (
  set "ARM_STATUS=FAIL"
  set "ARM_REASON=osm_ref_failed"
)
if exist "!BASE_OUT!\qgis_package\layers.json" (
  rem reuse cached qgis package
) else (
  .venv\Scripts\python.exe tools\make_qgis_package.py --outputs-dir "!BASE_OUT!"
)
if errorlevel 1 (
  set "ARM_STATUS=FAIL"
  if "!ARM_REASON!"=="" (
    set "ARM_REASON=qgis_package_failed"
  )
)
call :ensure_runcard !ARM! "!BASE_OUT!"
call :sync_pack !ARM! "!BASE_OUT!"
call :write_index !ARM_STATUS! "!ARM_REASON!" "!BASE_RUN!" "!BASE_OUT!" "!BASE_OUT!\osm_ref_metrics.json" "!BASE_OUT!\qgis_package\layers.json"
goto :eof

:ensure_runcard
set "ARM=%~1"
set "BASE_OUT=%~2"
if exist "%REGRESS_DIR%\%ARM%\RunCard.json" goto :eof
.venv\Scripts\python.exe -c "import json,os; from pathlib import Path; base=Path(os.environ.get('BASE_OUT')); summary=json.loads((base/'GeomSummary.json').read_text(encoding='utf-8')); arm=os.environ.get('ARM'); cfg={'arm':arm,'evidence':{'osm': arm in ('Arm1','Arm3'), 'sat': arm in ('Arm2','Arm3')},'geom':{'backend_used':summary.get('backend_used'),'model_id':summary.get('model_id'),'camera':summary.get('camera'),'stride':summary.get('stride')},'osm_root':os.environ.get('OSM_ROOT') or str(Path(os.environ.get('POC_DATA_ROOT',''))/'_osm_download'),'dop20_root':os.environ.get('DOP20_ROOT') or r'E:\\KITTI360\\KITTI-360\\_lglbw_dop20','golden_drives_file': os.environ.get('GOLDEN_FILE')}; out=Path(os.environ.get('REGRESS_DIR'))/arm/'RunCard.json'; out.write_text(json.dumps(cfg,ensure_ascii=True,indent=2),encoding='utf-8')" >nul
goto :eof

:sync_pack
set "ARM=%~1"
set "BASE_OUT=%~2"
set "DEST=%REGRESS_DIR%\%ARM%\SyncPack\%DRIVE%"
if not exist "%DEST%" mkdir "%DEST%"
copy /y "%BASE_OUT%\road_polygon_wgs84.geojson" "%DEST%\road_polygon_wgs84.geojson" >nul 2>nul
copy /y "%BASE_OUT%\centerlines_wgs84.geojson" "%DEST%\centerlines_wgs84.geojson" >nul 2>nul
copy /y "%BASE_OUT%\intersections_wgs84.geojson" "%DEST%\intersections_wgs84.geojson" >nul 2>nul
copy /y "%BASE_OUT%\crs.json" "%DEST%\crs.json" >nul 2>nul
if /I "%ARM%"=="Arm1" (
  copy /y "%BASE_OUT%\osm_ref_roads.geojson" "%DEST%\osm_ref_roads.geojson" >nul 2>nul
  copy /y "%BASE_OUT%\osm_ref_metrics.json" "%DEST%\osm_ref_metrics.json" >nul 2>nul
)
if /I "%ARM%"=="Arm2" (
  if not exist "%DEST%\qgis_package" mkdir "%DEST%\qgis_package"
  copy /y "%BASE_OUT%\qgis_package\layers.json" "%DEST%\qgis_package\layers.json" >nul 2>nul
  copy /y "%BASE_OUT%\qgis_package\qgis_layers.md" "%DEST%\qgis_package\qgis_layers.md" >nul 2>nul
)
if /I "%ARM%"=="Arm3" (
  copy /y "%BASE_OUT%\osm_ref_roads.geojson" "%DEST%\osm_ref_roads.geojson" >nul 2>nul
  copy /y "%BASE_OUT%\osm_ref_metrics.json" "%DEST%\osm_ref_metrics.json" >nul 2>nul
  if not exist "%DEST%\qgis_package" mkdir "%DEST%\qgis_package"
  copy /y "%BASE_OUT%\qgis_package\layers.json" "%DEST%\qgis_package\layers.json" >nul 2>nul
  copy /y "%BASE_OUT%\qgis_package\qgis_layers.md" "%DEST%\qgis_package\qgis_layers.md" >nul 2>nul
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
set "IDX_OSM=%~5"
set "IDX_DOP=%~6"
for /f "usebackq delims=" %%t in (`powershell -NoProfile -Command "Get-Date -Format yyyy-MM-ddTHH:mm:ss"`) do set "IDX_TS=%%t"
.venv\Scripts\python.exe -c "import json,os; print(json.dumps({'arm_id':os.environ.get('ARM'), 'drive':os.environ.get('DRIVE'), 'status':os.environ.get('IDX_STATUS'), 'reason':os.environ.get('IDX_REASON') or None, 'base_geom_run_id':os.environ.get('IDX_RUN') or None, 'base_outputs_dir':os.environ.get('IDX_OUT') or None, 'osm_metrics_path':os.environ.get('IDX_OSM') or None, 'dop20_layers_path':os.environ.get('IDX_DOP') or None, 'timestamp':os.environ.get('IDX_TS')}))" >> "%INDEX_FILE%"
set "IDX_STATUS="
set "IDX_REASON="
set "IDX_RUN="
set "IDX_OUT="
set "IDX_OSM="
set "IDX_DOP="
set "IDX_TS="
goto :eof

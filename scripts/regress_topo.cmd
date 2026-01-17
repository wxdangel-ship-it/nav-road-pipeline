@echo off
setlocal enabledelayedexpansion
cd /d %~dp0\..

rem validate_topo_outputs.py supports --summary/--issues/--actions or env TOPO_*.

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

if "%MAX_FRAMES%"=="" set "MAX_FRAMES=2000"
if "%MAX_DRIVES%"=="" set "MAX_DRIVES=3"

set "DRIVES_LIST=%DRIVES%"
if not "%DRIVES_LIST%"=="" set "DRIVES_LIST=%DRIVES_LIST:,= %"

if "%DRIVES_LIST%"=="" (
  if "%DRIVES_FILE%"=="" set "DRIVES_FILE=configs\drives.txt"
  if exist "%DRIVES_FILE%" (
    for /f "usebackq delims=" %%d in ("%DRIVES_FILE%") do (
      set "LINE=%%d"
      if not "!LINE!"=="" if not "!LINE:~0,1!"=="#" set "DRIVES_LIST=!DRIVES_LIST! !LINE!"
    )
  )
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
if exist "%INDEX_FILE%" del "%INDEX_FILE%"

for %%d in (%DRIVES_LIST%) do (
  set "DRIVE=%%d"
  echo [REGRESS] drive=%%d

  call .\scripts\build_geom.cmd --drive %%d --max-frames %MAX_FRAMES%
  if errorlevel 1 exit /b 1

  call .\scripts\build_topo.cmd --drive %%d --max-frames %MAX_FRAMES%
  if errorlevel 1 exit /b 1

  for /f "usebackq delims=" %%p in (`
    .venv\Scripts\python.exe -c "from pathlib import Path; runs=Path('runs'); dirs=sorted([p for p in runs.iterdir() if p.is_dir() and p.name.startswith('topo_')], key=lambda p: p.stat().st_mtime); print(dirs[-1].name if dirs else '')"
  `) do set "TOPO_RUN=%%p"
  if "!TOPO_RUN!"=="" (
    echo [REGRESS] ERROR: could not find topo run for drive %%d
    exit /b 1
  )
  set "TOPO_OUT=runs\!TOPO_RUN!\outputs"

  call .\scripts\eval.cmd --max-frames %MAX_FRAMES%
  if errorlevel 1 exit /b 1

  .venv\Scripts\python.exe tools\validate_topo_outputs.py --summary "!TOPO_OUT!\TopoSummary.md" --issues "!TOPO_OUT!\TopoIssues.jsonl" --actions "!TOPO_OUT!\TopoActions.jsonl"
  if errorlevel 1 exit /b 1

  .venv\Scripts\python.exe -c "import json,os; print(json.dumps({'drive':os.environ.get('DRIVE'), 'topo_run_id':os.environ.get('TOPO_RUN'), 'topo_outputs':os.environ.get('TOPO_OUT'), 'summary_path':os.path.join(os.environ.get('TOPO_OUT',''), 'TopoSummary.md'), 'issues_path':os.path.join(os.environ.get('TOPO_OUT',''), 'TopoIssues.jsonl'), 'actions_path':os.path.join(os.environ.get('TOPO_OUT',''), 'TopoActions.jsonl')}))" >> "%INDEX_FILE%"
)

.venv\Scripts\python.exe tools\collect_topo_regress.py --regress-dir "%REGRESS_DIR%"
if errorlevel 1 exit /b 1

echo [REGRESS] DONE -> %REGRESS_DIR%
endlocal

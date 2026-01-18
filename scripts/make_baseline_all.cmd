@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "DRIVES_FILE=configs\golden_drives.txt"

set "BASELINE_MODE=update"
set "ENABLE_GATE=0"
set "BASELINE_PATH=configs\topo_regress_baseline.yaml"
call .\scripts\regress_topo.cmd
if errorlevel 1 exit /b %errorlevel%

set "GEOM_BACKEND=nn"
call .\scripts\regress_geom.cmd
if errorlevel 1 exit /b %errorlevel%

for /f "usebackq delims=" %%p in (`
  .venv\Scripts\python.exe -c "from pathlib import Path; runs=Path('runs'); dirs=sorted([p for p in runs.iterdir() if p.is_dir() and p.name.startswith('regress_geom_')], key=lambda p: p.stat().st_mtime); print(dirs[-1] if dirs else '')"
`) do set "REGRESS_DIR=%%p"

if "%REGRESS_DIR%"=="" (
  echo [BASELINE] ERROR: regress_geom run not found.
  exit /b 1
)

.venv\Scripts\python.exe tools\update_geom_baseline.py --regress-dir "%REGRESS_DIR%"
if errorlevel 1 exit /b %errorlevel%

echo [BASELINE] DONE
endlocal

@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "DRIVES_FILE=configs\golden_drives.txt"
set "GEOM_BACKEND=nn"

call .\scripts\regress_geom.cmd
if errorlevel 1 exit /b %errorlevel%

for /f "usebackq delims=" %%p in (`
  .venv\Scripts\python.exe -c "from pathlib import Path; runs=Path('runs'); dirs=sorted([p for p in runs.iterdir() if p.is_dir() and p.name.startswith('regress_geom_')], key=lambda p: p.stat().st_mtime); print(dirs[-1] if dirs else '')"
`) do set "REGRESS_DIR=%%p"

if "%REGRESS_DIR%"=="" (
  echo [GEOM-BASELINE] ERROR: regress_geom run not found.
  exit /b 1
)

.venv\Scripts\python.exe tools\update_geom_baseline.py --regress-dir "%REGRESS_DIR%"
if errorlevel 1 exit /b %errorlevel%

endlocal

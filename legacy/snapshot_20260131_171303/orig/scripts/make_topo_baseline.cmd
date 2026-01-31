@echo off
setlocal
cd /d %~dp0\..

set "DRIVES_FILE=configs\golden_drives.txt"
set "BASELINE_MODE=update"
set "BASELINE_PATH=configs\topo_regress_baseline.yaml"

call .\scripts\regress_topo.cmd
if errorlevel 1 exit /b %errorlevel%

endlocal

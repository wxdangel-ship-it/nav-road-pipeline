@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

.venv\Scripts\python.exe tools\run_full_geom_roadseg.py %*
endlocal

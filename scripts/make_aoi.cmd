@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

.venv\Scripts\python.exe tools\make_aoi_bbox.py %*
if errorlevel 1 exit /b %errorlevel%

endlocal

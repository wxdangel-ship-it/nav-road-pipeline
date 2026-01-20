@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "ARGS=%*"
.venv\Scripts\python.exe tools\build_image_feature_store.py %ARGS%
endlocal

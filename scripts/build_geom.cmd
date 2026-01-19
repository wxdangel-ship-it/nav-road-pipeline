@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "ARGS=%*"
if not "%CENTERLINES_CONFIG%"=="" (
  set "ARGS=%ARGS% --centerlines-config %CENTERLINES_CONFIG%"
)
.venv\Scripts\python.exe -m pipeline.build_geom %ARGS%
endlocal

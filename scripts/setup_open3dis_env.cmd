@echo off
setlocal
cd /d %~dp0\..

set "OPEN3DIS_ENV=cache\\open3dis_env"

if not exist "%OPEN3DIS_ENV%\\Scripts\\python.exe" (
  python -m venv "%OPEN3DIS_ENV%"
)

"%OPEN3DIS_ENV%\\Scripts\\python.exe" -m pip install -U pip
if exist requirements-open3dis.txt (
  "%OPEN3DIS_ENV%\\Scripts\\python.exe" -m pip install -r requirements-open3dis.txt
) else (
  echo requirements-open3dis.txt not found.
)

endlocal

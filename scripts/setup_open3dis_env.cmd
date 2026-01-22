@echo off
setlocal
cd /d %~dp0\..

set "OPEN3DIS_ENV=cache\\env_open3dis"
set "OPEN3DIS_PY=%OPEN3DIS_ENV%\\Scripts\\python.exe"
set "OPEN3DIS_PY_TXT=cache\\open3dis_python_path.txt"

if not exist "cache" (
  mkdir cache
)
set "TMP=%CD%"
set "TEMP=%CD%"

if not exist "%OPEN3DIS_PY%" (
  python -m venv "%OPEN3DIS_ENV%"
)

"%OPEN3DIS_PY%" -m pip --version >nul 2>nul
if errorlevel 1 (
  python -m pip install --upgrade pip --target "%OPEN3DIS_ENV%\\Lib\\site-packages"
)

"%OPEN3DIS_PY%" -m pip install -U pip
if exist requirements-open3dis.txt (
  "%OPEN3DIS_PY%" -m pip install -r requirements-open3dis.txt
) else (
  echo requirements-open3dis.txt not found.
)

echo %OPEN3DIS_PY%> "%OPEN3DIS_PY_TXT%"
echo Open3DIS Python: %OPEN3DIS_PY%

endlocal

@echo off
setlocal
cd /d %~dp0\..

set "OPEN3DIS_ENV=cache\\env_open3dis"
set "OPEN3DIS_PY=%OPEN3DIS_ENV%\\Scripts\\python.exe"
set "OPEN3DIS_PY_TXT=cache\\open3dis_python_path.txt"
set "OPEN3DIS_READY=cache\\open3dis_env_ready.txt"
set "OPEN3DIS_TMP=%CD%\\cache\\tmp"
set "OPEN3DIS_PIP_CACHE=%CD%\\cache\\pip_cache"

if not exist "cache" (
  mkdir cache
)
set "TMP=%CD%"
set "TEMP=%CD%"

if not exist "cache\\tmp" (
  mkdir "cache\\tmp"
)
if not exist "cache\\pip_cache" (
  mkdir "cache\\pip_cache"
)
set "TMP=%OPEN3DIS_TMP%"
set "TEMP=%OPEN3DIS_TMP%"
set "PIP_CACHE_DIR=%OPEN3DIS_PIP_CACHE%"

if not exist "%OPEN3DIS_PY%" (
  python -m venv "%OPEN3DIS_ENV%"
)

if exist "%OPEN3DIS_READY%" (
  goto :write_path
)

"%OPEN3DIS_PY%" -c "import numpy, pyproj, shapely, yaml" >nul 2>nul
if not errorlevel 1 (
  echo ready> "%OPEN3DIS_READY%"
  goto :write_path
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

echo ready> "%OPEN3DIS_READY%"

:write_path
echo %OPEN3DIS_PY%> "%OPEN3DIS_PY_TXT%"
echo Open3DIS Python: %OPEN3DIS_PY%

endlocal

@echo off
setlocal enableextensions

set "PYTHONUTF8=1"
set "PYTHONPATH=%cd%"

set "PYEXE=%cd%\\.venv\\Scripts\\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

echo [LIDAR-SEM] python=%PYEXE%
"%PYEXE%" -m scripts.run_lidar_semantic_golden8
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo [LIDAR-SEM] FAILED with exit code %EXITCODE%
  exit /b %EXITCODE%
)
echo [LIDAR-SEM] OK
exit /b 0


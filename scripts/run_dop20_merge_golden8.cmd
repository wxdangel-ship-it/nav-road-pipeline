@echo off
setlocal enableextensions

set "PYTHONUTF8=1"
set "PYTHONPATH=%cd%"

set "PYEXE=%cd%\\.venv\\Scripts\\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

echo [DOP20-MERGE] python=%PYEXE%
"%PYEXE%" -m scripts.run_dop20_merge_golden8
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo [DOP20-MERGE] FAILED with exit code %EXITCODE%
  exit /b %EXITCODE%
)
echo [DOP20-MERGE] OK
exit /b 0


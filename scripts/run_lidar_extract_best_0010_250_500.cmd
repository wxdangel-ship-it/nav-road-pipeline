@echo off
setlocal
set PYTHONUTF8=1
set PYTHONPATH=%cd%
if exist .venv\Scripts\python.exe (
  set PYEXE=.venv\Scripts\python.exe
) else (
  set PYEXE=python
)
"%PYEXE%" -m scripts.run_lidar_extract_best_0010_250_500
if errorlevel 1 exit /b 1
exit /b 0

@echo off
setlocal enabledelayedexpansion
set "PYTHONUTF8=1"
set "PYTHONPATH=%cd%"
if exist .venv\Scripts\python.exe (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)
"%PY%" -m scripts.run_proj_patch_0010_frames_250_341_500
if errorlevel 1 exit /b 1

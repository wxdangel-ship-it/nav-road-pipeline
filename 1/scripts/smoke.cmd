@echo off
setlocal enabledelayedexpansion

REM 一键 smoke（分钟级）
REM 说明：默认调用 .venv\Scripts\python.exe
REM 运行：scripts\smoke.cmd

set PYTHON=.venv\Scripts\python.exe

if not exist "%PYTHON%" (
  echo [ERROR] Python interpreter not found: %PYTHON%
  echo         Please create venv: py -3.11 -m venv .venv
  exit /b 1
)

"%PYTHON%" scripts\run_smoke.py
if errorlevel 1 (
  echo [ERROR] smoke failed.
  exit /b 1
)

echo [OK] done.

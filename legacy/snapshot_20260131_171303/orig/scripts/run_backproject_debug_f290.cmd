@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

.venv\Scripts\python.exe scripts\run_backproject_debug_f290.py

endlocal

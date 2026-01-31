@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

.venv\Scripts\python.exe scripts\run_backproject_cycle_gate_0010_frame290.py

endlocal

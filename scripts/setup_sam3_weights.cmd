@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

.venv\Scripts\python.exe tools\setup_sam3_weights.py %*

endlocal

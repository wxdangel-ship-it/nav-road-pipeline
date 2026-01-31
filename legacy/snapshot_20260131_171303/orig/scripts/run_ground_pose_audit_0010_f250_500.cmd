@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set PYTHONUTF8=1
set PYTHONPATH=%cd%

.venv\Scripts\python.exe -m scripts.run_ground_pose_audit_0010_f250_500 %*
if errorlevel 1 exit /b %errorlevel%

endlocal

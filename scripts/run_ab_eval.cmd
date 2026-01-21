@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "ARGS=%*"
if "%ARGS%"=="" (
  .venv\Scripts\python.exe tools\ab_eval_image_evidence.py
) else (
  .venv\Scripts\python.exe tools\ab_eval_image_evidence.py %ARGS%
)

endlocal

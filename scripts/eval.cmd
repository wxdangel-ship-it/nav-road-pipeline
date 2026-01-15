@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

REM Usage:
REM   eval.cmd               -> eval active.yaml
REM   eval.cmd path\to\cfg   -> eval that cfg

if "%~1"=="" (
  .venv\Scripts\python.exe -m pipeline.eval_all --config "configs\active.yaml"
) else (
  .venv\Scripts\python.exe -m pipeline.eval_all --config "%~1"
)

endlocal

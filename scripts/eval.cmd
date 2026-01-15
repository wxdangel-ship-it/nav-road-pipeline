@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

REM Defaults
set CFG=configs\active.yaml
set DATA=%POC_DATA_ROOT%
set PRIOR=%POC_PRIOR_ROOT%
set MAXF=

:parse
if "%~1"=="" goto run
if /I "%~1"=="--config" (set CFG=%~2 & shift & shift & goto parse)
if /I "%~1"=="--data-root" (set DATA=%~2 & shift & shift & goto parse)
if /I "%~1"=="--prior-root" (set PRIOR=%~2 & shift & shift & goto parse)
if /I "%~1"=="--max-frames" (set MAXF=%~2 & shift & shift & goto parse)
shift
goto parse

:run
if "%DATA%"=="" (
  echo ERROR: POC_DATA_ROOT not set. In PowerShell: $env:POC_DATA_ROOT="E:\KITTI360\KITTI-360"
  exit /b 2
)

if "%MAXF%"=="" (
  if "%PRIOR%"=="" (
    .venv\Scripts\python.exe -m pipeline.eval_all --config "%CFG%" --data-root "%DATA%"
  ) else (
    .venv\Scripts\python.exe -m pipeline.eval_all --config "%CFG%" --data-root "%DATA%" --prior-root "%PRIOR%"
  )
) else (
  if "%PRIOR%"=="" (
    .venv\Scripts\python.exe -m pipeline.eval_all --config "%CFG%" --data-root "%DATA%" --max-frames %MAXF%
  ) else (
    .venv\Scripts\python.exe -m pipeline.eval_all --config "%CFG%" --data-root "%DATA%" --prior-root "%PRIOR%" --max-frames %MAXF%
  )
)

endlocal

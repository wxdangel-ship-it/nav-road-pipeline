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
set DRIVES=
set INDEX=
set EVALMODE=
set DRIVE=

:parse
if "%~1"=="" goto run
if /I "%~1"=="--config" (set CFG=%~2 & shift & shift & goto parse)
if /I "%~1"=="--data-root" (set DATA=%~2 & shift & shift & goto parse)
if /I "%~1"=="--prior-root" (set PRIOR=%~2 & shift & shift & goto parse)
if /I "%~1"=="--max-frames" (set MAXF=%~2 & shift & shift & goto parse)
if /I "%~1"=="--drives" (set DRIVES=%~2 & shift & shift & goto parse)
if /I "%~1"=="--index" (set INDEX=%~2 & shift & shift & goto parse)
if /I "%~1"=="--eval-mode" (set EVALMODE=%~2 & shift & shift & goto parse)
if /I "%~1"=="--drive" (set DRIVE=%~2 & shift & shift & goto parse)
shift
goto parse

:run
if "%DATA%"=="" (
  echo ERROR: POC_DATA_ROOT not set. In PowerShell: $env:POC_DATA_ROOT="E:\KITTI360\KITTI-360"
  exit /b 2
)

set CMD=.venv\Scripts\python.exe -m pipeline.eval_all --config "%CFG%" --data-root "%DATA%"
if not "%PRIOR%"=="" set CMD=%CMD% --prior-root "%PRIOR%"
if not "%MAXF%"=="" set CMD=%CMD% --max-frames %MAXF%
if not "%DRIVES%"=="" set CMD=%CMD% --drives "%DRIVES%"
if not "%INDEX%"=="" set CMD=%CMD% --index "%INDEX%"
if not "%EVALMODE%"=="" set CMD=%CMD% --eval-mode "%EVALMODE%"
if not "%DRIVE%"=="" set CMD=%CMD% --drive "%DRIVE%"
call %CMD%

endlocal

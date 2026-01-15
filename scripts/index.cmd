@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

REM Usage:
REM   index.cmd --max-frames 2000
REM   index.cmd --data-root E:\KITTI360\KITTI-360 --out cache\kitti360_index.json --max-frames 2000

set DATA=%POC_DATA_ROOT%
set OUT=cache\kitti360_index.json
set MAXF=0

:parse
if "%~1"=="" goto run
if /I "%~1"=="--data-root" (set DATA=%~2 & shift & shift & goto parse)
if /I "%~1"=="--out" (set OUT=%~2 & shift & shift & goto parse)
if /I "%~1"=="--max-frames" (set MAXF=%~2 & shift & shift & goto parse)
shift
goto parse

:run
if "%DATA%"=="" (
  echo ERROR: POC_DATA_ROOT not set.
  exit /b 2
)

.venv\Scripts\python.exe -m pipeline.build_index --data-root "%DATA%" --out "%OUT%" --max-frames %MAXF%
endlocal

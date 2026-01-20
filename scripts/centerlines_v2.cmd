@echo off
setlocal enabledelayedexpansion
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "ARGS="
set "NEXT_IS_STORE=0"
for %%A in (%*) do (
  if "!NEXT_IS_STORE!"=="1" (
    set "FEATURE_STORE_DIR=%%~A"
    set "NEXT_IS_STORE=0"
  ) else (
    if "%%~A"=="--feature-store" (
      set "NEXT_IS_STORE=1"
    ) else (
      set "ARGS=!ARGS! %%~A"
    )
  )
)

if not "%CENTERLINES_CONFIG%"=="" (
  set "ARGS=%ARGS% --centerlines-config %CENTERLINES_CONFIG%"
)

call .\scripts\build_geom.cmd %ARGS%
endlocal

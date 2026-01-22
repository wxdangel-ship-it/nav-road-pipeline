@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "ARGS=%*"
if "%ARGS%"=="" (
  if "%FULL_GOLDEN8%"=="1" (
    if "!FULL_INDEX!"=="" (
      set "FULL_INDEX=runs\image_samples_golden8_full\sample_index.jsonl"
    )
    if not exist "!FULL_INDEX!" (
      if "!FULL_FRAMES_PER_DRIVE!"=="" (
        set "FULL_FRAMES_PER_DRIVE=0"
      )
      if "!FULL_STRIDE!"=="" (
        set "FULL_STRIDE=5"
      )
      .venv\Scripts\python.exe tools\build_image_sample_index.py --index runs\sweep_geom_postopt_20260119_061421\postopt_index.jsonl --out "!FULL_INDEX!" --frames-per-drive !FULL_FRAMES_PER_DRIVE! --stride !FULL_STRIDE!
    )
    .venv\Scripts\python.exe tools\run_image_providers.py --index "!FULL_INDEX!"
  ) else (
    call .\scripts\run_image_providers.cmd
  )
) else (
  call .\scripts\run_image_providers.cmd %ARGS%
)

call .\scripts\run_ab_eval.cmd

if "%MAP_EVAL%"=="1" (
  if "%MAP_A%"=="" (
    echo [MAP_EVAL] MAP_A not set. Skipping map eval.
    goto :done
  )
  if "%MAP_B%"=="" (
    echo [MAP_EVAL] MAP_B not set. Skipping map eval.
    goto :done
  )
  if "%MAP_DRIVE%"=="" (
    echo [MAP_EVAL] MAP_DRIVE not set. Skipping map eval.
    goto :done
  )
  if "%MAP_INDEX%"=="" (
    set "MAP_INDEX=runs\image_samples_golden8\sample_index.jsonl"
  )
  if "%MAP_EVAL_OUT%"=="" (
    set "MAP_EVAL_OUT=runs\map_eval_report.md"
  )
  .venv\Scripts\python.exe tools\ab_eval_map_evidence.py --map-a "%MAP_A%" --map-b "%MAP_B%" --drive "%MAP_DRIVE%" --frame-index "%MAP_INDEX%" --out "%MAP_EVAL_OUT%" --road-root "%ROAD_ROOT%"
)

:done
endlocal

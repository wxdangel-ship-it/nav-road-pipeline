@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set "FULL_STRIDE=5"
set "CROSSWALK_STAGE2=1"
set "CROSSWALK_MIN_FRAMES_HIT_FINAL=3"
set "CROSSWALK_MIN_FRAMES_HIT_COARSE=2"
set "QA_FRAMES_PER_ENTITY=20"
set "QA_TOPN_REJECT=30"

set "INDEX=runs\image_samples_golden8_full\sample_index.jsonl"
set "IMAGE_RUN=runs\image_evidence_20260122_114726"
set "IMAGE_PROVIDER=grounded_sam2_v1"
set "ROAD_ROOT=runs\full_stack_20260121_000548"
set "CONFIG=configs\road_entities.yaml"

if "%FULL_STRIDE%"=="" set "FULL_STRIDE=5"
if "%CROSSWALK_MIN_FRAMES_HIT_FINAL%"=="" set "CROSSWALK_MIN_FRAMES_HIT_FINAL=3"
if "%CROSSWALK_MIN_FRAMES_HIT_COARSE%"=="" set "CROSSWALK_MIN_FRAMES_HIT_COARSE=2"
if "%QA_FRAMES_PER_ENTITY%"=="" set "QA_FRAMES_PER_ENTITY=20"
if "%QA_TOPN_REJECT%"=="" set "QA_TOPN_REJECT=30"

.venv\Scripts\python.exe tools\run_crosswalk_full.py ^
  --image-run "%IMAGE_RUN%" ^
  --image-provider "%IMAGE_PROVIDER%" ^
  --road-root "%ROAD_ROOT%" ^
  --config "%CONFIG%" ^
  --index "%INDEX%" ^
  --full-stride %FULL_STRIDE% ^
  --stage2 %CROSSWALK_STAGE2% ^
  --min-frames-hit-final %CROSSWALK_MIN_FRAMES_HIT_FINAL% ^
  --min-frames-hit-coarse %CROSSWALK_MIN_FRAMES_HIT_COARSE% ^
  --qa-frames-per-entity %QA_FRAMES_PER_ENTITY% ^
  --qa-topn-reject %QA_TOPN_REJECT%

endlocal

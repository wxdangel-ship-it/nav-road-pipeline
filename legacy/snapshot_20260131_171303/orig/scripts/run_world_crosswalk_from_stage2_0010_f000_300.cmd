@echo off
setlocal

set "CFG=configs\world_crosswalk_from_stage2_0010_f000_300.yaml"

.venv\Scripts\python.exe scripts\run_world_crosswalk_from_stage2_0010_f000_300.py --config "%CFG%"

endlocal

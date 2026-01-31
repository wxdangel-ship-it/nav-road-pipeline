@echo off
setlocal

set "CFG=configs\image_crosswalk_stage12_to_world_0010_f000_500_nofilter.yaml"

.venv\Scripts\python.exe scripts\run_image_crosswalk_stage12_to_world_0010_f000_500_nofilter.py --config "%CFG%"

endlocal

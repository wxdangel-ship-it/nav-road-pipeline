@echo off
setlocal

set "CFG=configs\image_stage12_ensemble_gdino_yoloworld_0010_f000_300.yaml"

.venv\Scripts\python.exe scripts\run_image_stage12_ensemble_gdino_yoloworld_0010_f000_300.py --config "%CFG%"

endlocal

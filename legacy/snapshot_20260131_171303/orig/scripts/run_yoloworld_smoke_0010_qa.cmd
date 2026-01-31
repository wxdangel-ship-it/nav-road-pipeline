@echo off
setlocal

set "CFG=configs\yoloworld_smoke_0010.yaml"

.venv\Scripts\python.exe scripts\run_yoloworld_smoke_0010_qa.py --config "%CFG%"

endlocal

@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

set GEOM_BACKEND=nn
set GEOM_NN_FIXED=1
set GEOM_NN_BEST_CFG=configs\geom_nn_best.yaml

.venv\Scripts\python.exe tools\sweep_geom_postopt.py %*
endlocal

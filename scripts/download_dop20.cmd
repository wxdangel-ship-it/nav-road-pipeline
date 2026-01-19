@echo off
setlocal

if "%DOP20_ROOT%"=="" (
  set "DOP20_ROOT=E:\KITTI360\KITTI-360\_lglbw_dop20_golden8"
)
for /f "delims=" %%A in ("%DOP20_ROOT%") do set "DOP20_ROOT=%%A"

set "AOI=%~1"
if "%AOI%"=="" (
  set "AOI=runs\golden8_aoi\golden8_aoi.json"
)

set "WMS=https://owsproxy.lgl-bw.de/owsproxy/ows/WMS_LGL-BW_ATKIS_DOP_20_C?"
set "LAYER=IMAGES_DOP_20_RGB"
set "CRS=EPSG:25832"
set "TILE_SIZE=500"
set "RESOLUTION=0.2"
set "WORKERS=6"
set "RETRY=3"
set "TIMEOUT=60"

.venv\Scripts\python.exe tools\download_dop20_wms.py --aoi "%AOI%" --out-root "%DOP20_ROOT%" --tile-size-m %TILE_SIZE% --resolution-m %RESOLUTION% --crs %CRS% --wms "%WMS%" --layer %LAYER% --workers %WORKERS% --retry %RETRY% --timeout %TIMEOUT%
if errorlevel 1 exit /b %errorlevel%

echo [DOP20] done: %DOP20_ROOT%
echo [DOP20] next: set DOP20_ROOT and run SAT full
exit /b 0

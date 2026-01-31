@echo off
setlocal

REM Download OSM for current experiment area (auto-discover latest WGS84 polygon).
python "%~dp0..\tools\download_osm.py" %*
if errorlevel 1 exit /b %errorlevel%

endlocal

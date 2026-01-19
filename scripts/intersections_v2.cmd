@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

REM usage:
REM scripts\intersections_v2.cmd --index <postopt_index.jsonl> --stage quick|full --config <yaml> [--resume] [--candidate <id>] [--out-dir <runs\...>]

.venv\Scripts\python.exe tools\intersections_v2_runner.py %*
endlocal

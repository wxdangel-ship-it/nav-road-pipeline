@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

REM usage:
REM scripts\sat_intersections.cmd --index <postopt_index.jsonl> --stage quick|full --config <yaml> [--resume] [--finalize] [--out-dir <runs\sat_...>] [--candidate <id>] [--write-back]

.venv\Scripts\python.exe tools\sat_intersections_runner.py %*
endlocal

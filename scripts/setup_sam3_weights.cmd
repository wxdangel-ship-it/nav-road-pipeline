@echo off
setlocal
cd /d %~dp0\..

if not exist .venv\Scripts\python.exe (
  call .\scripts\setup.cmd
)

if "%HF_TOKEN%"=="" (
  echo [WARN] HF_TOKEN is not set. If you use gated weights, run: huggingface-cli login
  echo [WARN] Or set: set HF_TOKEN=your_token
)

.venv\Scripts\python.exe tools\setup_sam3_weights.py %*

endlocal

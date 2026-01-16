param(
  [string]$DataRoot = "",
  [string]$PriorRoot = ""
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path "$PSScriptRoot\.."
Set-Location $repo

if (!(Test-Path .\.venv)) { & .\scripts\setup.ps1 }

# 当前 eval_all 还没用到 DataRoot/PriorRoot（后续接入 Adapter 时会用）
.\.venv\Scripts\python.exe -m pipeline.eval_all

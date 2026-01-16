$ErrorActionPreference = "Stop"
$repo = Resolve-Path "$PSScriptRoot\.."
Set-Location $repo

if (!(Test-Path .\.venv)) {
  python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "[SETUP] OK" -ForegroundColor Green

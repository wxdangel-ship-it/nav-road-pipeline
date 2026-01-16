$ErrorActionPreference = "Stop"
$repo = Resolve-Path "$PSScriptRoot\.."
Set-Location $repo

if (!(Test-Path .\.venv)) { & .\scripts\setup.ps1 }

.\.venv\Scripts\python.exe -m pipeline.smoke

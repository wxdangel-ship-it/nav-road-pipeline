@echo off
setlocal EnableDelayedExpansion
cd /d %~dp0\..

set "FAIL=0"
set "FORBIDDEN=0"
set "WARN=0"
set "LOG_DIR=logs"
set "OUT_DIR=outputs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
set "DOC_LOG=%LOG_DIR%\precommit_doc_paths.txt"
if exist "%DOC_LOG%" del "%DOC_LOG%"

echo [PRECOMMIT] check staged forbidden artifacts...
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$paths = git diff --cached --name-only; $bad=@(); foreach($p in $paths){ if($p -match '^(runs/|cache/|tmp/|pip-|model_weights/|weights/)' -or $p -match '\\.(pth|pt|ckpt|onnx|engine)$'){ $bad += $p } }; $bad"`) do (
  echo [ERROR] forbidden staged: %%i
  set "FORBIDDEN=1"
)
if "%FORBIDDEN%"=="1" (
  echo [PRECOMMIT] failed: staged forbidden artifacts.
  exit /b 1
)

echo [PRECOMMIT] check untracked runtime artifacts...
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$lines = git status --porcelain; $warn=@(); foreach($l in $lines){ if($l -match '^\?\?\s+'){ $p=$l -replace '^\\?\\?\\s+',''; if($p -match '^(runs/|cache/|tmp/|pip-|model_weights/|weights/)' -or $p -match '\\.(pth|pt|ckpt|onnx|engine)$'){ $warn += $p } } }; $warn"`) do (
  echo [WARN] Detected untracked runtime artifacts; do not commit: %%i
  set "WARN=1"
)

echo [PRECOMMIT] check required scripts...
set "SCRIPT_MISSING=0"
for %%p in (
  "scripts/smoke.cmd"
  "scripts/eval.cmd"
  "scripts/run_crosswalk_strict_250_500.cmd"
  "scripts/run_roundtrip_report_v2.cmd"
  "scripts/run_failpack_summary.cmd"
  "scripts/run_crosswalk_autotune_0010_280_300.cmd"
) do (
  if not exist %%p (
    echo [ERROR] MISSING %%~p
    set "SCRIPT_MISSING=1"
  )
)
if "%SCRIPT_MISSING%"=="1" (
  set "FAIL=1"
)

echo [PRECOMMIT] scan docs for scripts paths...
powershell -NoProfile -Command ^
  "$files = @('README.md','SPEC.md') + (Get-ChildItem docs -Filter *.md | ForEach-Object { $_.FullName });" ^
  "$pattern = 'scripts/[A-Za-z0-9_./\\-]+\\.cmd';" ^
  "$paths = @();" ^
  "foreach($f in $files){ if(Test-Path $f){ foreach($line in Get-Content $f){ foreach($m in [regex]::Matches($line, $pattern)){ $paths += $m.Value } } } }" ^
  "$paths = $paths | Sort-Object -Unique;" ^
  "$out = @();" ^
  "foreach($p in $paths){ if(Test-Path $p){ $out += \"OK $p\" } else { $out += \"MISSING $p\" } }" ^
  "$out | Set-Content -Encoding ascii '%DOC_LOG%';"
echo [PRECOMMIT] wrote %DOC_LOG%

for /f "usebackq delims=" %%i in (`git rev-parse --abbrev-ref HEAD`) do set "BRANCH=%%i"
for /f "usebackq delims=" %%i in (`git rev-parse --short HEAD`) do set "COMMIT=%%i"
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "(git diff --cached --name-only | Measure-Object).Count"`) do set "STAGED_COUNT=%%i"

set "FORBIDDEN_STATUS=PASS"
if "%FORBIDDEN%"=="1" set "FORBIDDEN_STATUS=FAIL"
set "SCRIPT_STATUS=PASS"
if "%SCRIPT_MISSING%"=="1" set "SCRIPT_STATUS=FAIL"

set "SUMMARY=%OUT_DIR%\precommit_summary.md"
(
  echo # Precommit Summary
  echo.
  echo - branch: %BRANCH%
  echo - commit: %COMMIT%
  echo - staged_files: %STAGED_COUNT%
  echo - forbidden_check: %FORBIDDEN_STATUS%
  echo - required_scripts: %SCRIPT_STATUS%
  echo - doc_paths_log: %DOC_LOG%
) > "%SUMMARY%"
echo [PRECOMMIT] wrote %SUMMARY%

if "%FAIL%"=="1" exit /b 1
exit /b 0
endlocal

# Run from any working directory using this project's own environment.
$ErrorActionPreference = 'Stop'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.cache\matplotlib'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Python environment missing. Follow the setup steps in README.md.'
}
& $taskPython (Join-Path $PSScriptRoot 'main.py') @args
exit $LASTEXITCODE

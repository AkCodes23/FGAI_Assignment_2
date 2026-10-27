# Legacy entrypoint retained for compatibility. Training is already complete;
# this now runs only the fixed-sample inference and reporting pipeline.
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "run_all.ps1")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

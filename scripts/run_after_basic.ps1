# Legacy entrypoint retained for compatibility; the reduced pipeline reuses existing checkpoints.
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "run_all.ps1")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

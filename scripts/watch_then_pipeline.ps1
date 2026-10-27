# Wait for the retrained basic Transformer test pass, then run the decoder-only
# and evaluation pipeline. Times out rather than starting on stale artefacts.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$log = Join-Path $root "logs\watch_pipeline.log"
function Log([string]$m) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
    Add-Content -Path $log -Value $line -Encoding utf8
}
$start = Get-Date
$metrics = Join-Path $root "results\metrics_basic_transformer.json"
$deadline = $start.AddMinutes(150)
Log "waiting for a fresh basic-transformer test result (launch $($start.ToString('HH:mm:ss')))"
while ((Get-Date) -lt $deadline) {
    $fresh = (Test-Path $metrics) -and ((Get-Item $metrics).LastWriteTime -gt $start)
    $busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*01_train_transformer*' -or $_.CommandLine -like '*01b_infer_transformer*' }
    if ($fresh -and -not $busy) {
        Log "basic result is fresh and no basic python process is running"
        & powershell.exe -NoLogo -NoProfile -NonInteractive -File (Join-Path $root "scripts\run_after_basic.ps1")
        Log "run_after_basic.ps1 exit=$LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Start-Sleep -Seconds 30
}
Log "timed out without a fresh basic result; pipeline not started"
exit 2

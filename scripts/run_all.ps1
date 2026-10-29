# Reduced-scope evaluation pipeline. Uses existing model checkpoints; does not train models or redownload data.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $root "src"
$env:HF_HOME = Join-Path $root ".hf"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONUTF8 = "1"
$env:TOKENIZERS_PARALLELISM = "false"

function Invoke-Py {
    param([string[]]$Arguments)
    Write-Host ">> python $($Arguments -join ' ')"
    & $py @Arguments
    if ($LASTEXITCODE -ne 0) { throw "python $($Arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

foreach ($checkpoint in @("models\basic_transformer\best.pt", "models\mt5_base\best\config.json")) {
    if (-not (Test-Path (Join-Path $root $checkpoint))) { throw "Required trained checkpoint missing: $checkpoint" }
}

Invoke-Py @("scripts\00b_select_eval_sample.py")
Invoke-Py @("scripts\02b_infer_mt5.py")
Invoke-Py @("scripts\01b_infer_transformer.py")
Invoke-Py @("scripts\03_llama.py", "--mode", "prompt", "--shots", "3")
Invoke-Py @("scripts\04_gemma.py", "--mode", "prompt", "--shots", "3")
Invoke-Py @("scripts\05_evaluate.py")
Invoke-Py @("scripts\06_temperature.py", "--limit", "20")
Invoke-Py @("scripts\07_ethics_analysis.py")
Invoke-Py @("scripts\14_sync_parameter_counts.py")
Invoke-Py @("scripts\10_collect_facts.py")
Invoke-Py @("scripts\11_fill_results_table.py")
Invoke-Py @("scripts\12_qualitative_samples.py")
Invoke-Py @("scripts\15_eda.py")
foreach ($privateScript in @("09_make_submission_doc.py", "13_make_submission_pdf.py")) {
    $scriptPath = Join-Path $root ("scripts\" + $privateScript)
    if (Test-Path $scriptPath) {
        Invoke-Py @("scripts\" + $privateScript)
    } else {
        Write-Host "Skipping local, identity-bearing submission document generation."
    }
}
Invoke-Py @("scripts\08_check_artifacts.py")

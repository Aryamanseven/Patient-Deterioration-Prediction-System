param(
    [string]$ConfigPath = "configs/default.yaml",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "[PS-2] Project root: $ProjectRoot"
Write-Host "[PS-2] Python requirement: 3.10"

if (-not $SkipInstall) {
    Write-Host "[PS-2] Installing requirements..."
    py -3.10 -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Dependency install failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

Write-Host "[PS-2] Running full pipeline with $ConfigPath"
py -3.10 pipelines/run_full_pipeline.py --config $ConfigPath
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pipeline run failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "[PS-2] Running artifact audit"
py -3.10 pipelines/audit_artifacts.py --artifacts-dir artifacts
if ($LASTEXITCODE -ne 0) {
    Write-Error "Artifact audit failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "[PS-2] Submission pipeline completed successfully."
exit 0

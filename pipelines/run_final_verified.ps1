param(
    [string]$ConfigPath = "configs/final_full_replay.yaml"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$PythonExe = "C:/Users/sharm/AppData/Local/Programs/Python/Python310/python.exe"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$launcherLog = "artifacts/final_run_launcher_${stamp}.log"
$auditJson = "artifacts/final_audit_${stamp}.json"
$summaryJson = "artifacts/final_summary_${stamp}.json"

Write-Host "[FINAL] Project root: $ProjectRoot"
Write-Host "[FINAL] Python: $PythonExe"
Write-Host "[FINAL] Config: $ConfigPath"

$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "python.exe" -and $_.CommandLine -match "run_full_pipeline.py"
}
foreach ($proc in $active) {
    Write-Host "[FINAL] Stopping existing pipeline process PID=$($proc.ProcessId)"
    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
}

Write-Host "[FINAL] Running full pipeline..."
& $PythonExe pipelines/run_full_pipeline.py --config $ConfigPath 2>&1 | Tee-Object -FilePath $launcherLog
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pipeline failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "[FINAL] Auditing artifacts..."
& $PythonExe pipelines/audit_artifacts.py --artifacts-dir artifacts --json-out $auditJson 2>&1 | Tee-Object -FilePath $launcherLog -Append
if ($LASTEXITCODE -ne 0) {
    Write-Error "Artifact audit failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

$env:FINAL_SUMMARY_JSON = $summaryJson

$summaryScript = @'
import json
import os
from pathlib import Path

artifacts = Path("artifacts")
runs = sorted([d for d in artifacts.iterdir() if d.is_dir() and d.name.startswith("run_")], key=lambda p: p.stat().st_mtime, reverse=True)
if not runs:
    raise SystemExit("No run folders found.")
latest = runs[0]

summary = {
    "latest_run": latest.name,
    "latest_run_path": str(latest),
    "metrics_exists": (latest / "metrics.json").exists(),
    "predictions_exists": (latest / "predictions.csv").exists(),
    "core_model_exists": (latest / "model" / "model.cbm").exists(),
    "dl_model_exists": (latest / "model" / "dl_model_final.pt").exists(),
    "ensemble_exists": (latest / "model" / "ensemble.pkl").exists(),
}

if (latest / "metrics.json").exists():
    try:
        summary["metrics"] = json.loads((latest / "metrics.json").read_text(encoding="utf-8"))
    except Exception as exc:
        summary["metrics_error"] = str(exc)

Path(os.environ["FINAL_SUMMARY_JSON"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
'@

$summaryScript | & $PythonExe - 2>&1 | Tee-Object -FilePath $launcherLog -Append
if ($LASTEXITCODE -ne 0) {
    Write-Error "Summary generation failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "[FINAL] Completed successfully."
Write-Host "[FINAL] Launcher log: $launcherLog"
Write-Host "[FINAL] Audit JSON : $auditJson"
Write-Host "[FINAL] Summary JSON: $summaryJson"
exit 0

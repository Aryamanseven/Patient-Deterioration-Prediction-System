param(
    [string]$PythonExe = "C:/Users/sharm/AppData/Local/Programs/Python/Python310/python.exe",
    [string]$ProjectRoot = "D:/nEXU2.0 TRY/Patient-Deterioration-Prediction-System",
    [int]$Stage1MaxRelaunch = 1,
    [int]$Stage2MaxRelaunch = 1
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$logPath = Join-Path $ProjectRoot "artifacts/overnight_supervisor.log"

if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format s), $Message
    $line | Out-File -FilePath $logPath -Encoding utf8 -Append
}

function Test-PipelineRunning {
    param([string]$ConfigName)
    $needle = "pipelines/run_full_pipeline.py --config configs/$ConfigName"
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "python.exe" -and $_.CommandLine -match [regex]::Escape($needle)
    }
    return ($null -ne $procs)
}

function Get-LatestRunBySuffix {
    param([string]$Suffix)
    $artifactsDir = Join-Path $ProjectRoot "artifacts"
    if (-not (Test-Path $artifactsDir)) {
        return $null
    }

    $dirs = Get-ChildItem -Path $artifactsDir -Directory | Where-Object {
        $_.Name -like "run_*_$Suffix"
    } | Sort-Object LastWriteTime -Descending

    if ($dirs.Count -eq 0) {
        return $null
    }

    return $dirs[0].FullName
}

function Test-RequiredArtifacts {
    param(
        [string]$RunDir,
        [string]$Profile = "stage1"
    )

    $required = @(
        "model/dl_model_final.pt",
        "model/model.cbm",
        "model/scaler.pkl",
        "model/ensemble.pkl",
        "model/feature_columns.json",
        "metrics.json",
        "predictions.csv",
        "ssl_pretrained_tcntransformer.pt"
    )

    if ($Profile -eq "stage2") {
        $required += @(
            "fl_rounds_history.json",
            "lodo_results.csv",
            "top_features.csv",
            "shap_summary.png",
            "captum_temporal_heatmap.png"
        )
    }

    $missing = @()
    foreach ($rel in $required) {
        $path = Join-Path $RunDir $rel
        if (-not (Test-Path $path)) {
            $missing += $rel
            continue
        }

        try {
            $info = Get-Item $path
            if ($info.Length -le 0) {
                $missing += $rel
            }
        } catch {
            $missing += $rel
        }
    }

    return [PSCustomObject]@{
        Complete = ($missing.Count -eq 0)
        Missing = $missing
    }
}

function Invoke-PythonLogged {
    param(
        [string]$Context,
        [string[]]$Arguments
    )

    $exitCode = -1
    $stdoutPath = Join-Path $ProjectRoot "artifacts/supervisor_python_stdout.tmp.log"
    $stderrPath = Join-Path $ProjectRoot "artifacts/supervisor_python_stderr.tmp.log"

    Remove-Item $stdoutPath -Force -ErrorAction SilentlyContinue
    Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue

    try {
        $proc = Start-Process -FilePath $PythonExe -ArgumentList $Arguments -WorkingDirectory $ProjectRoot -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

        $exitCode = $proc.ExitCode
    } catch {
        Write-Log "$Context failed to launch via Start-Process: $($_.Exception.Message)"
    }

    if (Test-Path $stdoutPath) {
        Get-Content $stdoutPath | Add-Content -Path $logPath -Encoding utf8
    }
    if (Test-Path $stderrPath) {
        Get-Content $stderrPath | Add-Content -Path $logPath -Encoding utf8
    }

    Remove-Item $stdoutPath -Force -ErrorAction SilentlyContinue
    Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue

    Write-Log "$Context ended with exit code $exitCode."
    return $exitCode
}

Write-Log "Supervisor started. Python=$PythonExe"

$stage1Suffix = "retrain_from_ssl_strict_save"
$stage2Suffix = "stage2_fl10_from_recovered_dl"

$stage1Attempts = 0
$stage2Attempts = 0

# Stage 1 guard: wait for current run or relaunch if needed.
while ($true) {
    if (Test-PipelineRunning -ConfigName "recovery.yaml") {
        Write-Log "Stage 1 is running. Waiting 120s."
        Start-Sleep -Seconds 120
        continue
    }

    $latestStage1 = Get-LatestRunBySuffix -Suffix $stage1Suffix
    if ($null -eq $latestStage1) {
        if ($stage1Attempts -gt $Stage1MaxRelaunch) {
            Write-Log "Stage 1 run dir not found and relaunch limit reached. Exiting with failure."
            exit 1
        }

        $stage1Attempts += 1
        Write-Log "No Stage 1 run dir found. Launching Stage 1 attempt $stage1Attempts."
        [void](Invoke-PythonLogged -Context "Stage 1 attempt $stage1Attempts" -Arguments @("pipelines/run_full_pipeline.py", "--config", "configs/recovery.yaml"))
        continue
    }

    $check = Test-RequiredArtifacts -RunDir $latestStage1 -Profile "stage1"
    if ($check.Complete) {
        Write-Log "Stage 1 complete with required artifacts: $latestStage1"
        break
    }

    if ($stage1Attempts -gt $Stage1MaxRelaunch) {
        Write-Log "Stage 1 incomplete after relaunch limit. Missing: $($check.Missing -join ', ')"
        exit 1
    }

    $stage1Attempts += 1
    Write-Log "Stage 1 incomplete. Missing: $($check.Missing -join ', '). Relaunching attempt $stage1Attempts."
    [void](Invoke-PythonLogged -Context "Stage 1 attempt $stage1Attempts" -Arguments @("pipelines/run_full_pipeline.py", "--config", "configs/recovery.yaml"))
}

Write-Log "Running artifact audit after Stage 1."
$auditStage1Exit = Invoke-PythonLogged -Context "Artifact audit after Stage 1" -Arguments @("pipelines/audit_artifacts.py", "--artifacts-dir", "artifacts")
Write-Log "Artifact audit completed after Stage 1 with exit code $auditStage1Exit."

# Stage 2 launch and guard.
while ($true) {
    if (Test-PipelineRunning -ConfigName "recovery_fl_stage2_10rounds.yaml") {
        Write-Log "Stage 2 is already running. Waiting 120s."
        Start-Sleep -Seconds 120
        continue
    }

    if ($stage2Attempts -gt $Stage2MaxRelaunch) {
        $latestStage2 = Get-LatestRunBySuffix -Suffix $stage2Suffix
        if ($null -ne $latestStage2) {
            $check2 = Test-RequiredArtifacts -RunDir $latestStage2 -Profile "stage2"
            if ($check2.Complete) {
                Write-Log "Stage 2 complete after retries: $latestStage2"
                break
            }
        }

        Write-Log "Stage 2 relaunch limit reached without complete artifacts. Exiting with failure."
        exit 1
    }

    $stage2Attempts += 1
    Write-Log "Launching Stage 2 attempt $stage2Attempts."
    [void](Invoke-PythonLogged -Context "Stage 2 attempt $stage2Attempts" -Arguments @("pipelines/run_full_pipeline.py", "--config", "configs/recovery_fl_stage2_10rounds.yaml"))

    $latestStage2 = Get-LatestRunBySuffix -Suffix $stage2Suffix
    if ($null -eq $latestStage2) {
        Write-Log "Stage 2 run dir not found yet."
        continue
    }

    $check2 = Test-RequiredArtifacts -RunDir $latestStage2 -Profile "stage2"
    if ($check2.Complete) {
        Write-Log "Stage 2 complete with required artifacts: $latestStage2"
        break
    }

    Write-Log "Stage 2 incomplete. Missing: $($check2.Missing -join ', ')"
}

Write-Log "Running artifact audit after Stage 2."
$auditStage2Exit = Invoke-PythonLogged -Context "Artifact audit after Stage 2" -Arguments @("pipelines/audit_artifacts.py", "--artifacts-dir", "artifacts")
Write-Log "Artifact audit completed after Stage 2 with exit code $auditStage2Exit."
Write-Log "Supervisor finished successfully."
exit 0

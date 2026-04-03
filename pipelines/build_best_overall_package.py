"""
Build a judge-facing best-overall submission package.

This script creates a clean folder with:
- latest complete run artifacts
- evidence and log highlights
- submission notebooks and CSV outputs
- docs, configs, and reproducibility scripts
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUBMISSION_FILES = [
    "README.md",
    "Patient_Deterioration_Week1_Official_Submission_Notebook.ipynb",
    "Best_Overall_Submission_Walkthrough.ipynb",
    "Reproducible_EndToEnd_Runbook.ipynb",
    "NOTEBOOK_SUBMISSION_GUIDE.md",
    "PITCH_SCRIPT.md",
    "week1_official_submission_results.csv",
    "official_winner_reproduced_metrics.csv",
    "focused_subsample_lr0048_iter1450_official_submission_predictions.csv",
    "AesCodeNexus_Round1_Submission_Final_Technical_Patient_Deterioration_TeamReady.pptx",
]

DOC_FILES = [
    "README.md",
    "REPO_NAVIGATION.md",
    "ANC052_STRENGTHS_WEAKNESSES.md",
    "ANC052_COMPETITIVE_EDGE.md",
    "TRAINING_LOG_CONTEXT.md",
    "COMPETITION_FEEDBACK_SNAPSHOT.md",
    "BEST_OVERALL_NEXT_STEPS.md",
    "ARCHITECTURE.md",
    "MODULE_CONNECTIONS.md",
    "TECHNICAL_REPORT.md",
    "INNOVATION_FAQ.md",
]

CONFIG_FILES = [
    "README.md",
    "default.yaml",
    "final_full_replay.yaml",
]

PIPELINE_FILES = [
    "README.md",
    "run_full_pipeline.py",
    "run_submission_pipeline.ps1",
    "run_final_verified.ps1",
    "audit_artifacts.py",
    "export_run_evidence.py",
    "build_best_overall_package.py",
]

CORE_REQUIRED_ARTIFACTS = [
    "model/dl_model_final.pt",
    "model/model.cbm",
    "model/scaler.pkl",
    "model/ensemble.pkl",
    "model/feature_columns.json",
    "metrics.json",
    "predictions.csv",
]


def copy_file(src: Path, dst: Path, copied: list[str], missing: list[str]) -> None:
    if src.exists() and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(src))
    else:
        missing.append(str(src))


def copy_relative_files(
    repo_root: Path,
    src_folder: str,
    rel_paths: list[str],
    package_root: Path,
    package_folder: str,
    copied: list[str],
    missing: list[str],
) -> None:
    for rel in rel_paths:
        src = repo_root / src_folder / rel
        dst = package_root / package_folder / rel
        copy_file(src, dst, copied, missing)


def pick_latest_complete_run(artifacts_dir: Path) -> Path:
    run_dirs = sorted(
        [d for d in artifacts_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        if (run_dir / "metrics.json").exists() and (run_dir / "predictions.csv").exists():
            return run_dir
    raise FileNotFoundError("No complete run found with metrics.json and predictions.csv.")


def parse_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "ensemble" in data:
        return {
            "deep_learning_pr_auc": data.get("deep_learning", {}).get("pr_auc"),
            "catboost_pr_auc": data.get("catboost", {}).get("pr_auc"),
            "ensemble_pr_auc": data.get("ensemble", {}).get("pr_auc"),
            "deep_learning_roc_auc": data.get("deep_learning", {}).get("roc_auc"),
            "catboost_roc_auc": data.get("catboost", {}).get("roc_auc"),
            "ensemble_roc_auc": data.get("ensemble", {}).get("roc_auc"),
        }
    return {
        "ensemble_pr_auc": data.get("pr_auc"),
        "ensemble_roc_auc": data.get("roc_auc"),
    }


def extract_log_highlights(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    keys = (
        "PR-AUC",
        "HARDENED Artifact Saving",
        "VERIFIED",
        "PIPELINE COMPLETE",
        "Run directory:",
        "ALL 7/7 artifacts saved successfully",
    )
    return [line for line in lines if any(k in line for k in keys)]


def write_package_readme(package_root: Path, latest_run: Path, metrics: dict[str, Any], core_ok: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    readme = f"""# Best Overall Submission Package

This is the final judge-facing package for ANC-052.

Generated UTC: {now}
Latest complete run: {latest_run.name}

## Quick Score Snapshot

1. Ensemble PR-AUC: {metrics.get('ensemble_pr_auc', 'N/A')}
2. CatBoost PR-AUC: {metrics.get('catboost_pr_auc', 'N/A')}
3. Deep Learning PR-AUC: {metrics.get('deep_learning_pr_auc', 'N/A')}
4. Core artifacts present: {core_ok}/{len(CORE_REQUIRED_ARTIFACTS)}

## Package Contents

1. artifacts/ - latest complete run + evidence JSON
2. logs/ - full pipeline log and extracted highlights
3. submission/ - official notebook, walkthrough notebook, CSV outputs, pitch assets
4. docs/ - strengths, feedback snapshot, technical references, next-step plan
5. configs/ - run configs for reproducibility
6. pipelines/ - orchestration, audit, evidence, and package build scripts

## Reviewer Flow

1. Open artifacts/evidence_latest_run.json.
2. Open logs/pipeline_highlights.txt.
3. Open submission/Best_Overall_Submission_Walkthrough.ipynb.
4. Open docs/COMPETITION_FEEDBACK_SNAPSHOT.md.
"""
    (package_root / "README.md").write_text(readme, encoding="utf-8")


def write_logs_readme(logs_dir: Path) -> None:
    text = """# Logs Folder

1. latest_pipeline.log - full training/pipeline execution log for latest complete run.
2. pipeline_highlights.txt - filtered proof lines for quick judge review.
3. run_metrics_summary.json - parsed PR-AUC and ROC-AUC snapshot.
"""
    (logs_dir / "README.md").write_text(text, encoding="utf-8")


def write_content_manifest(package_root: Path, copied: list[str], missing: list[str]) -> None:
    lines = [
        "# Content Manifest",
        "",
        "Generated file list for best-overall package build.",
        "",
        "## Copied Sources",
    ]
    lines.extend([f"1. {path}" for path in copied] if copied else ["1. None"])
    lines.extend(["", "## Missing Sources"]) 
    lines.extend([f"1. {path}" for path in missing] if missing else ["1. None"])
    (package_root / "CONTENT_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_package(repo_root: Path, output_dir: Path) -> None:
    copied: list[str] = []
    missing: list[str] = []

    artifacts_dir = repo_root / "artifacts"
    latest_run = pick_latest_complete_run(artifacts_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for folder in ["artifacts", "logs", "submission", "docs", "configs", "pipelines"]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    copy_file(repo_root / "requirements.txt", output_dir / "requirements.txt", copied, missing)

    copy_relative_files(repo_root, "submission", SUBMISSION_FILES, output_dir, "submission", copied, missing)
    copy_relative_files(repo_root, "docs", DOC_FILES, output_dir, "docs", copied, missing)
    copy_relative_files(repo_root, "configs", CONFIG_FILES, output_dir, "configs", copied, missing)
    copy_relative_files(repo_root, "pipelines", PIPELINE_FILES, output_dir, "pipelines", copied, missing)

    artifacts_out = output_dir / "artifacts"
    copy_file(repo_root / "artifacts" / "README.md", artifacts_out / "README.md", copied, missing)

    latest_run_dst = artifacts_out / latest_run.name
    shutil.copytree(latest_run, latest_run_dst)
    copied.append(str(latest_run))

    evidence_src = repo_root / "artifacts" / "evidence_latest_run.json"
    copy_file(evidence_src, artifacts_out / "evidence_latest_run.json", copied, missing)

    log_src = latest_run / "logs" / "pipeline.log"
    logs_out = output_dir / "logs"
    copy_file(log_src, logs_out / "latest_pipeline.log", copied, missing)

    metrics = parse_metrics(latest_run / "metrics.json")
    (logs_out / "run_metrics_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    highlights = extract_log_highlights(log_src)
    (logs_out / "pipeline_highlights.txt").write_text("\n".join(highlights) + "\n", encoding="utf-8")
    write_logs_readme(logs_out)

    core_ok = sum((latest_run / rel).exists() for rel in CORE_REQUIRED_ARTIFACTS)
    write_package_readme(output_dir, latest_run, metrics, core_ok)
    write_content_manifest(output_dir, copied, missing)

    print(f"Package built: {output_dir}")
    print(f"Latest complete run: {latest_run.name}")
    print(f"Core artifacts present: {core_ok}/{len(CORE_REQUIRED_ARTIFACTS)}")
    if missing:
        print(f"Missing optional files: {len(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build best-overall submission package.")
    parser.add_argument(
        "--output-dir",
        default="submission_best_overall_package",
        help="Output folder for the generated package",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / args.output_dir
    build_package(repo_root, output_dir)


if __name__ == "__main__":
    main()

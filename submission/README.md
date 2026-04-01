# Submission Folder

This folder contains competition deliverables and presentation assets.

## Files

1. Patient_Deterioration_Week1_Official_Submission_Notebook.ipynb
2. Best_Overall_Submission_Walkthrough.ipynb
3. Reproducible_EndToEnd_Runbook.ipynb
4. PITCH_SCRIPT.md
5. NOTEBOOK_SUBMISSION_GUIDE.md
6. Submission CSV files and reproduced metrics

## Dependency notes

1. Notebooks depend on project root code paths (`core/`, `models/`, `pipelines/`).
2. Metrics and reproducibility evidence are sourced from `artifacts/run_*`.
3. Pitch and guide docs should align with `docs/` and latest evidence snapshot.

## Guidance

1. Treat this folder as presentation-facing, not training-runtime output.
2. Keep filenames stable once submission packaging starts.
3. Ensure README claims match final generated metrics.
4. Use Python 3.10 for all notebook and pipeline executions.

## Suggested reviewer path

1. Open `NOTEBOOK_SUBMISSION_GUIDE.md`.
2. Run `Patient_Deterioration_Week1_Official_Submission_Notebook.ipynb`.
3. Validate generated outputs against `artifacts/evidence_latest_run.json`.

# Dashboard Folder

This folder contains the final Streamlit demonstration app for submission.

## Files

1. app.py
   Main dashboard entrypoint.

## Runtime dependency graph

1. Auto-loads the best verified run using `artifacts/evidence_latest_run.json`.
2. Uses ensemble predictions from `artifacts/run_*/predictions.csv`.
3. Reconstructs the exact validation split using `core/data_loader.py`.
4. Displays final benchmark evidence from `artifacts/benchmark_final/` when available.

## What the dashboard is expected to show

1. Executive ward risk overview from the best ensemble model.
2. Patient-level deep-dive timeline (risk and vitals).
3. Explainability from best-run SHAP outputs.
4. Final benchmark deltas (ensemble vs TimeSFM proxy) from `benchmark_final`.
5. Final-round submission checklist.

## Run

```powershell
py -3.10 -m streamlit run dashboard/app.py
```

Run this command from repository root for correct relative paths.

## Dependency

Dashboard requires a complete best run (`metrics.json`, `predictions.csv`) and dataset access.

## Clinical usage boundary

1. Dashboard is decision support, not autonomous diagnosis.
2. Scores should be interpreted with bedside context and clinician judgment.

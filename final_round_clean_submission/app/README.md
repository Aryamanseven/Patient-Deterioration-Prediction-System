# App Folder

This dashboard is standalone and reads only from files inside final_round_clean_submission.

## Input files (local only)

1. evidence/evidence_latest_run.json
2. evidence/benchmark_summary.json
3. evidence/benchmark_subsample_summary.csv
4. evidence/benchmark_full_sample_metrics.csv

## Pages

1. Overview
2. Benchmark
3. Submission Checklist

## Run

From repository root:

```powershell
py -3.10 -m streamlit run final_round_clean_submission/app/app.py
```

# Dashboard Folder

This folder contains the Streamlit demonstration app.

## Files

1. app.py
   Main dashboard entrypoint.

## Runtime dependency graph

1. Reads run artifacts from `artifacts/run_*`.
2. Uses project feature logic from `core/` for data shaping.
3. Displays metrics and optional module outputs when present.

## What the dashboard is expected to show

1. Ward-level risk overview.
2. Patient-level deep-dive timelines.
3. Clinical Action Board for triage decisions.
4. FL and DG outputs when module artifacts exist.
5. XAI outputs (SHAP/Captum) when generated.

## Run

```powershell
py -3.10 -m streamlit run dashboard/app.py
```

Run this command from repository root for correct relative paths.

## Dependency

Dashboard quality depends on the latest run directory containing required artifacts.

## Clinical usage boundary

1. Dashboard is decision support, not autonomous diagnosis.
2. Scores should be interpreted with bedside context and clinician judgment.

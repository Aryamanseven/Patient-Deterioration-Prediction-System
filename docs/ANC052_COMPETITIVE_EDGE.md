# ANC-052 Competitive Edge (PS-2)

Short positioning summary for judges and engineering reviewers.

## Why this system is competitive

1. Multi-model robustness:
Deep temporal model plus CatBoost plus ensemble avoids single-model brittleness.

2. Real execution evidence:
Runs produce traceable artifacts, logs, audit JSON, and summary JSON instead of slide-only claims.

3. Clinical usability:
Dashboard presents triage-level and patient-level views with explainability outputs.

## How we compare better on common failure modes

1. Weak validation in competitors:
Our pipeline saves full metrics and predictions for objective PR-AUC and ROC-AUC verification.

2. Incomplete execution:
Our hardened save contract prevents reporting success without required files.

3. Low depth baselines:
Our full profile combines SSL reuse, FL simulation, DG robustness checks, and XAI in one run.

## Practical claim boundary

1. We claim reproducible engineering and measurable model quality.
2. We do not claim hospital deployment approval.

# TimeSFM Competitive Strategy (Evidence-First)

This document explains how to position ANC-052 against TimeSFM when direct same-split outputs are not yet available.

## Reality Check

1. TimeSFM 2.5 is documented as a 200M-parameter model family.
2. The widely cited 100B figure refers to pretraining time-points, not parameters.
3. A direct superiority claim requires same-split, same-label, same-metric evaluation.

## Current Evidence We Can Use Now

1. Internal same-split benchmark shows ensemble superiority over single-branch baselines.
2. Calibration evidence is available via Brier score in run artifacts.
3. Reproducibility and governance evidence exists via config-driven runs, audit scripts, and structured artifacts.

## Where We Can Dominate Even Against a Large Foundation Model

1. Task Fit:
Clinical deterioration classification with target-specific objectives beats generic forecasting priors when labels and thresholds are tightly aligned to operations.

2. Calibration and Actionability:
In triage systems, confidence quality (Brier + threshold utility) can matter more than pure representation scale.

3. Reliability Under Execution:
Repeatable pipelines with strict artifact verification and explainability are often more deployable than higher-capacity but loosely integrated model stacks.

4. Clinical Explainability and Workflow:
Feature-level and patient-level interpretability in the dashboard provides practical clinical trust pathways.

5. System Economics:
Smaller, integrated, reproducible systems can outperform larger references in latency, iteration speed, and operational maintainability.

## Claim Policy

1. Allowed now:
"Our ensemble is statistically better than our internal baselines on the same split."

2. Not allowed now:
"We beat TimeSFM on this task." (until direct same-split benchmark is executed)

3. Allowed with attribution:
"TimeSFM has massive pretraining scale; we currently use it as an external reference family."

## Head-to-Head Checklist (When TimeSFM Outputs Are Reintroduced)

1. Generate TimeSFM predictions on the exact evaluation split.
2. Align schema to required columns: y_true and risk score(s).
3. Compute PR-AUC, ROC-AUC, Brier, Precision, Recall, F1.
4. Run paired bootstrap confidence intervals for delta metrics.
5. Publish both raw result files and summary markdown in artifacts benchmark suite.

## Judge-Safe One-Liner

"We respect foundation-model scale, but we optimize for clinical task fitness, calibration, reliability, and deployment-grade evidence, and that is where we can win."
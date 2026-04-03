# Artifact Governance Case Study: run_20260331_201944

This case study documents why strict artifact contracts are required in competition training systems.

## Observed artifact state

Run directory: artifacts/run_20260331_201944

Present:
1. model/ensemble.pkl

Missing at that time:
1. model/dl_model_final.pt
2. model/model.cbm
3. model/scaler.pkl
4. model/feature_columns.json

## Integrity gap identified

Control flow reached the end of training, but output completeness was not enforced strongly enough.
That creates risk for downstream notebook reproducibility and dashboard loading.

## Governance controls now applied

1. Required artifact verification is executed before success is declared.
2. Metrics are saved in structured per-model format.
3. Run folder naming uses timestamp plus run_name for traceability.
4. Non-destructive audit tool is available at pipelines/audit_artifacts.py.

## Practical takeaway

In this repository, run completion means both:
1. Training logic executed.
2. Required artifacts exist and are non-empty.

This is the expected quality bar for submission evidence.

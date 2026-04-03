# Execution Profiles Guide

This guide defines how to run PS-2 consistently from raw data to verified artifacts.

## Recommended run order

1. Smoke test:
	Use configs/quick_test.yaml to verify environment, imports, and pipeline flow.
2. Full submission run:
	Use configs/default.yaml for the complete module stack.
3. Artifact audit:
	Validate required files after each full run.

## Why this order works

1. Quick test catches environment problems early.
2. Full run executes the same end-to-end path used for submission evidence.
3. Audit avoids false success by checking required outputs explicitly.

## Core commands (Python 3.10)

```powershell
py -3.10 -m pip install -r requirements.txt
py -3.10 pipelines/run_full_pipeline.py --config configs/quick_test.yaml
py -3.10 pipelines/run_full_pipeline.py --config configs/default.yaml
py -3.10 pipelines/audit_artifacts.py --artifacts-dir artifacts
```

## Artifact alias shortcuts

Use these aliases only when a profile intentionally reuses prior artifacts.

1. AUTO_LATEST_SSL -> newest artifacts/run_*/ssl_pretrained_tcntransformer.pt
2. AUTO_LATEST_DL_CHECKPOINT -> newest artifacts/run_*/model/dl_checkpoint_latest.pt
3. AUTO_LATEST_DL_FINAL -> newest artifacts/run_*/model/dl_model_final.pt

## Required artifact contract

Every successful run must contain:

1. model/dl_model_final.pt
2. model/model.cbm
3. model/scaler.pkl
4. model/ensemble.pkl
5. model/feature_columns.json
6. metrics.json
7. predictions.csv
8. ssl_pretrained_tcntransformer.pt

When advanced modules are enabled, expect:

1. fl_rounds_history.json
2. lodo_results.csv
3. top_features.csv
4. shap_summary.png
5. captum_temporal_heatmap.png

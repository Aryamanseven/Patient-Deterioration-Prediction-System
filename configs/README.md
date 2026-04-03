# Configuration Files

This folder defines reproducible run profiles for Python 3.10 execution.

## Primary profile matrix

| File | Purpose | SSL | FL | DG | XAI |
| --- | --- | --- | --- | --- | --- |
| default.yaml | Full competition run | On | On | On | On |
| quick_test.yaml | Fast smoke test | On | Optional | Optional | Optional |

## Compatibility profiles

Additional profiles are kept for checkpoint-reuse and extended training workflows.
These are backward-compatible run options and are not required for standard submission execution.

## Important behavior

1. All runs are Python 3.10 and should use the same interpreter to avoid package drift.
2. Each run writes a fresh folder under artifacts/run_<timestamp>_<run_name>.
3. Required artifacts are verified before the pipeline reports success.

## Alias shortcuts

1. AUTO_LATEST_SSL -> newest artifacts/run_*/ssl_pretrained_tcntransformer.pt
2. AUTO_LATEST_DL_CHECKPOINT -> newest artifacts/run_*/model/dl_checkpoint_latest.pt
3. AUTO_LATEST_DL_FINAL -> newest artifacts/run_*/model/dl_model_final.pt

## Recommended commands

```powershell
py -3.10 pipelines/run_full_pipeline.py --config configs/quick_test.yaml
py -3.10 pipelines/run_full_pipeline.py --config configs/default.yaml
py -3.10 pipelines/audit_artifacts.py --artifacts-dir artifacts
```

## Minimal schema guide

```yaml
general:
  seed: 42
  python_version: "3.10"
  device: "auto"

data:
  path: "dataset/train.csv"
  test_size: 0.2

features:
  use_advanced: true
  use_clinical_scores: true

modules:
  ssl:
    enabled: true
    reuse_existing: false
    pretrained_weights_path: ""

  deep_learning:
    enabled: true
    resume_from_checkpoint: true
    resume_checkpoint_path: "AUTO_LATEST_DL_CHECKPOINT"

  federated_learning:
    enabled: true

  domain_generalization:
    enabled: true

  xai:
    enabled: true
```

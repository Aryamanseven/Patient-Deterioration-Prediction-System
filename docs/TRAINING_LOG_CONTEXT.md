# Training Log Context Guide

This note explains how to read train and validation logs for the PS-2 deep model path.

## Why validation loss can be higher than training loss

A pattern like:

- Epoch Train: 0.25
- Epoch Val: 1.73 to 2.37

can still be valid in this project because:

1. Strong class imbalance handling is used with weighted BCE (pos_weight).
2. Validation distribution can be harder than training after group-aware episode split.
3. Federated client rounds can shift local gradients and temporarily raise validation loss.
4. Training loss is averaged across mini-batches and can look smoother than strict epoch-level validation.

## Interpreting the provided log snippet

Example behavior:

1. Running training loss inside an epoch drops over batches.
2. Epoch-level validation may increase on some epochs.
3. Best validation checkpoint remains the main signal.
4. Training can still be healthy when best validation improves across rounds or clients.

So a sequence such as best val from 1.7358 to 1.3409 is generally a positive sign.

## When to worry

Investigate if all are true:

1. Validation loss rises monotonically for many epochs without any new best.
2. PR-AUC and ROC-AUC degrade together.
3. Predictions collapse to near-constant probabilities.

If these are not happening, temporary val spikes are usually expected behavior.

## Quick log checks

Use these commands from repo root:

```powershell
py -3.10 pipelines/audit_artifacts.py --artifacts-dir artifacts
Get-Content artifacts\run_*\logs\pipeline.log -Tail 80
```

For a completed run, always prioritize:

1. metrics.json values
2. predictions.csv quality
3. required artifact completeness

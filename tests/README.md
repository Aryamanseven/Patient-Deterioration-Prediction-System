# Tests Folder

This folder contains lightweight checks for repository health.

## Files

1. test_imports.py
   Verifies key package imports and module wiring.

2. test_pipeline_dryrun.py
   Dry-run pipeline sanity checks.

## Dependency notes

1. Import tests verify module wiring across `core/`, `models/`, `modules/`, and `pipelines/`.
2. Dry-run test validates end-to-end command-level execution behavior.

## Run

```powershell
py -3.10 -m pytest tests -q
```

## Scope

These tests are smoke checks, not full model-quality validation.

## Recommended expansion for production readiness

1. Add deterministic dashboard-risk source tests.
2. Add model-type interface compatibility tests.
3. Add script parameter contract and portability tests.

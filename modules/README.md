# Modules Folder

This folder contains optional capabilities layered on top of the core training flow.

## Submodules

1. ssl
   Self-supervised pretraining and weight reuse.

2. federated_learning
   FedAvg simulation over client partitions.

3. domain_generalization
   Leave-One-Domain-Out robustness evaluation.

4. xai
   SHAP and Captum explainability outputs.

5. differential_privacy
   Optional Opacus-based DP hooks.

6. deployment
   Model export utilities.

## Design intent

Modules are config-driven and can be enabled or disabled per profile. The default competition profile enables FL, DG, and XAI together.

## Dependency notes

1. Orchestrated by `pipelines/run_full_pipeline.py`.
2. Consumes model and feature outputs from `core/` and `models/`.
3. Writes supplemental evidence files into the active run directory.

## Output expectations by module

1. ssl
   Produces reusable SSL checkpoint/weights.

2. federated_learning
   Produces round history JSON.

3. domain_generalization
   Produces LODO evaluation CSV.

4. xai
   Produces SHAP and optional Captum explainability artifacts.

5. deployment
   Produces export-format model artifacts when enabled.

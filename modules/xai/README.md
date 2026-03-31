# Explainable AI (XAI) Module

Trust is critical in clinical AI. This module ensures our model is transparent and clinically verifiable.

## Features
- **SHAP (SHapley Additive exPlanations):** Global and local feature importances indicating *why* the model made a specific prediction.
- **Clinical Alignment:** Compares the AI prediction drivers against established clinical scores (NEWS, MEWS, qSOFA) to prove we are discovering novel deterioration signatures, not just memorizing scores.

## Artifacts Produced
- `shap_summary.png`: Global feature importance plot.
- `top_features.csv`: The top 20 most important clinical features discovered by the model.

## Config
Controlled via `modules.xai` in `default.yaml`.

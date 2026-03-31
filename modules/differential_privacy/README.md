# Differential Privacy (DP) Module

Protects the model against model-inversion and membership inference attacks, guaranteeing that no individual patient's records can be extracted from the final trained model.

## Method
- We use DP-SGD (Differentially Private Stochastic Gradient Descent) via the `opacus` library.
- Gradients are clipped per-sample to ensure bounded sensitivity (`max_grad_norm`).
- Gaussian noise is added to the aggregated gradients (`noise_multiplier`) to mask individual patient contributions.

## Config
Controlled via `modules.differential_privacy` in `default.yaml`.
The calculated `epsilon` (privacy budget) is logged at the end of training.

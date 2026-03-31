# Medical Device Deployment Module

Exports our models in a secure, containerized, and ONNX-ready format allowing direct integration with hospital Electronic Health Record (EHR) systems.

## Artifacts Produced
- `model.onnx`: Optimized ONNX export of deep learning models for CPU/GPU serving.
- `catboost_model.cbm`: CatBoost deployment binary.

## Config
Controlled via `modules.deployment` in `default.yaml`.

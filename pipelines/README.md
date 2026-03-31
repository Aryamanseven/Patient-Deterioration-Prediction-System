# Pipeline Module

This folder contains the master orchestration scripts that tie the entire system together.

## `run_full_pipeline.py`
The single entry point for the entire Patient Deterioration Prediction System.
It dynamically reads `configs/default.yaml` and executes:
1. Data Loading & Splitting
2. Feature Engineering & Clinical Score generation
3. Self-Supervised Learning (SSL) Pre-training
4. Supervised Baseline Training (CatBoost)
5. Deep Learning Training (TCN-Transformer with DP-SGD)
6. Out-of-Distribution Verification (LODO)
7. Federated Learning Simulation
8. Explainability (XAI / SHAP)
9. Deployment Model Export (ONNX / CBM)

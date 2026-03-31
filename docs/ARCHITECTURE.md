# PhysioGuard Architecture

PhysioGuard is a production-grade, modular, and extensible system predicting patient physiological deterioration 12 hours in advance using vital sign time-series data. 

## Design Philosophy
1. **Config-Driven**: Every parameter, from feature windows to model epochs, is defined in centralized YAML configuration files (`configs/`).
2. **Modular**: Clean separation between core utilities (`core/`), model definitions (`models/`), research novelties (`modules/`), and pipeline execution (`pipelines/`).
3. **No-Leakage Guarantee**: Feature engineering explicitly groups by independent patient episodes and uses `shift()` and constrained `rolling()` to ensure forward-looking statements are strictly masked.
4. **Reproducibility**: All random operations and data splits share a single seed, managed by `core/reproducibility.py`.

## Directory Structure

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `configs/` | YAML config definitions | `default.yaml`, `quick_test.yaml`, `schema.py` (future) |
| `core/` | Shared data processing and metric utilities | `features.py`, `clinical_scores.py`, `data_loader.py` |
| `models/` | Implementations of ML algorithms | `catboost_model.py`, `tcn_transformer.py`, `ensemble.py` |
| `modules/`| Advanced features & research novelties | `ssl/`, `federated_learning/`, `xai/`, `deployment/` |
| `pipelines/` | End-to-end execution | `run_full_pipeline.py` |
| `tests/` | Unit and integration tests | `test_imports.py`, `test_pipeline_dryrun.py` |
| `dashboard/` | Streamlit clinical UI | `app.py` |

## Data Flow
The pipeline Orchestrator (`run_full_pipeline.py`):
1. **Loads Config**: Reads `quick_test.yaml` or `default.yaml`.
2. **Loads Data**: `data_loader.py` reads data, optionally generating synthetic data if `train.csv` is missing.
3. **Engineers Features**: `features.py` generates 257 time-series features.
4. **Pre-trains (SSL)**: Executes `SSL Masked Autoencoder` if enabled.
5. **Trains Models**: Evaluates CatBoost and Deep Learning structures based on configuration.
6. **Novelty Injections**:
   - Computes LODO (Leave-One-Domain-Out) metrics.
   - Computes SHAP explainability.
   - Simulates Federated Learning across independent silos.
7. **Exports**: Generates inference-ready `.cbm` and `.onnx` outputs.

## Deployment Ready
The resulting pipeline seamlessly feeds into a Streamlit dashboard built for hospital infrastructure, visualizing global patient state alongside individual trajectory trajectories against traditional clinical baselines (NEWS/MEWS/qSOFA).

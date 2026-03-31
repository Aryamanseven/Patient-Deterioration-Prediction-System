# Core Module

Shared utilities used by **every other module** in the system. Nothing in `models/`, `modules/`, or `pipelines/` imports from anywhere else — they all use `core/`.

## Files

| File | Purpose | Used By |
|---|---|---|
| `config.py` | Loads and validates YAML configs. Every parameter comes from here. | All pipeline steps |
| `data_loader.py` | **SINGLE** centralized data loading + splitting. No other module touches the CSV. | `pipelines/steps/step_data.py` |
| `features.py` | All feature engineering (base 212 + advanced 257 features) | `data_loader.py` |
| `clinical_scores.py` | NEWS, MEWS, qSOFA score computation | `features.py`, `modules/xai/` |
| `metrics.py` | All evaluation metrics (ROC-AUC, PR-AUC, Brier, F1, etc.) + threshold optimization | All evaluation steps |
| `reproducibility.py` | Seed management for Python, NumPy, PyTorch. Deterministic operations. | Pipeline entry point |
| `logger.py` | Structured logging to file + console | All modules |

## Design Principle

**No circular imports.** The dependency graph is:
```
config.py ← (no deps)
logger.py ← config.py
reproducibility.py ← (no deps)
clinical_scores.py ← (no deps, just pandas/numpy)
features.py ← clinical_scores.py
metrics.py ← (no deps, just sklearn)
data_loader.py ← config.py, features.py, reproducibility.py
```

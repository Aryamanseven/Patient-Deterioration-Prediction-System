# Models

This directory contains the core model architectures and their wrappers.
All models expose a consistent `fit()`, `predict_proba()`, and `save()` / `load()` interface to the pipeline.

## Files

| File | Description | Output |
|---|---|---|
| `catboost_model.py` | Supervised baseline. Handles class imbalance and tabular dynamics. | CatBoostClassifier |
| `lstm_attention.py` | Deep learning baseline. GRU/LSTM with temporal attention. | PyTorch Model |
| `tcn_transformer.py` | The Championship Architecture. Multi-Scale TCN + Transformer. | PyTorch Model |
| `ensemble.py` | Combines CatBoost and Deep Learning predictions. | Metalearner / Dict |
| `model_registry.py` | Instantiates models based on YAML config strings. | Base Model |

## Design Rules
1. **No Data Processing:** Models expect data to be fully processed by `core/data_loader.py`.
2. **Config Driven:** Model hyper-parameters are passed via `**params` from the config file. No hardcoding.
3. **Consistent Output:** All classification models must return probabilities between 0 and 1.

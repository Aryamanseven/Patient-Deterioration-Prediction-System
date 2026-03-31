# Module Connections

This document details how the disparate modules communicate without circular dependencies.

## The Dependency Pyramid

The architecture enforce a strict unidirectional dependency graph:
1. `core/`: Zero dependencies on anything else in the project. Exposes pure functions and stateless utilities.
2. `models/`: Depends *only* on `core/`. Handles pure PyTorch / CatBoost architectures.
3. `modules/`: Interacts with `models/` and utilizes `core/`.
4. `pipelines/`: The orchestrator wrapper. Only `run_full_pipeline.py` is permitted to import across all domains simultaneously.

## Data Exchange format

The common currency between all modules is the Pandas DataFrame (for tabular data) or Numpy properties for predictions. 

- `data_loader` passes a DataFrame to `features`. 
- `features` adds up to 257 columns and returns a modified DataFrame.
- `run_full_pipeline` passes the feature matrix to the DL/CatBoost `fit` loops.
- `modules/xai/` takes the fitted CatBoost model and the evaluation DataFrame to yield SHAP data.
- `modules/deployment/` takes the raw model artifacts and packages them to the filesystem.

No module persists global state. File I/O side-effects are heavily isolated to `modules/deployment/` or specifically passed output strings.

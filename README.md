# AI-Based Early Warning System for Patient Physiological Deterioration

This project builds an early warning system for predicting whether a patient is likely to physiologically deteriorate within the next 12 hours using hourly vital-sign and lab time-series data. It includes:

- a CatBoost-based machine learning model
- a GPU-ready deep learning comparison pipeline
- a preprocessing and feature-engineering pipeline
- saved baseline predictions for `val_no_labels.csv`
- a Streamlit dashboard for visualizing episodes and predicted risk

The repository is intentionally kept slim. Bulky experiment outputs, training logs, Python caches, and extra model artifacts are generated locally when needed and are not committed by default. The tracked model assets keep two comparison anchors in the repo: the baseline dashboard model and the best CatBoost search result, `catboost_gpu_subsample_train80`.

## Project Structure

- `train_model.py`: trains the model, evaluates it on a group-aware holdout split, and saves artifacts
- `train_deep_models.py`: trains `TCN`, `GRU+Attention`, and `Transformer Encoder` sequence models on GPU and compares them against the CatBoost baseline
- `optimize_best_model.py`: tunes GPU CatBoost variants, searches validation-based ensembles, and identifies the strongest overall model
- `app.py`: launches the dashboard
- `src/physio_warning/features.py`: episode reconstruction and feature engineering logic
- `src/physio_warning/deep_learning.py`: sequence preprocessing, PyTorch models, and GPU training utilities
- `artifacts/`: baseline trained model, metrics, feature importances, and prediction CSVs used by the dashboard

## Modeling Approach

The dataset does not include an explicit patient identifier, so the pipeline reconstructs an `episode_id` whenever `hour_from_admission` resets back to `0` or decreases. Each row is then treated as an hourly snapshot within an admission.

The model uses:

- current vital signs and lab measurements
- derived clinical signals such as shock index, pulse pressure, mean arterial pressure, and oxygen deficit
- leakage-safe lag features using prior values at 1, 3, and 6 hours
- rolling mean and rolling standard deviation over 3, 6, and 12 hour windows
- prior care-context signals such as previous oxygen device, nurse alert, and mobility score

CatBoost was chosen because it handles tabular, nonlinear, mixed-type clinical data well and supports categorical features directly.

## How To Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model and generate artifacts:

```bash
python train_model.py
```

3. Launch the dashboard:

```bash
streamlit run app.py
```

4. Train and compare the GPU deep learning models:

```bash
python train_deep_models.py
```

5. Search for the strongest overall model:

```bash
python optimize_best_model.py
```

## Outputs

After training, the root `artifacts/` folder contains the baseline dashboard assets:

- `deterioration_model.cbm`
- `metadata.json`
- `feature_importance.csv`
- `holdout_predictions.csv`
- `val_predictions.csv`

Optional experiment runs save additional generated outputs that are intentionally gitignored:

- deep-model checkpoints and per-model prediction files under `artifacts/deep_models/`
- extra CatBoost search checkpoints and prediction files under `artifacts/model_search/`
- `catboost_info/` for CatBoost training logs

The best-model search writes additional artifacts under `artifacts/model_search/`, including:

- `best_model_comparison.csv`
- `best_model_summary.json`
- `best_model_metric_summary.md`
- tuned CatBoost checkpoints and prediction files

## Evaluation

The training script uses a group-aware holdout split by reconstructed episode, which is important to avoid leakage across time steps from the same admission.

Current holdout results from `artifacts/metadata.json`:

- ROC-AUC: `0.9649`
- PR-AUC: `0.7115`
- Brier score: `0.0482`
- Watch threshold: `0.4525` with precision `0.4194` and recall `0.8502`
- Alert threshold: `0.8423` with precision `0.7344`, recall `0.6796`, and F1 `0.7060`

Top model drivers on the holdout run:

- `hour_from_admission`
- `lactate_delta_3`
- `spo2_pct`
- `shock_index`
- `lactate_delta_6`
- `spo2_deficit`
- `respiratory_rate`
- `creatinine_delta_3`

## Deep Learning Comparison

The deep learning pipeline uses PyTorch with automatic CUDA detection and trains on rolling windows of the last `24` hours.

Current comparison results from `artifacts/deep_models/model_comparison.csv`:

- `CatBoost`: PR-AUC `0.7115`
- `CatBoost + best deep average`: PR-AUC `0.7083`
- `CatBoost + top-2 deep average`: PR-AUC `0.7056`
- `Transformer Encoder`: PR-AUC `0.6547`
- `Transformer Encoder Wide`: PR-AUC `0.6517`
- `Transformer Encoder Long`: PR-AUC `0.6459`
- `GRU + Attention`: PR-AUC `0.6148`
- `TCN`: PR-AUC `0.3588`

Current takeaway:

- `CatBoost` is still the strongest model for this dataset.
- The baseline `Transformer Encoder` is still the best deep learning candidate after the harder tuning pass.
- The CatBoost-plus-deep ensembles are competitive, but still do not beat CatBoost on PR-AUC.
- The clean per-model threshold summary lives in `artifacts/deep_models/model_metric_summary.md`.

## Best Overall Model

The strongest model found so far is the tuned GPU CatBoost retrained on the full `80%` development split:

- Model: `catboost_gpu_subsample_train80`
- ROC-AUC: `0.9621`
- PR-AUC: `0.7323`
- Brier score: `0.0279`
- Watch threshold: `0.1484` with precision `0.3905` and recall `0.8502`
- Alert threshold: `0.6533` with precision `0.7350`, recall `0.6860`, and F1 `0.7096`

Tracked model assets:

- `artifacts/deterioration_model.cbm`
- `artifacts/metadata.json`
- `artifacts/feature_importance.csv`
- `artifacts/holdout_predictions.csv`
- `artifacts/val_predictions.csv`
- `artifacts/model_search/catboost_gpu_subsample_train80.cbm`
- `artifacts/model_search/catboost_gpu_subsample_train80_holdout_predictions.csv`
- `artifacts/model_search/catboost_gpu_subsample_train80_metrics.json`
- `artifacts/model_search/best_model_summary.json`
- `artifacts/model_search/best_model_metric_summary.md`

Comparison summaries can stay in the repo, and any removed per-model artifacts can be regenerated locally with `python train_deep_models.py` or `python optimize_best_model.py` when you need them.

## Limitations

- `episode_id` is inferred from the hour reset assumption, because the dataset does not expose a native patient/admission identifier.
- The data appears pre-cleaned with no missing values, so the system has not been stress-tested on sparse home-monitoring streams.
- This is a decision-support prototype, not a clinical-grade alarm system. Thresholds should be clinically validated before real-world use.

# AI-Based Early Warning System for Patient Physiological Deterioration

This repository is the cleaned final submission workspace for the patient deterioration hackathon project. It predicts whether a patient is likely to deteriorate within the next `12` hours from hourly vital-sign and laboratory time-series data, and it keeps only the final four comparison models in the codebase:

- `catboost_baseline`
- `catboost_gpu_subsample_train80`
- `catboost_transformer_hybrid`
- `transformer_encoder_wide`

This `README.md` is the single up-to-date project guide and also serves as the technical report summary.

## Deliverables

### 1. Machine Learning Models

The final kept models are:

- `catboost_baseline`: stable dashboard-ready baseline trained by `train_model.py`
- `catboost_gpu_subsample_train80`: strongest overall model and recommended primary submission
- `catboost_transformer_hybrid`: tuned CatBoost + best transformer weighted blend
- `transformer_encoder_wide`: best deep learning model

### 2. Data Processing Pipeline

Implemented in `src/physio_warning/features.py` and shared across the training scripts.

Pipeline highlights:

- reconstructs `episode_id` whenever `hour_from_admission` resets or decreases
- creates leakage-safe lag features at `1`, `3`, and `6` hours
- creates rolling mean and rolling standard deviation features over `3`, `6`, and `12` hours
- adds derived clinical variables such as shock index, mean arterial pressure, pulse pressure, oxygen deficit, and fever/tachypnea/tachycardia excess
- keeps categorical context such as `oxygen_device`, `gender`, and `admission_type`

### 3. Prototype Interface

The Streamlit prototype is in `app.py`.

It supports:

- loading the validation dataset, training dataset, or an uploaded CSV
- episode-level risk visualization
- vital-sign trend plots
- risk-band display using watch and alert thresholds
- a patient snapshot and top model drivers

The current dashboard uses the baseline artifact bundle from `train_model.py`, which keeps the live demo simple and stable.

### 4. Demo

The demo flow is:

1. Train the baseline model with `python train_model.py`
2. Launch the interface with `streamlit run app.py`
3. Show risk trends on `dataset/val_no_labels.csv` or `dataset/train.csv`
4. Compare finalists using `python train_deep_models.py` and `python optimize_best_model.py`

### 5. Technical Report

The technical report content is captured in this README plus the generated metric summaries:

- `artifacts/deep_models/model_metric_summary.md`
- `artifacts/model_search/best_model_metric_summary.md`

## Repository Layout

- `train_model.py`: trains the baseline CatBoost model and writes dashboard-ready artifacts
- `train_deep_models.py`: trains the final transformer-based deep model on GPU
- `optimize_best_model.py`: rebuilds the final four-model comparison bundle
- `app.py`: Streamlit demo interface
- `src/physio_warning/features.py`: preprocessing and feature engineering
- `src/physio_warning/deep_learning.py`: sequence preprocessing and transformer training utilities
- `dataset/train.csv`: labeled training data
- `dataset/val_no_labels.csv`: unlabeled evaluation/demo data

## Modeling Approach

### Problem framing

Each row is treated as one hourly patient state, and the target is `deterioration_next_12h`.

Because the dataset does not provide a native patient or admission identifier, the pipeline reconstructs an episode boundary when `hour_from_admission` drops or resets. That assumption is used consistently across the baseline, transformer, and tuned-model comparison scripts.

### Baseline model

The baseline uses `CatBoostClassifier` on engineered tabular features.

Why it works well here:

- the dataset is medium-sized and structured
- vitals, labs, and categorical care-context variables mix naturally in a tabular model
- the engineered lag and rolling features already capture much of the temporal behavior

### Deep learning model

The retained deep model is `transformer_encoder_wide`.

It uses:

- up to `36` historical hourly steps
- normalized dynamic physiology features
- static patient context
- learned positional embeddings
- transformer encoder blocks plus attention pooling

### Final tuned model

The best overall model is the tuned GPU CatBoost variant:

- iterations: `1300`
- depth: `8`
- learning rate: `0.05`
- bootstrap: `Bernoulli`
- subsample: `0.85`
- `l2_leaf_reg`: `6.0`
- `random_strength`: `0.7`

This model is retrained on the full `80%` development split before final holdout evaluation.

### Hybrid model

The hybrid challenger blends the tuned CatBoost model with the best transformer:

- CatBoost weight: `0.57`
- Transformer weight: `0.43`

## Final Results

The final kept four-model comparison lives in `artifacts/model_search/best_model_comparison.csv`.

### `catboost_gpu_subsample_train80`

- ROC-AUC: `0.9624`
- PR-AUC: `0.7335`
- Brier score: `0.0277`
- Watch threshold: `0.1537` with precision `0.3962` and recall `0.8508`
- Alert threshold: `0.7117` with precision `0.7574`, recall `0.6688`, and F1 `0.7103`

### `catboost_transformer_hybrid`

- ROC-AUC: `0.9598`
- PR-AUC: `0.7263`
- Brier score: `0.0348`
- Watch threshold: `0.2662` with precision `0.3693` and recall `0.8505`
- Alert threshold: `0.7642` with precision `0.7679`, recall `0.6707`, and F1 `0.7160`

### `catboost_baseline`

- ROC-AUC: `0.9649`
- PR-AUC: `0.7115`
- Brier score: `0.0482`
- Watch threshold: `0.4525` with precision `0.4194` and recall `0.8502`
- Alert threshold: `0.8423` with precision `0.7344`, recall `0.6796`, and F1 `0.7060`

### `transformer_encoder_wide`

- ROC-AUC: `0.9484`
- PR-AUC: `0.6576`
- Brier score: `0.0618`
- Watch threshold: `0.4077` with precision `0.3310` and recall `0.8502`
- Alert threshold: `0.9062` with precision `0.6664`, recall `0.6427`, and F1 `0.6543`

## GPU Revalidation Search

The March 27, 2026 rerun of `revalidate_model_search.py` was executed on the local NVIDIA GPU and saved to `artifacts/model_search_revalidated_20260327_gpu`.

- runtime: `CatBoost GPU`, device `0`, `19` screened candidates, `2` screening repeats, `50%` screening episode subsample
- dataset summary: `293,248` rows, `7,000` reconstructed episodes, `5.405%` positive rate
- best overall observed model in the combined leaderboard remained the saved submission artifact `catboost_gpu_subsample_train80`
- saved submission artifact metrics: ROC-AUC `0.9624`, PR-AUC `0.7335`, Brier score `0.0277`
- best newly revalidated single model: `repo_catboost_subsample` with ROC-AUC `0.9635`, PR-AUC `0.7044`, Brier score `0.0521`
- best newly revalidated ensemble: `repo_catboost_subsample_plus_top3` with ROC-AUC `0.9643`, PR-AUC `0.7052`, Brier score `0.0543`

The GPU rerun is intentionally more conservative than the saved repo bundle because it uses repeated screening plus an untouched outer holdout. Its main value is as a robustness check, not as evidence that the saved submission model was surpassed.

## Recommended Submission Strategy

- Primary model: `catboost_gpu_subsample_train80`
- Backup challenger: `catboost_transformer_hybrid`
- Deep-learning architecture to present: `transformer_encoder_wide`
- Dashboard demo model: `catboost_baseline`

This gives a strong competition story:

- a stable baseline that already performs well
- a tuned production-style winner
- a deep-learning branch for temporal modeling depth
- a hybrid ensemble for additional experimentation

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the baseline dashboard artifacts:

```bash
python train_model.py
```

Launch the Streamlit demo:

```bash
streamlit run app.py
```

Train the final transformer model on your NVIDIA GPU:

```bash
python train_deep_models.py
```

Regenerate the final four-model bundle:

```bash
python optimize_best_model.py
```

Run the GPU revalidation search:

```bash
python revalidate_model_search.py --device gpu --gpu-devices 0 --output-dir artifacts/model_search_revalidated_20260327_gpu
```

If you omit `--device`, the script now defaults to `auto` and will use CatBoost GPU mode when a compatible GPU is available.
If you omit `--output-dir`, the script now writes into a dated folder such as `artifacts/model_search_revalidated_20260327_gpu` or `artifacts/model_search_revalidated_20260327_cpu` based on the actual runtime device.

## Final Artifacts

### Baseline artifacts

- `artifacts/deterioration_model.cbm`
- `artifacts/metadata.json`
- `artifacts/feature_importance.csv`
- `artifacts/holdout_predictions.csv`
- `artifacts/val_predictions.csv`

### Deep-learning artifacts

- `artifacts/deep_models/transformer_encoder_wide.pt`
- `artifacts/deep_models/model_comparison.csv`
- `artifacts/deep_models/model_metric_summary.md`

### Final comparison artifacts

- `artifacts/model_search/catboost_gpu_subsample_train80.cbm`
- `artifacts/model_search/catboost_gpu_subsample_train80_holdout_predictions.csv`
- `artifacts/model_search/catboost_transformer_hybrid_holdout_predictions.csv`
- `artifacts/model_search/best_model_comparison.csv`
- `artifacts/model_search/best_model_metric_summary.md`
- `artifacts/model_search/best_model_summary.json`
- `artifacts/model_search/final_model_registry.json`

### GPU revalidation artifacts

- `artifacts/model_search_revalidated_20260327_gpu/search_summary.json`
- `artifacts/model_search_revalidated_20260327_gpu/screening_results.csv`
- `artifacts/model_search_revalidated_20260327_gpu/finalist_holdout_results.csv`
- `artifacts/model_search_revalidated_20260327_gpu/ensemble_results.csv`
- `artifacts/model_search_revalidated_20260327_gpu/combined_comparison.csv`

## Evaluation Protocol

Evaluation uses group-aware splitting by reconstructed episode to avoid leakage across hourly rows from the same admission.

The comparison pipeline uses:

- outer holdout split: `20%`
- inner validation split from the remaining development data
- PR-AUC as the main model-selection metric
- ROC-AUC and Brier score as supporting metrics

## Limitations

- `episode_id` is inferred from hour resets because no native patient/admission identifier is provided
- the dashboard currently uses the baseline model bundle, not the tuned finalist bundle
- the hybrid model is a weighted ensemble artifact, not a single standalone checkpoint
- the dataset appears pre-cleaned and does not stress-test missingness common in real home-monitoring streams
- this is a clinical decision-support prototype, not a deployment-ready medical device

# AI-Based Early Warning System for Patient Physiological Deterioration

This repository is the final hackathon workspace for predicting whether a patient will deteriorate in the next 12 hours using hourly physiological, laboratory, demographic, and care-context signals.

The final project result is a focused CatBoost model that outperformed the baseline tabular model, the pure transformer benchmark, and the CatBoost + Transformer hybrid benchmark.

## Final Submission Package

Everything the hackathon team needs is already in `submission/`:

- `submission/AesCodeNexus_Round1_Submission_Final_Technical_Patient_Deterioration_TeamReady.pptx`
  Presentation deck for the final submission
- `submission/Patient_Deterioration_Week1_Official_Submission_Notebook.ipynb`
  Official notebook to upload to Google Colab or Kaggle and share publicly
- `submission/week1_official_submission_results.csv`
  Final four-experiment comparison table used for submission
- `submission/official_winner_reproduced_metrics.csv`
  Exact reproduced metrics for the final winning model
- `submission/focused_subsample_lr0048_iter1450_official_submission_predictions.csv`
  Blind validation predictions generated with the official winning model
- `submission/NOTEBOOK_SUBMISSION_GUIDE.md`
  Short upload and sharing guide for the notebook submission

## Final Result

The verified final winner in this repository is:

- Model: `focused_subsample_lr0048_iter1450`
- PR-AUC: `0.7396128871401634`
- ROC-AUC: `0.9630869229234396`
- Brier score: `0.0279038392646778`
- Watch threshold: `0.16533940710070974`
- Alert threshold: `0.7072099763613172`

Final model parameters:

- `iterations=1450`
- `depth=8`
- `learning_rate=0.048`
- `l2_leaf_reg=7.0`
- `random_strength=0.6`
- `bootstrap_type=Bernoulli`
- `subsample=0.88`
- `border_count=254`

## Where The Final Result Is Proven

The source of truth for the final winner is the focused-sweep artifact folder:

- `artifacts/model_search_revalidated_20260326/README.md`
- `artifacts/model_search_revalidated_20260326/best_overall_summary.json`
- `artifacts/model_search_revalidated_20260326/aggregate_best_results.csv`
- `artifacts/model_search_revalidated_20260326/focused_subsample_round2_results.csv`
- `artifacts/model_search_revalidated_20260326/focused_subsample_round3_results.csv`
- `artifacts/model_search_revalidated_20260326/focused_subsample_lr0048_iter1450_final_artifact_summary.json`
- `artifacts/model_search_revalidated_20260326/focused_subsample_lr0048_iter1450_best_holdout_predictions.csv`
- `artifacts/model_search_revalidated_20260326/focused_subsample_lr0048_iter1450_full_train_model.cbm`

Most important proof files:

- `best_overall_summary.json`
  Declares `focused_subsample_lr0048_iter1450` as the best model overall
- `focused_subsample_round2_results.csv`
  Contains the winning result row
- `focused_subsample_lr0048_iter1450_final_artifact_summary.json`
  Stores the exact winning hyperparameters and reference metrics
- `focused_subsample_lr0048_iter1450_best_holdout_predictions.csv`
  Reproduces the exact final metrics when rescored

Important clarification:

- The PPT is a presentation summary, not the source of truth
- `revalidate_model_search.py` is useful for conservative revalidation, but by itself it does not recreate the final `0.7396` winner in one run
- The final lift came from the later focused CatBoost sweep preserved in `artifacts/model_search_revalidated_20260326`

## Official Experiment Comparison

This is the final four-experiment story for the hackathon submission.

| Experiment | Model | PR-AUC | ROC-AUC | Brier | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Exp 2. Final Focused CatBoost Winner | `focused_subsample_lr0048_iter1450` | `0.7396128871401634` | `0.9630869229234396` | `0.0279038392646778` | Final submission model and best overall result |
| Exp 1. Baseline CatBoost | `catboost` | `0.7114498552976203` | `0.9648972833490874` | `0.04818897155863399` | Strong tabular baseline and dashboard model |
| Exp 4. CatBoost + Transformer Hybrid | `catboost_gpu_subsample_plus_transformer_encoder` | `0.705305` | `0.963737` | `0.05408` | Weighted hybrid benchmark |
| Exp 3. Transformer Encoder | `transformer_encoder` | `0.654732` | `0.951335` | `0.074196` | Pure sequence-model benchmark |

Submission-facing copies of these results are saved in:

- `submission/week1_official_submission_results.csv`
- `submission/official_winner_reproduced_metrics.csv`

## What The Team Should Demo

For a live demo, use the Streamlit app:

```bash
streamlit run app.py
```

The app now defaults to the final winning model in:

- artifact directory: `artifacts/model_search_revalidated_20260326`
- model: `focused_subsample_lr0048_iter1450`
- watch threshold: `0.16533940710070974`
- alert threshold: `0.7072099763613172`

Demo recommendation:

1. Open the app.
2. Show the risk score and risk band workflow.
3. Mention that the deployed demo uses the final verified winner, not an older baseline.

## Official Notebook Submission

If the hackathon form asks for a public Google Colab or Kaggle notebook link, use:

- `submission/Patient_Deterioration_Week1_Official_Submission_Notebook.ipynb`

What this notebook does:

- explains the dataset and project workflow
- shows the official four-experiment comparison
- verifies the final winner metrics exactly
- loads the retained full-train winning model
- generates blind validation predictions with the official winner

Related file:

- `submission/NOTEBOOK_SUBMISSION_GUIDE.md`

## Project Workflow

The project has four main modeling stages:

### 1. Baseline CatBoost

- Script: `train_model.py`
- Purpose: build the baseline tabular patient-deterioration model
- Main retained outputs:
  - `artifacts/deterioration_model.cbm`
  - `artifacts/metadata.json`
  - `artifacts/holdout_predictions.csv`

### 2. Deep Sequence Models

- Script: `train_deep_models.py`
- Purpose: train deep-learning sequence baselines such as the transformer encoder
- Main retained outputs:
  - `artifacts/deep_models/model_metric_details.json`
  - `artifacts/deep_models/model_metric_summary.md`
  - `artifacts/deep_models/model_comparison.csv`

### 3. Hybrid / Ensemble Search

- Script: `optimize_best_model.py`
- Purpose: combine tuned CatBoost and deep-model predictions into hybrid benchmarks
- Main retained outputs:
  - `artifacts/model_search/best_model_metric_details.json`
  - `artifacts/model_search/best_model_summary.json`
  - `artifacts/model_search/best_model_comparison.csv`

### 4. Focused Revalidation And Final Winner Selection

- Script family: `revalidate_model_search.py` plus later focused sweep artifacts
- Purpose: verify the search cleanly and preserve the final winner
- Main retained outputs:
  - `artifacts/model_search_revalidated_20260326/best_overall_summary.json`
  - `artifacts/model_search_revalidated_20260326/aggregate_best_results.csv`
  - `artifacts/model_search_revalidated_20260326/focused_subsample_round2_results.csv`
  - `artifacts/model_search_revalidated_20260326/focused_subsample_lr0048_iter1450_full_train_model.cbm`

## Data And Feature Engineering

The core modeling pipeline does the following:

- reconstructs `episode_id` from `hour_from_admission` resets
- engineers lag, delta, rolling, and clinically derived features
- uses group-aware splitting by episode to reduce leakage
- ranks models primarily by `PR-AUC`, then `ROC-AUC`
- keeps `dataset/val_no_labels.csv` for unlabeled inference only

Feature engineering is implemented in:

- `src/physio_warning/features.py`

The final focused winner uses:

- `212` engineered model features

## Quick Review Path For Hackathon Judges Or Teammates

If someone is new to the repo, this is the fastest review order:

1. Read this `README.md`
2. Open `submission/AesCodeNexus_Round1_Submission_Final_Technical_Patient_Deterioration_TeamReady.pptx`
3. Open `submission/Patient_Deterioration_Week1_Official_Submission_Notebook.ipynb`
4. Check `submission/week1_official_submission_results.csv`
5. Check `submission/official_winner_reproduced_metrics.csv`
6. If needed, inspect `artifacts/model_search_revalidated_20260326/best_overall_summary.json`
7. Run `streamlit run app.py` for the live demo

## Final Repository Components

Main files worth knowing:

- `app.py`
  Streamlit demo app using the final winner by default
- `train_model.py`
  Baseline CatBoost training
- `train_deep_models.py`
  Deep sequence-model training
- `optimize_best_model.py`
  Earlier hybrid and ensemble comparison workflow
- `revalidate_model_search.py`
  Conservative revalidation sweep over built-in candidate pools
- `src/physio_warning/features.py`
  Shared feature engineering
- `artifacts/model_search_revalidated_20260326/`
  Final focused winner and proof artifacts
- `submission/`
  Final submission bundle

## Important Limitations

- `episode_id` is reconstructed because the dataset does not include a native admission identifier
- `dataset/val_no_labels.csv` has no labels, so true blind-validation benchmarking depends on internal holdout evaluation
- the final `0.7396` result is a preserved official focused-sweep result, not something recreated by a single baseline rerun cell
- the root repo still contains older workflows for comparison, but the official final submission story is the one documented in this README

## Minimal Run Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the demo:

```bash
streamlit run app.py
```

Optional conservative revalidation check:

```bash
python revalidate_model_search.py --output-dir artifacts/model_search_revalidated_local_check
```

This local check is useful for verification, but the official final winner remains the preserved focused CatBoost model in `artifacts/model_search_revalidated_20260326`.

# Revalidated Model Search Findings

This folder contains the cleaned final outputs from the March 26, 2026 revalidation and focused CatBoost tuning pass run on the official provided dataset only.

## Dataset Used

- `dataset/train.csv`: `293,248` rows x `22` columns, `29,630,460` bytes
- `dataset/val_no_labels.csv`: `62,487` rows x `21` columns, `6,180,992` bytes
- Total provided dataset size: `35,811,452` bytes
- Target column: `deterioration_next_12h`

No external data was used.

## Evaluation Approach

- Reconstructed `episode_id` boundaries from `hour_from_admission` resets
- Used a group-aware split so the same admission did not leak across train and holdout
- Ranked models by holdout `PR-AUC` first, then `ROC-AUC`
- Compared repo baselines, focused CatBoost sweeps, random CatBoost candidates, and simple ensembles

## Best Model

- Model name: `focused_subsample_lr0048_iter1450`
- Holdout PR-AUC: `0.739613`
- Holdout ROC-AUC: `0.963087`
- Holdout Brier score: `0.027904`
- Previous repo best PR-AUC: `0.732291`
- Absolute PR-AUC gain vs previous repo best: `+0.007322`

### Final Parameters

- `iterations=1450`
- `depth=8`
- `learning_rate=0.048`
- `l2_leaf_reg=7.0`
- `random_strength=0.6`
- `bootstrap_type=Bernoulli`
- `subsample=0.88`
- `border_count=254`

## Overfitting Check

- Literal zero overfitting could not be verified honestly
- Train PR-AUC: `0.982226`
- Calibration PR-AUC: `0.746511`
- Final holdout PR-AUC: `0.739613`
- The internal train-to-calibration gap is large, but the calibration-to-holdout gap is small enough to support the final selection

## Files Kept In This Folder

- `best_overall_summary.json`: top-level summary of the winning run
- `aggregate_best_results.csv`: ranked strongest runs observed during search
- `combined_comparison.csv`: comparison against repo baselines and prior saved winners
- `screening_results.csv`: first-pass screening sweep across candidate models
- `ensemble_results.csv`: simple ensemble comparison results
- `focused_fulltrain_results.csv`: first focused retrain sweep
- `focused_subsample_results.csv`: first neighborhood refinement
- `focused_subsample_round2_results.csv`: second neighborhood refinement
- `focused_subsample_round3_results.csv`: final neighborhood refinement
- `focused_subsample_lr0048_iter1450_full_train_model.cbm`: final model trained on the full provided training set
- `focused_subsample_lr0048_iter1450_val_predictions.csv`: predictions for `dataset/val_no_labels.csv`
- `focused_subsample_lr0048_iter1450_best_holdout_predictions.csv`: predictions for the untouched reference holdout
- `focused_subsample_lr0048_iter1450_feature_importance.csv`: feature importance export for the winning model
- `focused_subsample_lr0048_iter1450_final_artifact_summary.json`: winning model parameters and reference metrics
- `focused_subsample_lr0048_iter1450_overfit_check.json`: compact overfitting check
- `search_summary.json`: compact search metadata and outcome summary

Intermediate scratch files and duplicate prediction dumps were intentionally removed to keep the repo submission-friendly. The full search can be regenerated with `python revalidate_model_search.py`.

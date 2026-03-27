# Notebook Submission Guide

Use `submission/Patient_Deterioration_Week1_Official_Submission_Notebook.ipynb` as the official notebook submission file.

What the notebook contains:

- dataset overview and quick analysis
- feature engineering summary
- project workflow explanation
- official four-experiment comparison
- exact verification of the final winning model
- blind validation scoring with the retained winning model

Recommended submission flow:

1. Open the notebook in Google Colab or Kaggle.
2. Run all cells.
3. Confirm that `official_submission_table` shows the final four-experiment comparison.
4. Confirm that `official_reproduced_metrics` shows the winning model `focused_subsample_lr0048_iter1450`.
5. Upload or keep the generated CSV outputs if needed.
6. Set notebook visibility to public or anyone-with-the-link before submitting the form.

Key notebook outputs:

- `submission/week1_official_submission_results.csv`
- `submission/official_winner_reproduced_metrics.csv`
- `submission/focused_subsample_lr0048_iter1450_official_submission_predictions.csv`

Important note:

- The official notebook is aligned with the verified final artifact files in `artifacts/model_search_revalidated_20260326`
- It is the correct notebook to share for the hackathon submission

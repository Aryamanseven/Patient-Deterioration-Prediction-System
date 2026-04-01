# Dataset Folder

This folder stores competition input files.

## Files

1. train.csv
   Labeled training data used by pipeline training stages.

2. val_no_labels.csv
   Validation/inference-style data without target labels.

## Dependency notes

1. `configs/*.yaml` profiles reference this folder as default input location.
2. `core/data_loader.py` and notebook workflows read these files directly.

## Minimal expected columns

1. Time and grouping fields required by feature engineering and sequence preparation.
2. Target field in training data (`train.csv`) for supervised learning.
3. Matching feature schema in unlabeled validation file for scoring workflows.

## Rules

1. Treat source data as read-only.
2. Do not commit private or unauthorized data files.
3. Any preprocessing should occur in code, not by mutating these files.

## Sharing guidance

1. Keep only required shareable data in repository.
2. Large private raw exports should be stored externally and documented, not committed.

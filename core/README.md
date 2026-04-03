# Core Module

This package contains the shared data and utility foundation used by every pipeline stage.

## Files and purpose

1. config.py
	Loads YAML, validates required fields, resolves device, and creates run output folders.

2. data_loader.py
	Reads dataset CSV, applies feature engineering, performs group-aware split, and builds sequence datasets.

3. features.py
	Defines engineered feature families and canonical feature column lists.

4. clinical_scores.py
	Computes bedside-inspired scores (NEWS, MEWS, qSOFA) used as model inputs.

5. metrics.py
	Computes classification metrics (PR-AUC, ROC-AUC and related diagnostics).

6. reproducibility.py
	Sets global random seeds for consistent runs.

7. logger.py
	Unified logging style across pipeline and modules.

## Invariants

1. The split must be group-aware by episode to avoid leakage.
2. Feature engineering is centralized and reused by all model paths.
3. Config is the source of truth for optional module enablement.

## Runtime notes

1. On DirectML, some PyTorch operators can fall back to CPU; this is expected and can reduce speed.
2. Sequence dataset code intentionally uses contiguous arrays for safer tensor conversion.

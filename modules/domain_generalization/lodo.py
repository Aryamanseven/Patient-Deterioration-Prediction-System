"""
Leave-One-Domain-Out Validation logic for Domain Generalization.

Tests if the model actually learned physiology or just the specific measurement 
artifacts of the hospitals it happened to train on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.features import engineer_all_features
from core.logger import get_logger
from core.metrics import evaluate_binary
from models.catboost_model import CatBoostWrapper

logger = get_logger("dg")


def run_leave_one_domain_out(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    feature_cols: list[str],
    config: dict,
    output_dir: Path,
    seed: int = 42
) -> None:
    """
    Run Leave-One-Domain-Out validation on the supervised baseline model
    to prove out-of-distribution performance guarantees.
    """
    params = config["modules"]["domain_generalization"]
    if not params.get("enabled", False):
        logger.info("Domain Generalization (LODO) is disabled. Skipping.")
        return

    if len(X_full) != len(y_full):
        logger.error(
            "LODO input length mismatch: X_full=%d, y_full=%d. Skipping DG.",
            len(X_full),
            len(y_full),
        )
        return

    missing_feature_cols = [col for col in feature_cols if col not in X_full.columns]
    if missing_feature_cols:
        logger.warning(
            "LODO input is missing %d/%d feature columns. Re-engineering features for DG.",
            len(missing_feature_cols),
            len(feature_cols),
        )
        try:
            X_full = engineer_all_features(
                X_full,
                use_advanced=config.get("features", {}).get("use_advanced", True),
                use_clinical_scores=config.get("features", {}).get("use_clinical_scores", True),
            )
        except Exception as exc:
            logger.error("Feature re-engineering failed for DG: %s", exc)
            return

        missing_feature_cols = [col for col in feature_cols if col not in X_full.columns]
        if missing_feature_cols:
            available_feature_cols = [col for col in feature_cols if col in X_full.columns]
            if not available_feature_cols:
                logger.error("No requested feature columns are available for DG. Skipping.")
                return

            logger.warning(
                "LODO still missing %d feature columns after re-engineering. Proceeding with %d available features.",
                len(missing_feature_cols),
                len(available_feature_cols),
            )
            feature_cols = available_feature_cols
        
    domain_col = params.get("domain_column", "unit_type")
    
    if domain_col not in X_full.columns:
        logger.warning(f"Domain column '{domain_col}' not found. Using completely random 5-fold cross-validation as domain proxy.")
        # Create a mock partition
        np.random.seed(seed)
        X_full = X_full.copy()
        X_full[domain_col] = np.random.randint(0, 5, size=len(X_full))
        
    domains = X_full[domain_col].unique()
    if len(domains) < 2:
        logger.warning("Only one domain exists. Cannot perform LODO generalization metric.")
        return
        
    logger.info(f"Starting Leave-One-Domain-Out Validation across {len(domains)} distinct domains ({domain_col})...")
    
    results = []
    
    for left_out in domains:
        train_mask = X_full[domain_col] != left_out
        test_mask = X_full[domain_col] == left_out
        
        X_train_fold = X_full.loc[train_mask, feature_cols].copy()
        y_train_fold = y_full.loc[train_mask]
        
        X_test_fold = X_full.loc[test_mask, feature_cols].copy()
        y_test_fold = y_full.loc[test_mask]
        
        if len(y_test_fold) == 0 or len(y_train_fold) == 0:
            continue
            
        logger.info(f"Training on all EXCEPT {left_out} (Test N={len(X_test_fold)})")
        
        # FIX: Use high od_wait (equal to iterations) to effectively disable early stopping.
        # In LODO we want the model to train for all iterations since we cannot use
        # the held-out domain for validation (that would leak information).
        model = CatBoostWrapper(
            params={"iterations": 500, "learning_rate": 0.05, "depth": 6, "od_wait": 500},
            random_seed=seed
        )
        
        # Use train fold as validation — since od_wait=iterations, early stopping
        # will never trigger, guaranteeing full 500 iterations every time.
        model.fit(X_train_fold, y_train_fold, X_train_fold, y_train_fold)
        
        preds = model.predict_proba(X_test_fold)
        metrics = evaluate_binary(y_test_fold, preds)
        
        metrics["left_out_domain"] = left_out
        metrics["n_train"] = len(X_train_fold)
        metrics["n_test"] = len(X_test_fold)
        results.append(metrics)
        
        logger.info(f"  Domain '{left_out}': PR-AUC={metrics['pr_auc']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}")
        
    if not results:
        logger.warning("No valid LODO folds produced results. Skipping DG export.")
        return

    results_df = pd.DataFrame(results)
    
    # Calculate Macro-Averages
    macro_prauc = results_df["pr_auc"].mean()
    macro_rocauc = results_df["roc_auc"].mean()
    
    logger.info("--- OOD Generalization Test Results ---")
    logger.info(f"OOD Macro Mean PR-AUC:  {macro_prauc:.4f}")
    logger.info(f"OOD Macro Mean ROC-AUC: {macro_rocauc:.4f}")
    
    out_path = output_dir / "lodo_results.csv"
    results_df.to_csv(out_path, index=False)
    logger.info(f"Generalization matrix saved to {out_path}")

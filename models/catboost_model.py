"""
CatBoost Model Wrapper.

Wraps the CatBoost parameter handling, categorical feature identification,
and early stopping logic into a clean interface.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from core.features import CATEGORICAL_FEATURE_COLUMNS
from core.logger import get_logger

logger = get_logger("catboost")


class CatBoostWrapper:
    """Wrapper for training and predicting with CatBoost."""

    def __init__(self, params: dict[str, Any], device: str = "cpu", random_seed: int = 42):
        self.params = params.copy()
        self.params["random_seed"] = random_seed
        
        # Smart Hardware Fallback for CatBoost
        if "task_type" not in self.params:
            if device.lower() == "cuda":
                self.params["task_type"] = "GPU"
            else:
                self.params["task_type"] = "CPU"
                
        self.model = CatBoostClassifier(**self.params)
        self.feature_cols: list[str] = []
        self.cat_indices: list[int] = []

    def _get_cat_indices(self, feature_cols: list[str]) -> list[int]:
        """Find the exact column indices for categorical features."""
        indices = []
        for i, col in enumerate(feature_cols):
            if col in CATEGORICAL_FEATURE_COLUMNS:
                indices.append(i)
        return indices

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> "CatBoostWrapper":
        """Train the model with early stopping."""
        self.feature_cols = list(X_train.columns)
        self.cat_indices = self._get_cat_indices(self.feature_cols)

        logger.info(f"Training CatBoost with {len(self.feature_cols)} features...")
        logger.info(f"Categorical features: {[self.feature_cols[i] for i in self.cat_indices]}")

        # Ensure categoricals are properly cast to strings for CatBoost
        X_train_cb = X_train.copy()
        X_val_cb = X_val.copy()
        for idx in self.cat_indices:
            col = self.feature_cols[idx]
            X_train_cb[col] = X_train_cb[col].astype(str)
            X_val_cb[col] = X_val_cb[col].astype(str)

        self.model.fit(
            X_train_cb,
            y_train,
            eval_set=(X_val_cb, y_val),
            cat_features=self.cat_indices,
            verbose=100,
        )

        logger.info(f"Training finished. Best iteration: {self.model.get_best_iteration()}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for the positive class."""
        X_cb = X[self.feature_cols].copy()
        for idx in self.cat_indices:
            col = self.feature_cols[idx]
            X_cb[col] = X_cb[col].astype(str)
            
        return self.model.predict_proba(X_cb)[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importances sorted descending."""
        importance = self.model.get_feature_importance()
        df = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": importance
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        """Save the model to disk."""
        path = str(path)
        self.model.save_model(path)
        logger.info(f"CatBoost model saved to {path}")

    def load(self, path: str | Path) -> "CatBoostWrapper":
        """Load the model from disk."""
        path = str(path)
        self.model.load_model(path)
        logger.info(f"CatBoost model loaded from {path}")
        return self

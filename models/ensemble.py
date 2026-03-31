"""
Ensemble module.

Combines predictions from multiple models (e.g. CatBoost + Deep Learning)
using optimized weighted blending.
"""
from __future__ import annotations

import numpy as np

from core.logger import get_logger
from core.metrics import evaluate_binary

logger = get_logger("ensemble")


class EnsembleModel:
    """Finds optimal blend weights between multiple models to maximize a metric."""
    
    def __init__(self, method: str = "weighted_blend"):
        self.method = method
        self.weights: np.ndarray | None = None
        
    def fit(self, preds_list: list[np.ndarray], y_true: np.ndarray, metric: str = "pr_auc") -> "EnsembleModel":
        """
        preds_list: List of 1D arrays of probabilities from each model.
        Finds weights that maximize `metric`.
        """
        if len(preds_list) < 2:
            logger.warning("Ensemble needs at least 2 models. Returning trivial ensemble.")
            self.weights = np.array([1.0] * len(preds_list)) / len(preds_list)
            return self
            
        logger.info(f"Optimizing ensemble weights for {len(preds_list)} models using metric: {metric}...")
        
        preds_matrix = np.column_stack(preds_list)  # (N, models)
        
        # Grid search over weights with step 0.05
        best_score = -1.0
        best_weights = np.ones(len(preds_list)) / len(preds_list)
        
        # We assume 2 models (e.g. CatBoost + DL) for simplicity of grid search, 
        # but generalize with random search if > 2
        if len(preds_list) == 2:
            for w in np.linspace(0, 1, 21):
                w_arr = np.array([w, 1.0 - w])
                ensemble_pred = np.sum(preds_matrix * w_arr, axis=1)
                
                # Evaluate using threshold=0.5 (threshold tuning happens AFTER ensemble)
                metrics = evaluate_binary(y_true, ensemble_pred, threshold=0.5)
                score = metrics[metric]
                
                if score > best_score:
                    best_score = score
                    best_weights = w_arr
        else:
            # Random search for >2 models
            for _ in range(500):
                w_arr = np.random.dirichlet(np.ones(len(preds_list)))
                ensemble_pred = np.sum(preds_matrix * w_arr, axis=1)
                metrics = evaluate_binary(y_true, ensemble_pred, threshold=0.5)
                score = metrics[metric]
                
                if score > best_score:
                    best_score = score
                    best_weights = w_arr
                    
        self.weights = best_weights
        logger.info(f"Optimal ensemble weights found: {self.weights} (score: {best_score:.4f})")
        return self

    def predict_proba(self, preds_list: list[np.ndarray]) -> np.ndarray:
        """Apply learned weights to new predictions."""
        if self.weights is None:
            raise ValueError("Ensemble has not been fitted.")
        preds_matrix = np.column_stack(preds_list)
        return np.sum(preds_matrix * self.weights, axis=1)

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"weights": self.weights, "method": self.method}, f)
        logger.info(f"Ensemble weights saved to {path}")

    @classmethod
    def load(cls, path: str) -> "EnsembleModel":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(method=data["method"])
        model.weights = data["weights"]
        return model

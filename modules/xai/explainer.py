"""
Explainability logic — SHAP based insights.

Calculates how much each feature contributed to the model predictions.
Generates summary plots and artifacts for the dashboard and reports.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.logger import get_logger

logger = get_logger("xai")


def run_xai_analysis(
    model_wrapper: Any,
    X_val: pd.DataFrame,
    config: dict,
    output_dir: Path,
) -> None:
    """
    Run SHAP analysis on the supervised model (CatBoost) to understand
    which novel features are driving predictions.
    """
    params = config["modules"]["xai"]
    if not params.get("enabled", False):
        logger.info("XAI is disabled. Skipping.")
        return
        
    num_samples = params.get("shap_samples", 1000)
    
    logger.info(f"Running XAI (SHAP) on {min(num_samples, len(X_val))} samples...")
    
    try:
        import shap
        import matplotlib.pyplot as plt
        # CatBoost has native SHAP support and fast tree explainer
        
        # Take a sample if dataset is too large
        if len(X_val) > num_samples:
            X_sample = X_val.sample(n=num_samples, random_state=42)
        else:
            X_sample = X_val.copy()
            
        # CatBoost requires string categoricals for prediction
        from core.features import CATEGORICAL_FEATURE_COLUMNS
        feature_cols = [c for c in X_sample.columns]
        
        # SHAP calculation over Tree models
        explainer = shap.TreeExplainer(model_wrapper.model)
        
        X_sample_cat = X_sample.copy()
        for col in CATEGORICAL_FEATURE_COLUMNS:
            if col in X_sample_cat.columns:
                X_sample_cat[col] = X_sample_cat[col].astype(str)
                
        shap_values = explainer.shap_values(X_sample_cat)
        
        # Save summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample_cat, show=False)
        plt.tight_layout()
        plot_path = output_dir / "shap_summary.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP summary plot to {plot_path}")
        
        # Calculate mean absolute SHAP values for feature importance ranking
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": X_sample.columns,
            "mean_abs_shap": mean_shap
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        
        csv_path = output_dir / "top_features.csv"
        shap_df.to_csv(csv_path, index=False)
        logger.info(f"Saved XAI feature importance to {csv_path}")
        
        # Log top 5 directly to console
        logger.info("Top 5 clinical drivers discovered by AI:")
        for i, row in shap_df.head(5).iterrows():
            logger.info(f"  {i+1}. {row['feature']} (SHAP: {row['mean_abs_shap']:.4f})")
            
    except ImportError:
        logger.warning("SHAP or Matplotlib not installed. Skipping XAI plots.")
    except Exception as e:
        logger.error(f"XAI analysis failed: {e}")

def run_captum_analysis(
    dl_model_wrapper: Any,
    X_val_seq: np.ndarray,
    val_masks: np.ndarray, 
    val_static: np.ndarray,
    config: dict,
    output_dir: Path,
) -> None:
    """
    Run Captum IntegratedGradients analysis on the Deep Learning Model (TCN-Transformer)
    to generate temporal heatmaps of feature importance.
    """
    params = config["modules"]["xai"]
    if not params.get("enabled", False):
        return
        
    logger.info("Running Deep Learning XAI (Captum) on temporal sequences...")
    
    try:
        import torch
        from captum.attr import IntegratedGradients
        import matplotlib.pyplot as plt

        try:
            import seaborn as sns
            use_seaborn = True
        except ImportError:
            sns = None
            use_seaborn = False
            logger.info("Seaborn not installed; using matplotlib heatmap fallback.")
        
        device = dl_model_wrapper.device
        model = dl_model_wrapper.model
        model.eval()
        
        # Take a single representative batch for heatmap generation
        sample_size = min(params.get("shap_samples", 32), len(X_val_seq))
        X_seq_t = torch.tensor(X_val_seq[:sample_size], dtype=torch.float32, device=device)
        masks_t = torch.tensor(val_masks[:sample_size], dtype=torch.bool, device=device)
        static_t = torch.tensor(val_static[:sample_size], dtype=torch.float32, device=device)
        
        # Wrapper to forward only X_seq natively for Captum
        class ModelCaptumWrapper(torch.nn.Module):
            def __init__(self, base_model, masks, static):
                super().__init__()
                self.base_model = base_model
                self.masks = masks
                self.static = static
            def forward(self, x):
                return self.base_model(x, self.masks, self.static).unsqueeze(-1)
                
        wrapper = ModelCaptumWrapper(model, masks_t, static_t)
        
        ig = IntegratedGradients(wrapper)
        # Attributions for the sequential input
        baseline = torch.zeros_like(X_seq_t)
        attributions, delta = ig.attribute(X_seq_t, baseline, return_convergence_delta=True)
        
        attributions_np = attributions.cpu().detach().numpy()
        
        # Average attributions over batch
        mean_attr = np.mean(np.abs(attributions_np), axis=0)
        
        # Generate Heatmap
        plt.figure(figsize=(12, 8))
        if use_seaborn:
            sns.heatmap(
                mean_attr.T,
                cmap="viridis",
                cbar_kws={"label": "Mean Integrated Gradients"},
            )
        else:
            im = plt.imshow(mean_attr.T, aspect="auto", cmap="viridis", origin="lower")
            cbar = plt.colorbar(im)
            cbar.set_label("Mean Integrated Gradients")
        plt.title("Temporal Feature Importance (Captum Heatmap)")
        plt.xlabel("Time Step")
        plt.ylabel("Feature Index")
        plt.tight_layout()
        
        plot_path = output_dir / "captum_temporal_heatmap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved Captum temporal heatmap to {plot_path}")
        
    except ImportError:
        logger.warning("Captum or Matplotlib not installed. Skipping Deep Learning XAI plots.")
    except Exception as e:
        logger.error(f"Captum analysis failed: {e}")

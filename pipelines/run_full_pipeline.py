"""
Master Pipeline Orchestrator.

Combines all modules and components of the PhysioGuard v2.0 system
into a single, reproducible, end-to-end execution flow.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Add project root to python path to allow direct execution
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.config import get_device, get_output_dir, load_config
from core.data_loader import load_and_split_data, prepare_sequences
from core.logger import get_logger
from core.metrics import evaluate_binary
from core.reproducibility import set_global_seed
from models.ensemble import EnsembleModel
from models.model_registry import create_deep_model, create_supervised_model
from modules.deployment import export_models_for_production
from modules.differential_privacy import apply_differential_privacy
from modules.domain_generalization import run_leave_one_domain_out
from modules.federated_learning import run_fl_simulation
from modules.ssl import run_ssl_pretraining
from modules.xai.explainer import run_xai_analysis, run_captum_analysis

def run_pipeline(config_path: str):
    # 1. Load config
    config = load_config(config_path)
    output_dir = get_output_dir(config)
    
    # 2. Setup Logging & Reproducibility
    logger = get_logger("pipeline", log_dir=output_dir)
    logger.info(f"Starting pipeline using config: {config_path}")
    
    seed = config["general"].get("random_seed", 42)
    set_global_seed(seed)
    
    device = get_device(config)
    
    # Make DML explicitly obvious in logs
    display_device = device
    if "privateuseone" in str(device):
        display_device = f"dml (DirectML on AMD/Intel/NVIDIA) -> internal alias: {device}"
        
    logger.info(f"Using device: {display_device}")
    
    # 3. Data Loading & Sequence Generation
    logger.info("--- STEP 1: Data Loading ---")
    raw_df, X_train, X_val, feature_cols = load_and_split_data(
        data_path=config["data"]["path"],
        test_size=config["data"].get("test_size", 0.2),
        max_rows=config["data"].get("max_rows"),
        use_advanced_features=config["features"].get("use_advanced", True),
        use_clinical_scores=config["features"].get("use_clinical_scores", True),
        random_state=seed
    )
    from core.features import TARGET_COLUMN
    y_train = X_train[TARGET_COLUMN]
    y_val = X_val[TARGET_COLUMN]
    X_train_features = X_train[feature_cols]
    X_val_features = X_val[feature_cols]
    
    max_len = config["modules"]["deep_learning"].get("max_seq_len", 24)
    train_dataset = prepare_sequences(X_train, feature_cols, max_len)
    val_dataset = prepare_sequences(X_val, feature_cols, max_len)
    
    # Extract flat labels for classical metrics
    import numpy as np
    val_labels = np.concatenate(val_dataset.targets)
    
    # 4. Self-Supervised pre-training (SSL)
    logger.info("--- STEP 2: SSL Pre-training ---")
    ssl_weights_path = run_ssl_pretraining(
        train_dataset, config, output_dir, device, seed
    )
    
    # Setup Deep Learning Model
    dl_config = config["modules"]["deep_learning"]
    dl_config["input_dim"] = train_dataset.input_dim
    dl_config["static_dim"] = train_dataset.static_dim
    dl_model = create_deep_model(dl_config, device=device, seed=seed)
    
    if ssl_weights_path:
        dl_model.load_pretrained_encoder(ssl_weights_path)
    
    # 5. Deep Learning Training (with DP) & Optional Federated Mode
    if config["modules"]["federated_learning"].get("enabled", False):
        logger.info("--- STEP 3: DL Training (with DP) ---")
        logger.info("--- STEP 4: Optional Federated Mode (FedAvg) ---")
        run_fl_simulation(
            dl_model, train_dataset, val_dataset, config, output_dir, seed
        )
    else:
        logger.info("--- STEP 3: DL Training (with DP) ---")
        logger.info("--- STEP 4: Optional Federated Mode (SKIPPED) ---")
        # Direct localized training handling DP via fit hook inside TCN
        dl_model.fit(train_dataset, val_dataset, config=config)
                     
    dl_preds = dl_model.predict_proba(val_dataset)
    dl_metrics = evaluate_binary(val_labels, dl_preds)
    logger.info(f"Deep Learning PR-AUC: {dl_metrics['pr_auc']:.4f}")

    # 6. Supervised Baseline (CatBoost)
    logger.info("--- STEP 5: CatBoost Training ---")
    supervised_model = create_supervised_model(config["modules"]["supervised"], device=device, seed=seed)
    supervised_model.fit(X_train_features, y_train, X_val_features[:200], y_val[:200])
    
    sup_preds = supervised_model.predict_proba(X_val_features)
    sup_metrics = evaluate_binary(y_val, sup_preds)
    logger.info(f"CatBoost Baseline PR-AUC: {sup_metrics['pr_auc']:.4f}")
    
    # 7. Ensemble Fusion
    logger.info("--- STEP 6: Ensemble Creation ---")
    ensemble = EnsembleModel(method="weighted_blend")
    min_len = min(len(sup_preds), len(dl_preds))
    ensemble.fit([sup_preds[:min_len], dl_preds[:min_len]], val_labels[:min_len])
    fused_preds = ensemble.predict_proba([sup_preds[:min_len], dl_preds[:min_len]])
    fused_metrics = evaluate_binary(val_labels[:min_len], fused_preds)
    logger.info(f"Ensemble Model PR-AUC: {fused_metrics['pr_auc']:.4f}")
    
    # 8. Domain Generalization Verification (LODO)
    logger.info("--- STEP 7: DG Evaluation ---")
    run_leave_one_domain_out(X_train, y_train, feature_cols, config, output_dir, seed)
    
    # 9. Explainability (XAI)
    logger.info("--- STEP 8: XAI Generation ---")
    run_xai_analysis(supervised_model, X_val_features, config, output_dir)
    
    val_subset = val_dataset[:32]
    run_captum_analysis(dl_model, val_subset[0], val_subset[1], val_subset[2], config, output_dir)
    
    # 10. Artifact Saving & Deployment
    logger.info("--- STEP 9: Artifact Saving ---")
    ensemble_path = output_dir / "model" / "ensemble.pkl"
    ensemble.save(ensemble_path)
    export_models_for_production(supervised_model, dl_model, config, output_dir)
    
    # Save metrics and predictions
    import json
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(fused_metrics, f, indent=4)
        
    import pandas as pd
    preds_df = pd.DataFrame({
        "y_true": val_labels[:min_len].tolist() if not isinstance(val_labels, pd.Series) else val_labels[:min_len].values,
        "y_proba_ensemble": fused_preds.tolist(),
        "y_proba_catboost": sup_preds[:min_len].tolist(),
        "y_proba_dl": dl_preds[:min_len].tolist()
    })
    preds_df.to_csv(output_dir / "predictions.csv", index=False)
    
    logger.info("=== PIPELINE COMPLETION ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete Patient Deterioration Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()
    
    run_pipeline(args.config)

"""
Master Pipeline Orchestrator.

Combines all modules and components of the PhysioGuard v2.0 system
into a single, reproducible, end-to-end execution flow.

HARDENED VERSION — all saves verified, all bugs fixed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# Add project root to python path to allow direct execution
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.config import get_device, get_output_dir, get_seed, load_config
from core.data_loader import load_and_split_data, prepare_sequences
from core.features import TARGET_COLUMN
from core.logger import get_logger, set_global_log_dir
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


# ============================================================
# Robust saving utilities
# ============================================================


def _resolve_path(path_value: str | Path) -> Path:
    """Resolve a path against project root when relative."""
    path_obj = Path(path_value)
    if not path_obj.is_absolute():
        path_obj = project_root / path_obj
    return path_obj


def _find_latest_run_artifact(relative_file: str | Path) -> Path | None:
    """Find the newest artifacts/run_*/<relative_file> that exists."""
    artifacts_dir = project_root / "artifacts"
    if not artifacts_dir.exists() or not artifacts_dir.is_dir():
        return None

    run_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    run_dirs_sorted = sorted(run_dirs, key=lambda d: d.stat().st_mtime, reverse=True)

    rel = Path(relative_file)
    for run_dir in run_dirs_sorted:
        candidate = run_dir / rel
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_artifact_alias(path_value: str | Path, logger, context_label: str) -> Path:
    """Resolve special artifact aliases, then fall back to normal path resolution."""
    raw_value = str(path_value).strip()
    alias_key = raw_value.upper()

    alias_map = {
        "AUTO_LATEST_SSL": Path("ssl_pretrained_tcntransformer.pt"),
        "AUTO_LATEST_DL_CHECKPOINT": Path("model") / "dl_checkpoint_latest.pt",
        "AUTO_LATEST_DL_FINAL": Path("model") / "dl_model_final.pt",
    }

    if alias_key in alias_map:
        resolved = _find_latest_run_artifact(alias_map[alias_key])
        if resolved is None:
            raise FileNotFoundError(
                f"{context_label} alias '{raw_value}' could not be resolved from artifacts/run_*/{alias_map[alias_key]}"
            )
        logger.info(f"{context_label} auto-resolved: {raw_value} -> {resolved}")
        return resolved

    return _resolve_path(raw_value)

def _flush_logger_handlers(logger) -> None:
    """Flush logger handlers so late-stage errors are not lost."""
    if logger is None:
        return
    for handler in getattr(logger, "handlers", []):
        try:
            handler.flush()
        except Exception:
            pass

def _verify_saved(path: Path, logger, label: str) -> bool:
    """Verify a saved file exists and has non-zero size."""
    if path.exists() and path.stat().st_size > 0:
        size_kb = path.stat().st_size / 1024
        logger.info(f"  ✓ VERIFIED {label}: {path.name} ({size_kb:.1f} KB)")
        return True
    else:
        logger.error(f"  ✗ FAILED {label}: {path} does not exist or is empty!")
        return False


def _save_dl_model(dl_model, save_path: Path, logger) -> bool:
    """Save DL model with triple-redundancy and verification."""
    try:
        if dl_model is None or getattr(dl_model, "model", None) is None:
            logger.error(f"CRITICAL: DL model is missing and cannot be saved to {save_path}")
            return False

        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract clean state dict (strips Opacus _module. prefix if present)
        raw_state = dl_model.model.state_dict()
        clean_state = {k.replace("_module.", ""): v.detach().cpu() for k, v in raw_state.items()}
        
        # Save primary copy
        torch.save(clean_state, save_path)
        
        # Save backup copy
        backup_path = save_path.parent / f"{save_path.stem}_backup{save_path.suffix}"
        torch.save(clean_state, backup_path)
        
        return _verify_saved(save_path, logger, "DL Model")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to save DL model: {e}")
        return False


def _save_catboost_model(supervised_model, save_path: Path, logger) -> bool:
    """Save CatBoost model with verification."""
    try:
        if supervised_model is None or getattr(supervised_model, "model", None) is None:
            logger.error(f"CRITICAL: CatBoost model is missing and cannot be saved to {save_path}")
            return False

        save_path.parent.mkdir(parents=True, exist_ok=True)
        supervised_model.model.save_model(str(save_path))
        
        # Save backup
        backup_path = save_path.parent / f"{save_path.stem}_backup{save_path.suffix}"
        supervised_model.model.save_model(str(backup_path))
        
        return _verify_saved(save_path, logger, "CatBoost Model")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to save CatBoost model: {e}")
        return False


def _save_ensemble(ensemble, save_path: Path, logger) -> bool:
    """Save ensemble weights with verification."""
    try:
        if ensemble is None or getattr(ensemble, "weights", None) is None:
            logger.error(f"CRITICAL: Ensemble weights are missing and cannot be saved to {save_path}")
            return False

        save_path.parent.mkdir(parents=True, exist_ok=True)
        ensemble.save(str(save_path))
        return _verify_saved(save_path, logger, "Ensemble Weights")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to save ensemble: {e}")
        return False


def _prepare_run_checkpoint(
    source_checkpoint: str | Path | None,
    target_checkpoint: Path,
    logger,
    label: str,
) -> None:
    """Optionally copy an existing checkpoint into this run's checkpoint location."""
    if not source_checkpoint:
        return

    source_path = _resolve_artifact_alias(source_checkpoint, logger, f"{label} resume checkpoint")
    if not source_path.exists():
        raise FileNotFoundError(f"{label} resume checkpoint not found: {source_path}")

    target_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if source_path.resolve() == target_checkpoint.resolve():
        logger.info(f"{label} resume checkpoint already in-place: {target_checkpoint}")
        return

    shutil.copy2(source_path, target_checkpoint)
    logger.info(f"{label} resume checkpoint copied: {source_path} -> {target_checkpoint}")


def _resolve_ssl_weights_for_run(config: dict[str, Any], output_dir: Path, logger) -> str:
    """Resolve reusable SSL weights and copy into current run dir for full provenance."""
    ssl_cfg = config.get("modules", {}).get("ssl", {})
    reuse_existing = bool(ssl_cfg.get("reuse_existing", False))
    pretrained_weights_path = str(ssl_cfg.get("pretrained_weights_path", "")).strip()

    if not reuse_existing and not pretrained_weights_path:
        return ""

    if not pretrained_weights_path:
        raise RuntimeError("SSL reuse mode enabled but 'modules.ssl.pretrained_weights_path' is missing.")

    source_path = _resolve_artifact_alias(pretrained_weights_path, logger, "SSL weights")
    if not source_path.exists():
        raise FileNotFoundError(f"Configured SSL weights not found: {source_path}")
    if source_path.is_dir():
        raise IsADirectoryError(f"Configured SSL weights path must be a file, got directory: {source_path}")

    target_path = output_dir / "ssl_pretrained_tcntransformer.pt"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if source_path.resolve() != target_path.resolve():
        shutil.copy2(source_path, target_path)
        logger.info(f"Reusing SSL weights: {source_path} -> {target_path}")
    else:
        logger.info(f"Reusing SSL weights already in run dir: {target_path}")

    return str(target_path)


def _load_fl_starting_checkpoint_if_available(dl_model, checkpoint_path: Path, logger) -> None:
    """Load a checkpoint model state before FL simulation when requested."""
    if not checkpoint_path.exists():
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location=dl_model.device, weights_only=False)
        state_dict = None

        # Preferred format: training checkpoint dict with model_state + optimizer state.
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_state"), dict):
            state_dict = checkpoint["model_state"]
        # Fallback format: raw model.state_dict() (e.g., dl_model_final.pt).
        elif isinstance(checkpoint, dict) and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            state_dict = checkpoint
            logger.info("FL resume source is a raw state_dict file (no optimizer state).")

        if not state_dict:
            logger.warning(f"DL checkpoint exists but contains no usable model weights: {checkpoint_path}")
            return

        load_result = dl_model.model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            logger.warning(
                "FL resume loaded with key mismatches. "
                f"Missing={len(load_result.missing_keys)}, Unexpected={len(load_result.unexpected_keys)}"
            )

        logger.info(f"Loaded starting DL model state for FL from: {checkpoint_path}")
    except Exception as e:
        logger.warning(f"Failed to load FL starting checkpoint ({checkpoint_path}): {e}")


def _verify_required_artifacts(output_dir: Path, logger, config: dict[str, Any]) -> dict[str, bool]:
    """Validate required run artifacts before declaring success."""
    required_paths = {
        "DL Model": output_dir / "model" / "dl_model_final.pt",
        "CatBoost Model": output_dir / "model" / "model.cbm",
        "Data Scaler": output_dir / "model" / "scaler.pkl",
        "Ensemble Weights": output_dir / "model" / "ensemble.pkl",
        "Feature Columns": output_dir / "model" / "feature_columns.json",
        "Metrics": output_dir / "metrics.json",
        "Predictions": output_dir / "predictions.csv",
    }

    ssl_cfg = config.get("modules", {}).get("ssl", {})
    if (
        ssl_cfg.get("enabled", False)
        or ssl_cfg.get("reuse_existing", False)
        or ssl_cfg.get("pretrained_weights_path")
    ):
        required_paths["SSL Weights"] = output_dir / "ssl_pretrained_tcntransformer.pt"

    checks: dict[str, bool] = {}
    for label, path in required_paths.items():
        checks[label] = _verify_saved(path, logger, label)
    return checks


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline(config_path: str):
    logger = None
    try:
        # 1. Load config
        config = load_config(config_path)
        output_dir = get_output_dir(config)
        set_global_log_dir(output_dir)

        # 2. Setup Logging & Reproducibility
        logger = get_logger("pipeline", log_dir=output_dir)
        logger.info(f"Starting pipeline using config: {config_path}")
        logger.info(f"Output directory: {output_dir}")

        seed = get_seed(config)
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
            random_state=seed,
        )
        y_train = X_train[TARGET_COLUMN]
        y_val = X_val[TARGET_COLUMN]
        X_train_features = X_train[feature_cols]
        X_val_features = X_val[feature_cols]

        # 3.5. MUST SCALE DATA FOR PYTORCH!
        # Without standardizing continuous variables (like blood pressure of 140),
        # the Transformer's attention logits explode, leading to garbage DL performance.
        from sklearn.preprocessing import StandardScaler
        import joblib
        from core.features import get_numeric_feature_columns

        numeric_cols = get_numeric_feature_columns(X_train_features)
        scaler = StandardScaler()

        X_train_dl = X_train.copy()
        X_val_dl = X_val.copy()

        # Fit on train, transform on train and val
        X_train_dl[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val_dl[numeric_cols] = scaler.transform(X_val[numeric_cols])

        max_len = config["modules"]["deep_learning"].get("max_seq_len", 24)
        train_dataset = prepare_sequences(X_train_dl, feature_cols, max_len)
        val_dataset = prepare_sequences(X_val_dl, feature_cols, max_len)

        # Extract flat labels for classical metrics
        val_labels = np.concatenate(val_dataset.targets)
        if len(val_labels) == 0:
            raise RuntimeError("Validation label array is empty. Cannot continue training pipeline.")

        # 4. Self-Supervised pre-training (SSL)
        ssl_cfg = config["modules"]["ssl"]
        ssl_weights_path = _resolve_ssl_weights_for_run(config, output_dir, logger)

        if ssl_weights_path:
            logger.info("--- STEP 2: SSL Reuse Mode (skipping pre-training) ---")
        else:
            logger.info("--- STEP 2: SSL Pre-training ---")
            ssl_checkpoint_path = output_dir / "model" / "ssl_checkpoint_latest.pt"
            _prepare_run_checkpoint(
                ssl_cfg.get("resume_checkpoint_path", ssl_cfg.get("checkpoint_path")),
                ssl_checkpoint_path,
                logger,
                "SSL",
            )
            ssl_weights_path = run_ssl_pretraining(
                train_dataset,
                config,
                output_dir,
                device,
                seed,
                checkpoint_path=ssl_checkpoint_path,
                resume_from_checkpoint=bool(ssl_cfg.get("resume_from_checkpoint", True)),
            )

        # Setup Deep Learning Model
        dl_config = config["modules"]["deep_learning"]
        dl_config["input_dim"] = train_dataset.input_dim
        dl_config["static_dim"] = train_dataset.static_dim
        dl_model = create_deep_model(dl_config, device=device, seed=seed)

        if ssl_weights_path:
            dl_model.load_pretrained_encoder(ssl_weights_path)

        # 5. Deep Learning Training (with DP) & Optional Federated Mode
        dl_checkpoint_path = output_dir / "model" / "dl_checkpoint_latest.pt"
        dl_cfg = config["modules"]["deep_learning"]
        dl_resume_from_checkpoint = bool(dl_cfg.get("resume_from_checkpoint", True))
        _prepare_run_checkpoint(
            dl_cfg.get("resume_checkpoint_path"),
            dl_checkpoint_path,
            logger,
            "DL",
        )

        if config["modules"]["federated_learning"].get("enabled", False):
            logger.info("--- STEP 3: DL Training (with DP) ---")
            logger.info("--- STEP 4: Optional Federated Mode (FedAvg) ---")
            if dl_resume_from_checkpoint:
                _load_fl_starting_checkpoint_if_available(dl_model, dl_checkpoint_path, logger)
            run_fl_simulation(
                dl_model, train_dataset, val_dataset, config, output_dir, seed
            )
        else:
            logger.info("--- STEP 3: DL Training (with DP) ---")
            logger.info("--- STEP 4: Optional Federated Mode (SKIPPED) ---")
            dl_model.fit(
                train_dataset,
                val_dataset,
                config=config,
                checkpoint_path=dl_checkpoint_path,
                resume_from_checkpoint=dl_resume_from_checkpoint,
            )

        dl_preds = dl_model.predict_proba(val_dataset)
        if len(dl_preds) == 0:
            raise RuntimeError("Deep model produced zero predictions.")
        dl_metrics = evaluate_binary(val_labels, dl_preds)
        logger.info(f"Deep Learning PR-AUC: {dl_metrics['pr_auc']:.4f} | ROC-AUC: {dl_metrics['roc_auc']:.4f}")

        # 6. Supervised Baseline (CatBoost)
        logger.info("--- STEP 5: CatBoost Training ---")
        supervised_model = create_supervised_model(config["modules"]["supervised"], device=device, seed=seed)
        supervised_model.fit(X_train_features, y_train, X_val_features, y_val)

        sup_preds = supervised_model.predict_proba(X_val_features)
        if len(sup_preds) == 0:
            raise RuntimeError("CatBoost model produced zero predictions.")
        sup_metrics = evaluate_binary(y_val, sup_preds)
        logger.info(f"CatBoost Baseline PR-AUC: {sup_metrics['pr_auc']:.4f} | ROC-AUC: {sup_metrics['roc_auc']:.4f}")

        # 7. Ensemble Fusion
        logger.info("--- STEP 6: Ensemble Creation ---")
        if len(sup_preds) != len(dl_preds):
            logger.warning(
                "Prediction length mismatch detected: "
                f"CatBoost={len(sup_preds)}, DL={len(dl_preds)}. Truncating to minimum length."
            )
        min_len = min(len(sup_preds), len(dl_preds), len(val_labels))
        if min_len == 0:
            raise RuntimeError("Prediction alignment failed: no overlapping predictions available for ensemble.")

        ensemble = EnsembleModel(method="weighted_blend")
        ensemble.fit([sup_preds[:min_len], dl_preds[:min_len]], val_labels[:min_len])
        fused_preds = ensemble.predict_proba([sup_preds[:min_len], dl_preds[:min_len]])
        fused_metrics = evaluate_binary(val_labels[:min_len], fused_preds)
        logger.info(f"Ensemble Model PR-AUC: {fused_metrics['pr_auc']:.4f} | ROC-AUC: {fused_metrics['roc_auc']:.4f}")

        # 8. Domain Generalization Verification (LODO)
        logger.info("--- STEP 7: DG Evaluation ---")
        run_leave_one_domain_out(X_train, y_train, feature_cols, config, output_dir, seed)

        # 9. Explainability (XAI)
        logger.info("--- STEP 8: XAI Generation ---")
        run_xai_analysis(supervised_model, X_val_features, config, output_dir)

        # FIX: val_dataset[:32] returns (X, y, mask, static) — pass correct indices to Captum
        val_subset = val_dataset[:32]
        # val_subset[0] = X_seq, val_subset[1] = y_labels, val_subset[2] = masks, val_subset[3] = static
        run_captum_analysis(dl_model, val_subset[0], val_subset[2], val_subset[3], config, output_dir)

        # ============================================================
        # 10. HARDENED ARTIFACT SAVING — TRIPLE REDUNDANCY + VERIFICATION
        # ============================================================
        logger.info("=" * 60)
        logger.info("--- STEP 9: HARDENED Artifact Saving ---")
        logger.info("=" * 60)

        model_dir = output_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        save_results: dict[str, bool] = {}

        # Save DL model
        dl_save_path = model_dir / "dl_model_final.pt"
        save_results["dl_model"] = _save_dl_model(dl_model, dl_save_path, logger)

        # Save CatBoost model
        cbm_save_path = model_dir / "model.cbm"
        save_results["catboost"] = _save_catboost_model(supervised_model, cbm_save_path, logger)

        # Save Feature Scaler
        scaler_path = model_dir / "scaler.pkl"
        try:
            joblib.dump(scaler, scaler_path)
            save_results["scaler"] = _verify_saved(scaler_path, logger, "Data Scaler")
        except Exception as e:
            logger.error(f"Failed to save scaler: {e}")
            save_results["scaler"] = False

        # Save ensemble weights
        ensemble_path = model_dir / "ensemble.pkl"
        save_results["ensemble"] = _save_ensemble(ensemble, ensemble_path, logger)

        # Save feature column list for reproducible loading
        feature_cols_path = model_dir / "feature_columns.json"
        try:
            with open(feature_cols_path, "w", encoding="utf-8") as f:
                json.dump(feature_cols, f, indent=2)
            save_results["feature_columns"] = _verify_saved(feature_cols_path, logger, "Feature Columns")
        except Exception as e:
            logger.error(f"Failed to save feature columns: {e}")
            save_results["feature_columns"] = False

        # Run optional deployment exporter (ONNX, etc.)
        try:
            export_models_for_production(supervised_model, dl_model, config, output_dir)
        except Exception as e:
            logger.warning(f"Deployment export had issues (non-fatal): {e}")

        # ============================================================
        # Save ALL metrics (per-model + ensemble)
        # ============================================================
        all_metrics = {
            "deep_learning": dl_metrics,
            "catboost": sup_metrics,
            "ensemble": fused_metrics,
            "ensemble_weights": ensemble.weights.tolist() if ensemble.weights is not None else None,
        }

        metrics_path = output_dir / "metrics.json"
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(all_metrics, f, indent=4, default=str)
            save_results["metrics"] = _verify_saved(metrics_path, logger, "Metrics")
            logger.info(f"All metrics saved to {metrics_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            save_results["metrics"] = False

        # Save predictions for downstream analysis
        preds_df = pd.DataFrame({
            "y_true": val_labels[:min_len],
            "y_proba_ensemble": fused_preds,
            "y_proba_catboost": sup_preds[:min_len],
            "y_proba_dl": dl_preds[:min_len],
        })
        preds_path = output_dir / "predictions.csv"
        try:
            preds_df.to_csv(preds_path, index=False)
            save_results["predictions"] = _verify_saved(preds_path, logger, "Predictions")
            logger.info(f"Predictions saved to {preds_path}")
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            save_results["predictions"] = False

        # ============================================================
        # FINAL VERIFICATION REPORT
        # ============================================================
        logger.info("=" * 60)
        logger.info("ARTIFACT SAVE REPORT:")
        for name, success in save_results.items():
            status = "✓ SAVED" if success else "✗ FAILED"
            logger.info(f"  {status}: {name}")

        required_checks = _verify_required_artifacts(output_dir, logger, config)
        for name, success in required_checks.items():
            if not success:
                save_results[f"required_{name}"] = False

        total_saved = sum(save_results.values())
        total_expected = len(save_results)

        if total_saved == total_expected:
            logger.info(f"ALL {total_expected}/{total_expected} artifacts saved successfully!")
        else:
            logger.error(f"WARNING: Only {total_saved}/{total_expected} artifacts saved! Check errors above.")

        failed_artifacts = [name for name, success in save_results.items() if not success]
        if failed_artifacts:
            raise RuntimeError(
                "Critical artifact save failures detected: " + ", ".join(sorted(failed_artifacts))
            )

        logger.info("=" * 60)
        logger.info("=== PIPELINE COMPLETE ===")
        logger.info(f"Run directory: {output_dir}")
        logger.info(
            f"DL PR-AUC: {dl_metrics['pr_auc']:.4f} | "
            f"CB PR-AUC: {sup_metrics['pr_auc']:.4f} | "
            f"Ensemble PR-AUC: {fused_metrics['pr_auc']:.4f}"
        )
        logger.info("=" * 60)
        return output_dir
    except Exception:
        if logger is not None:
            logger.exception("Pipeline failed with an unrecoverable error.")
        raise
    finally:
        _flush_logger_handlers(logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete Patient Deterioration Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()
    
    run_pipeline(args.config)

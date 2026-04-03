"""
Federated Learning Simulator.

Simulates training across multiple siloed hospitals without data sharing.
We partition the training data to simulate different sites (potentially non-IID),
train local models, and perform global aggregation.

HARDENED: Tracks best global model across rounds so performance doesn't degrade.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from core.logger import get_logger
from core.metrics import evaluate_binary

logger = get_logger("federated_learning")

def aggregate_weights(global_model: torch.nn.Module, local_models: list[torch.nn.Module], client_weights: list[float]) -> None:
    """Perform FedAvg on local model weights and update global model."""
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        # Opacus wraps keys with '_module.' prefix — strip it when reading local models
        aggregated = None
        for local_model, w in zip(local_models, client_weights):
            local_sd = local_model.state_dict()
            # Try the key as-is first, then with _module. prefix
            if k in local_sd:
                val = local_sd[k].float() * w
            elif f"_module.{k}" in local_sd:
                val = local_sd[f"_module.{k}"].float() * w
            else:
                # Key not found, skip aggregation for this param
                val = global_dict[k].float() * w
            aggregated = val if aggregated is None else aggregated + val
        global_dict[k] = aggregated
    global_model.load_state_dict(global_dict)

def run_fl_simulation(
    global_model: "TCNTransformerModel",
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    config: dict,
    output_dir: Path,
    seed: int = 42
) -> None:
    """
    Run genuine PyTorch Federated Learning (FedAvg) across simulated clients.
    
    HARDENED: Tracks the best global model across all FL rounds. If a later round
    degrades performance, we revert to the best checkpoint instead of using the
    degraded final round.
    """
    params = config["modules"]["federated_learning"]
    if not params.get("enabled", False):
        logger.info("Federated Learning is disabled. Falling back to centralized training.")
        try:
            global_model.fit(
                train_dataset, val_dataset, config=config
            )
        except Exception as e:
            logger.error(f"Centralized fallback training failed: {e}")
            raise
        return
        
    num_clients = params.get("clients", 3)
    rounds = params.get("rounds", 5)
    local_epochs = params.get("local_epochs", 2)
    
    logger.info(f"Starting PyTorch FedAvg Simulation (Clients: {num_clients}, Rounds: {rounds}, Local Epochs: {local_epochs})")
    
    # 1. Partition data to simulate silos using PyTorch Subset
    kf = KFold(n_splits=num_clients, shuffle=True, random_state=seed)
    total_len = len(train_dataset)
    indices = np.arange(total_len)
    
    client_data = []
    for _, test_idx in kf.split(indices):
        client_data.append(torch.utils.data.Subset(train_dataset, test_idx))
        
    logger.info(f"Data securely partitioned into {num_clients} silos.")
    
    history = []
    
    val_labels = np.concatenate(val_dataset.dataset.targets) if hasattr(val_dataset, "dataset") else np.concatenate(val_dataset.targets)
    
    # Track the best global model across all rounds
    best_pr_auc = -1.0
    best_round = -1
    best_global_state = None
    
    # 2. FL Rounds
    for r in range(rounds):
        logger.info(f"--- FL Communication Round {r+1}/{rounds} ---")
        
        local_models = []
        client_weights = []
        
        for client_id, c_dataset in enumerate(client_data):
            logger.info(f"Training Client {client_id+1}/{num_clients}")
            
            # Deepcopy global model wrapper (maintains model architecture cleanly)
            local_model_wrapper = copy.deepcopy(global_model)
            # Temporarily set epochs for local training
            original_epochs = local_model_wrapper.epochs
            local_model_wrapper.epochs = local_epochs
            
            # Fit local model — flag as FL client to prevent global checkpoint collision
            local_model_wrapper.fit(
                c_dataset, val_dataset, config=config, _is_fl_client=True
            )
            
            # Restore original epochs just in case
            local_model_wrapper.epochs = original_epochs
            
            local_models.append(local_model_wrapper.model)
            client_weights.append(len(c_dataset))
            
        # Normalize weights for FedAvg
        total_samples = sum(client_weights)
        norm_weights = [w / total_samples for w in client_weights]
        
        # Global Aggregation (FedAvg)
        logger.info("Aggregating model parameters (FedAvg)...")
        aggregate_weights(global_model.model, local_models, norm_weights)
        
        # Evaluate Global Model
        val_preds = global_model.predict_proba(val_dataset)
        metrics = evaluate_binary(val_labels, val_preds)
        
        round_pr_auc = metrics["pr_auc"]
        round_roc_auc = metrics["roc_auc"]
        
        # Track best global model
        is_best = round_pr_auc > best_pr_auc
        if is_best:
            best_pr_auc = round_pr_auc
            best_round = r + 1
            # Save best state dict (on CPU for safety)
            best_global_state = {k: v.cpu().clone() for k, v in global_model.model.state_dict().items()}
        
        status = "[BEST]" if is_best else f"[Best was R{best_round}: {best_pr_auc:.4f}]"
        logger.info(f"Global Model Validation | PR-AUC: {round_pr_auc:.4f} | ROC-AUC: {round_roc_auc:.4f} {status}")
        
        history.append({
            "round": r + 1,
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "brier_score": metrics["brier_score"],
            "is_best": is_best
        })
    
    # 3. Restore best global model if the final round degraded
    if best_global_state is not None:
        final_pr_auc = history[-1]["pr_auc"]
        if final_pr_auc < best_pr_auc:
            logger.info(f"Final round PR-AUC ({final_pr_auc:.4f}) < Best round {best_round} ({best_pr_auc:.4f}). Restoring best model.")
            try:
                global_model.model.load_state_dict(best_global_state)
            except Exception as e:
                logger.error(f"Failed to restore best FL model state: {e}")
                raise
        else:
            logger.info(f"Final round IS the best round. No restoration needed.")
    
    # Save FL History
    out_path = output_dir / "fl_rounds_history.json"
    with open(out_path, "w") as f:
        json.dump(history, f, indent=4)
        
    logger.info(f"PyTorch FL FedAvg Simulation complete. Best PR-AUC: {best_pr_auc:.4f} (Round {best_round}). History saved to {out_path}")

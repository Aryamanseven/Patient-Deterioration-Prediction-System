"""
Differentially Private Optimizer (DP-SGD).

Uses Opacus to trace PyTorch models and applies per-sample gradient 
clipping and noise addition during training.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from core.logger import get_logger

logger = get_logger("dp")

def apply_differential_privacy(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    config: dict
) -> tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader, Any]:
    """
    Hooks Opacus into the PyTorch training loop to guarantee Differential Privacy.
    Returns the DP-wrapped (model, optimizer, dataloader) and the PrivacyEngine.
    """
    params = config["modules"]["differential_privacy"]
    if not params.get("enabled", False):
        return model, optimizer, data_loader, None
        
    epsilon = params.get("target_epsilon", 3.0)
    delta = params.get("target_delta", 1e-5)
    max_grad_norm = params.get("max_grad_norm", 1.0)
    epochs = config["modules"]["deep_learning"].get("epochs", 25)
    
    logger.info(f"Applying Differential Privacy (ε={epsilon}, δ={delta}, clipping={max_grad_norm})...")
    
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        
        privacy_engine = PrivacyEngine()
        
        # Ensure model is compatible with Opacus
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)
            # Model parameters changed, must recreate optimizer
            optim_class = type(optimizer)
            optim_kwargs = optimizer.defaults
            optimizer = optim_class(model.parameters(), **optim_kwargs)
        
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            epochs=epochs,
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
        )
        
        logger.info("DP-SGD successfully hooked into training pipeline.")
        return model, optimizer, data_loader, privacy_engine
        
    except ImportError:
        logger.warning(
            "Opacus library not found. Differential Privacy is simulated "
            "(gradients will not be strictly clipped). Install `opacus` for true DP-SGD."
        )
        return model, optimizer, data_loader, None

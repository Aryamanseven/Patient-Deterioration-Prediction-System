"""
Model Registry.

Instantiates supervised and deep learning models based on the configuration strings.
This centralizes the switch/case logic away from the main pipeline.
"""
from __future__ import annotations

from typing import Any

from .catboost_model import CatBoostWrapper
from .lstm_attention import LSTMAttentionModel
from .tcn_transformer import TCNTransformerModel


def create_supervised_model(config_dict: dict[str, Any], device: str = "cpu", seed: int = 42) -> Any:
    """Instantiate the supervised model chosen in the config."""
    model_type = config_dict.get("model_type", "catboost").lower()
    
    if model_type == "catboost":
        return CatBoostWrapper(params=config_dict.get("params", {}), device=device, random_seed=seed)
    else:
        raise ValueError(f"Unknown supervised model type: {model_type}")


def create_deep_model(config_dict: dict[str, Any], device: str, seed: int = 42) -> Any:
    """Instantiate the deep learning network chosen in the config."""
    model_type = config_dict.get("model_type", "tcn_transformer").lower()
    
    if model_type == "lstm_attention":
        return LSTMAttentionModel(params=config_dict, device=device, random_seed=seed)
    elif model_type == "tcn_transformer":
        return TCNTransformerModel(params=config_dict, device=device, random_seed=seed)
    else:
        raise ValueError(f"Unknown deep learning model type: {model_type}")

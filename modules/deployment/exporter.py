"""
Model Exporter.

Packages and serializes models in highly optimized formats (e.g. ONNX, CatBoost Binary)
for fast, secure production deployment in hospital settings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from core.logger import get_logger

logger = get_logger("deployment")


def export_models_for_production(
    supervised_model: Any | None,
    dl_model: Any | None,
    config: dict,
    output_dir: Path,
) -> None:
    """
    Export all trained models to standardized deployment formats.
    """
    params = config.get("modules", {}).get("deployment", {})
    enabled = params.get("enabled", bool(params) or ("onnx_export" in params))
    if not enabled:
        logger.info("Deployment exporter is disabled. Skipping.")
        return

    export_catboost = params.get("export_catboost", True)
    export_onnx = params.get("export_onnx", params.get("onnx_export", True))
        
    logger.info("Starting Medical Device Deployment Export...")
    
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. CatBoost Export
    if export_catboost and supervised_model is not None:
        try:
            cbm_path = model_dir / "model.cbm"
            supervised_model.model.save_model(str(cbm_path))
            logger.info(f"Deployed supervised model to {cbm_path}")
        except Exception as e:
            logger.error(f"Failed to export CatBoost: {e}")
            
    # 2. PyTorch -> ONNX Export
    if export_onnx and dl_model is not None and dl_model.model is not None:
        try:
            onnx_path = model_dir / "dl_model.onnx"
            
            # Get dimensions from the actual model and config (not broken config["core"])
            seq_len = config.get("modules", {}).get("deep_learning", {}).get("max_seq_len", 24)
            input_dim = dl_model.model.input_proj.in_features
            static_dim = dl_model.model.static_mlp[0].in_features
            
            dummy_seq = torch.randn(1, seq_len, input_dim).to(dl_model.device)
            dummy_mask = torch.ones(1, seq_len, dtype=torch.bool).to(dl_model.device)
            dummy_static = torch.randn(1, static_dim).to(dl_model.device)
            
            dl_model.model.eval()
            
            torch.onnx.export(
                dl_model.model,
                (dummy_seq, dummy_mask, dummy_static),
                str(onnx_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["sequence", "mask", "static"],
                output_names=["logits"],
                dynamic_axes={
                    "sequence": {0: "batch_size"},
                    "mask": {0: "batch_size"},
                    "static": {0: "batch_size"},
                    "logits": {0: "batch_size"},
                }
            )
            logger.info(f"Deployed Deep Learning model to {onnx_path}")
        except ImportError:
            logger.warning("ONNX missing. Cannot export Pytorch model to ONNX. `pip install onnx`.")
        except Exception as e:
            logger.warning(f"Failed to export ONNX (expected if model uses advanced un-traceable ops): {e}")

    logger.info("Deployment artifacts ready.")

"""
Model Exporter.

Packages and serializes models in highly optimized formats (e.g. ONNX, CatBoost Binary)
for fast, secure production deployment in hospital settings.
"""
from __future__ import annotations

from pathlib import Path

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
    params = config["modules"]["deployment"]
    if not params.get("enabled", False):
        logger.info("Deployment exporter is disabled. Skipping.")
        return
        
    logger.info("Starting Medical Device Deployment Export...")
    
    # 1. CatBoost Export
    if params.get("export_catboost", True) and supervised_model is not None:
        try:
            cbm_path = output_dir / "model" / "model.cbm"
            supervised_model.model.save_model(str(cbm_path))
            logger.info(f"Deployed supervised model to {cbm_path}")
        except Exception as e:
            logger.error(f"Failed to export CatBoost: {e}")
            
    # 2. PyTorch -> ONNX Export
    if params.get("export_onnx", True) and dl_model is not None:
        try:
            onnx_path = output_dir / "model" / "dl_model.onnx"
            # We need dummy inputs for Torch tracing
            dummy_seq = torch.randn(1, config["core"]["sequence_length"], dl_model.input_dim).to(dl_model.device)
            dummy_static = torch.randn(1, dl_model.static_dim).to(dl_model.device)
            dummy_mask = torch.ones(1, config["core"]["sequence_length"], dtype=torch.bool).to(dl_model.device)
            
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

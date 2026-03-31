"""
Self-Supervised Pre-Training (Masked Sequence Prediction).

Forces the model to predict missing temporal segments of vital signs,
teaching it intrinsic physiological dynamics before supervised learning.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.logger import get_logger
from models.tcn_transformer import TCNTransformerNetwork

logger = get_logger("ssl")


class MaskedPredictionHead(nn.Module):
    """Linear layer to project transformer hidden dim back to feature dim."""
    def __init__(self, d_model: int, out_features: int):
        super().__init__()
        self.proj = nn.Linear(d_model, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def run_ssl_pretraining(
    dataset: torch.utils.data.Dataset,
    config: dict,
    output_dir: Path,
    device: str,
    seed: int = 42
) -> str:
    """
    Run masked prediction SSL on unlabelled data.
    
    Returns the path to the saved weights.
    """
    torch.manual_seed(seed)
    device_obj = torch.device(device)
    
    params = config["modules"]["ssl"]
    if not params.get("enabled", False):
        logger.info("SSL is disabled. Skipping.")
        return ""
        
    mask_ratio = params.get("mask_ratio", 0.15)
    epochs = params.get("pretrain_epochs", 10)
    batch_size = params.get("batch_size", 512)
    lr = params.get("learning_rate", 1e-3)
    
    input_dim = dataset.input_dim
    static_dim = getattr(dataset, "static_dim", 2)
    
    logger.info(f"Starting SSL Pre-training (Mask Ratio: {mask_ratio}, Epochs: {epochs})...")
    
    # Use exact same network architecture as the downstream task
    base_model = TCNTransformerNetwork(
        input_dim=input_dim,
        static_dim=static_dim,
        d_model=params.get("d_model", 64),
        num_heads=params.get("num_heads", 4),
        num_layers=params.get("num_layers", 2),
        dropout=0.1
    ).to(device_obj)
    
    # An SSL head to reconstruct the raw features from the Transformer output
    # The transformer output is (batch, seq, d_model)
    head = MaskedPredictionHead(params.get("d_model", 64), input_dim).to(device_obj)
    
    optimizer = torch.optim.AdamW(
        list(base_model.parameters()) + list(head.parameters()),
        lr=lr
    )
    criterion = nn.MSELoss()
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    global_ckpt_path = Path("artifacts") / "ssl_checkpoint_latest.pt"
    start_epoch = 0
    
    # Auto-resume logic
    if global_ckpt_path.exists():
        logger.info(f"Auto-resuming SSL from {global_ckpt_path}")
        checkpoint = torch.load(global_ckpt_path, map_location=device_obj, weights_only=False)
        base_model.load_state_dict(checkpoint["model_state"])
        head.load_state_dict(checkpoint["head_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resuming at Epoch {start_epoch+1}")
        
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        for batch_idx, (batch_x, _, batch_mask, batch_static) in enumerate(loader):
            if batch_idx % 10 == 0:
                logger.info(f"SSL Epoch {epoch+1:02d} | Batch {batch_idx}/{len(loader)} | Active")
                
            batch_x = batch_x.to(device_obj)
            batch_mask = batch_mask.to(device_obj)
            
            # Generate mask on CPU first, then transfer (DML compatibility)
            rand_mask = torch.rand(batch_x.shape[:2], device="cpu") < mask_ratio
            rand_mask = rand_mask.to(device_obj)
            actual_mask = rand_mask & batch_mask
            
            corrupted_x = batch_x.clone()
            corrupted_x[actual_mask] = 0.0
            
            optimizer.zero_grad()
            
            projected = base_model.input_proj(corrupted_x)
            x_conv = projected.transpose(1, 2)
            s3 = base_model.scale_3h(x_conv).transpose(1, 2)
            s6 = base_model.scale_6h(x_conv).transpose(1, 2)
            fused = base_model.scale_fusion(torch.cat([s3, s6], dim=-1))
            encoded = base_model.transformer(fused, src_key_padding_mask=~batch_mask)
            
            reconstructed = head(encoded)
            loss = criterion(reconstructed[actual_mask], batch_x[actual_mask])
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= len(dataset)
        logger.info(f"SSL Epoch {epoch+1:02d}/{epochs} completed | Final MSE: {epoch_loss:.4f}")
        
        # Save checkpoint globally
        torch.save({
            "epoch": epoch,
            "model_state": base_model.state_dict(),
            "head_state": head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": epoch_loss
        }, global_ckpt_path)
        
    # Final save to the specific run directory as a finalized asset
    out_path = output_dir / "ssl_pretrained_tcntransformer.pt"
    torch.save(base_model.state_dict(), out_path)
    logger.info(f"SSL pre-training complete. Final Weights saved to {out_path}")
    
    return str(out_path)

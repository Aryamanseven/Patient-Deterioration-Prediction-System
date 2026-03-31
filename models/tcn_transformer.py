"""
Multi-Scale Temporal Convolutional Network (TCN) + Transformer.

This is the championship architecture. It captures local temporal patterns at
different scales (e.g. 3h, 6h, 12h windows) using dilated convolutions,
and then fuses them using self-attention across the sequence.

Also supports initializing from Self-Supervised Learning (SSL) weights.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.logger import get_logger

logger = get_logger("tcn_transformer")


class MultiScaleTemporalBlock(nn.Module):
    """TCN block at a specific scale (kernel size)."""
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp = padding
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        if self.chomp > 0:
            out = out[:, :, :-self.chomp]
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        if self.chomp > 0:
            out = out[:, :, :-self.chomp]
        out = self.norm2(out)
        out = self.activation(out)
        return out + x  # Residual connection


class TCNTransformerNetwork(nn.Module):
    """
    Novel architecture: Multi-scale TCN captures patterns at different
    temporal resolutions, then a Transformer encoder fuses them.
    """
    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Multi-scale TCN branches
        self.scale_3h = nn.Sequential(
            MultiScaleTemporalBlock(d_model, kernel_size=3, dilation=1, dropout=dropout),
            MultiScaleTemporalBlock(d_model, kernel_size=3, dilation=2, dropout=dropout),
        )
        self.scale_6h = nn.Sequential(
            MultiScaleTemporalBlock(d_model, kernel_size=5, dilation=1, dropout=dropout),
            MultiScaleTemporalBlock(d_model, kernel_size=5, dilation=2, dropout=dropout),
        )
        
        # Cross-scale fusion
        self.scale_fusion = nn.Linear(d_model * 2, d_model)
        
        # Transformer for global temporal attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Variable length pooling logic
        self.pool_score = nn.Linear(d_model, 1)
        
        # Static enrichment
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
        )
        
        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        static: torch.Tensor
    ) -> torch.Tensor:
        # x: (batch, seq, in_features)
        # Ensure float32 — DML augmentation can leave x as bool after masking
        projected = self.input_proj(x.float())
        
        # Prep for Conv1d: (batch, channels, seq)
        x_conv = projected.transpose(1, 2)
        
        s3 = self.scale_3h(x_conv).transpose(1, 2)
        s6 = self.scale_6h(x_conv).transpose(1, 2)
        
        fused = self.scale_fusion(torch.cat([s3, s6], dim=-1))
        
        # Transformer (expects True for ignored padding tokens in src_key_padding_mask)
        # Our mask is True for valid, False for padding. So we invert it.
        encoded = self.transformer(fused, src_key_padding_mask=~mask)
        
        # Attention pooling instead of simple mean
        scores = self.pool_score(encoded).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), encoded).squeeze(1)
        
        # Static features
        static_feat = self.static_mlp(static)
        
        # Combine and classify
        combined = torch.cat([pooled, static_feat], dim=-1)
        logits = self.classifier(combined).squeeze(-1)
        return logits


class TCNTransformerModel:
    """Wrapper that manages PyTorch training loop for TCN-Transformer."""
    
    def __init__(
        self,
        params: dict[str, Any],
        device: str = "cpu",
        random_seed: int = 42,
    ):
        self.params = params
        self.device = torch.device(device)
        self.random_seed = random_seed
        self.model: TCNTransformerNetwork | None = None
        
        # Hyperparams
        self.epochs = params.get("epochs", 25)
        self.batch_size = params.get("batch_size", 256)
        self.lr = params.get("learning_rate", 1e-3)
        self.weight_decay = params.get("weight_decay", 1e-4)
        self.patience = params.get("patience", 7)
        
        if "input_dim" in params and "static_dim" in params:
            self._build_model(params["input_dim"], params["static_dim"])

    def _build_model(self, input_dim: int, static_dim: int) -> None:
        if self.model is None:
            self.model = TCNTransformerNetwork(
                input_dim=input_dim,
                static_dim=static_dim,
                d_model=self.params.get("d_model", 64),
                num_heads=self.params.get("num_heads", 4),
                num_layers=self.params.get("num_layers", 2),
                dropout=self.params.get("dropout", 0.15)
            ).to(self.device)

    def load_pretrained_encoder(self, ssl_weights_path: str) -> None:
        """Load SSL weights into the encoder."""
        if self.model is None:
            raise ValueError("Model must be built before loading weights. Provide input_dim and static_dim in params.")
        try:
            state_dict = torch.load(ssl_weights_path, map_location=self.device, weights_only=False)
            filtered_dict = {k: v for k, v in state_dict.items() if k in self.model.state_dict() and "classifier" not in k}
            self.model.load_state_dict(filtered_dict, strict=False)
            logger.info(f"Successfully initialized encoder from SSL weights: {ssl_weights_path}")
        except Exception as e:
            logger.warning(f"Failed to load SSL weights: {e}")

    def _prepare_data(
        self,
        dataset: torch.utils.data.Dataset,
        shuffle: bool = False
    ) -> DataLoader:
        from torch.utils.data import DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def fit(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        ssl_weights_path: str | None = None,
        config: dict = None,
        _is_fl_client: bool = False,
        **kwargs
    ) -> "TCNTransformerModel":
        from torch.utils.data import DataLoader, Subset
        torch.manual_seed(self.random_seed)
        
        # Handle both raw datasets and Subset objects (e.g., from FL split)
        base_ds = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
        input_dim = getattr(base_ds, "input_dim", getattr(base_ds, "dataset", base_ds).input_dim if hasattr(base_ds, "dataset") else None)
        static_dim = getattr(base_ds, "static_dim", getattr(base_ds, "dataset", base_ds).static_dim if hasattr(base_ds, "dataset") else 2)
        
        self._build_model(input_dim, static_dim)
        
        if ssl_weights_path and self.params.get("use_ssl_weights", True):
            try:
                state_dict = torch.load(ssl_weights_path, map_location=self.device, weights_only=False)
                # Load only matching encoder parameters
                filtered_dict = {k: v for k, v in state_dict.items() if k in self.model.state_dict() and "classifier" not in k}
                self.model.load_state_dict(filtered_dict, strict=False)
                logger.info(f"Successfully initialized encoder from SSL weights: {ssl_weights_path}")
            except Exception as e:
                logger.warning(f"Failed to load SSL weights, continuing from scratch: {e}")
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        import numpy as np
        y_arr = np.concatenate(train_dataset.dataset.targets) if isinstance(train_dataset, Subset) else np.concatenate(train_dataset.targets)
        if isinstance(train_dataset, Subset):
            y_arr = y_arr[train_dataset.indices]
            
        pw = (len(y_arr) - sum(y_arr)) / max(sum(y_arr), 1)
        pw = min(pw, 20.0)  # Clamp to prevent extreme imbalance blowing up loss
        pos_weight = torch.tensor([pw], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        train_loader = self._prepare_data(train_dataset, True)
        val_loader = self._prepare_data(val_dataset, False)
        
        privacy_engine = None
        is_dml = "privateuseone" in str(self.device)
        if config and config.get("modules", {}).get("differential_privacy", {}).get("enabled", False):
            if is_dml:
                logger.warning("Differential Privacy (Opacus) is incompatible with DirectML — skipping DP-SGD. "
                               "Train on CUDA or CPU for true DP guarantees.")
            else:
                from modules.differential_privacy.optimizer import apply_differential_privacy
                self.model, optimizer, train_loader, privacy_engine = apply_differential_privacy(
                    self.model, optimizer, train_loader, config
                )
            
        logger.info(f"Training TCN-Transformer with {input_dim} seq features...")
        
        augmenter = None
        if config and config.get("modules", {}).get("domain_generalization", {}).get("enabled", False):
            try:
                from modules.domain_generalization.augmentation import SequenceAugmenter
                augmenter = SequenceAugmenter(jitter_std=0.05, dropout_prob=0.1)
                logger.info("Sequence-level Domain Generalization Augmentation Enabled.")
            except ImportError:
                pass
        
        import os
        from pathlib import Path
        global_dl_ckpt = Path("artifacts") / "dl_checkpoint_latest.pt"
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        start_epoch = 0
        
        # Auto-resume logic — ONLY for the main pipeline run, NOT for FL clients
        if not _is_fl_client and global_dl_ckpt.exists():
            logger.info(f"Auto-resuming DL from {global_dl_ckpt}")
            try:
                checkpoint = torch.load(global_dl_ckpt, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                start_epoch = checkpoint["epoch"] + 1
                best_val_loss = checkpoint.get("best_val_loss", float('inf'))
                patience_counter = checkpoint.get("patience_counter", 0)
                best_state = checkpoint.get("best_state", None)
                if best_state is None:
                    best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                logger.info(f"Resuming at DL Epoch {start_epoch+1}")
            except Exception as e:
                logger.warning(f"Failed to load DL checkpoint ({e}). Starting fresh.")
        
        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y, batch_mask, batch_static in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                batch_mask, batch_static = batch_mask.to(self.device), batch_static.to(self.device)
                
                if augmenter is not None:
                    batch_x, batch_mask = augmenter(batch_x, batch_mask)
                
                optimizer.zero_grad()
                logits = self.model(batch_x, batch_mask, batch_static)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * batch_x.size(0)
                if getattr(self, "printed_batch", None) is None:
                    self.printed_batch = 0
                self.printed_batch += 1
                if self.printed_batch % 50 == 0:
                    logger.info(f"DL Epoch {epoch+1:02d} | Running Training Loss: {loss.item():.4f}")
            train_loss /= len(train_loader.dataset)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y, batch_mask, batch_static in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    batch_mask, batch_static = batch_mask.to(self.device), batch_static.to(self.device)
                    
                    logits = self.model(batch_x, batch_mask, batch_static)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
            val_loss /= len(val_loader.dataset)
            
            # Extract clean state dict (unwrap Opacus _module prefix if present)
            raw_state = self.model.state_dict()
            clean_state = {k.replace("_module.", ""): v for k, v in raw_state.items()}
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu() for k, v in clean_state.items()}
                
                dp_msg = ""
                if privacy_engine:
                    eps = privacy_engine.get_epsilon(config["modules"]["differential_privacy"].get("target_delta", 1e-5))
                    dp_msg = f" | DP(ε={eps:.2f})"
                    
                logger.info(f"Epoch {epoch+1:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} [Best]{dp_msg}")
            else:
                patience_counter += 1
                dp_msg = ""
                if privacy_engine:
                    eps = privacy_engine.get_epsilon(config["modules"]["differential_privacy"].get("target_delta", 1e-5))
                    dp_msg = f" | DP(ε={eps:.2f})"
                    
                logger.info(f"Epoch {epoch+1:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} [Patience {patience_counter}/{self.patience}]{dp_msg}")
                if patience_counter >= self.patience:
                    logger.info("Early stopping triggered by patience.")
                    break
        
            # Save global checkpoint every epoch — only for main pipeline, not FL clients
            if not _is_fl_client:
                try:
                    torch.save({
                        "epoch": epoch,
                        "model_state": clean_state,
                        "optimizer_state": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "patience_counter": patience_counter,
                        "best_state": best_state
                    }, global_dl_ckpt)
                except Exception as e:
                    logger.warning(f"Failed to save DL checkpoint at epoch {epoch+1}: {e}")
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
            
        logger.info(f"Training completed. Best Val Loss: {best_val_loss:.4f}")
        return self

    def predict_proba(
        self,
        dataset: torch.utils.data.Dataset
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        loader = self._prepare_data(dataset, False)
        self.model.eval()
        
        all_probs = []
        with torch.no_grad():
            for batch_x, _, batch_mask, batch_static in loader:
                batch_x = batch_x.to(self.device)
                batch_mask = batch_mask.to(self.device)
                batch_static = batch_static.to(self.device)
                
                logits = self.model(batch_x, batch_mask, batch_static)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                
        result = np.concatenate(all_probs)
        # Clamp NaN/Inf to safe values for downstream metrics
        result = np.nan_to_num(result, nan=0.5, posinf=1.0, neginf=0.0)
        return result
        
    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save an untrained model.")
        torch.save(self.model.state_dict(), path)
        logger.info(f"TCN-Transformer model saved to {path}")
        
    def load(self, path: str, input_dim: int, static_dim: int = 2) -> "TCNTransformerModel":
        self.model = TCNTransformerNetwork(
            input_dim=input_dim,
            static_dim=static_dim,
            d_model=self.params.get("d_model", 64),
            num_heads=self.params.get("num_heads", 4),
            num_layers=self.params.get("num_layers", 2),
            dropout=self.params.get("dropout", 0.15)
        ).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
        logger.info(f"TCN-Transformer model loaded from {path}")
        return self

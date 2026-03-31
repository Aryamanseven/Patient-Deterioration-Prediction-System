"""
LSTM with Attention Output Model.

This is the standard deep learning baseline for sequence learning.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.logger import get_logger

logger = get_logger("lstm")


class LSTMAttentionNetwork(nn.Module):
    """The PyTorch LSTM architecture."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        static_dim: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        # Bidirectional means hidden state is 2 * hidden_dim
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        # Final classifier layer takes context vector and static features (age, etc.)
        self.classifier = nn.Sequential(
            nn.Linear((hidden_dim * 2) + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        static: torch.Tensor
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, features)
        mask: (batch, seq_len) boolean
        static: (batch, static_dim)
        """
        # Pass through LSTM
        # lstm_out: (batch, seq_len, 2 * hidden_dim)
        lstm_out, _ = self.lstm(x)

        # Apply attention over time steps
        attn_weights = self.attn(lstm_out)  # (batch, seq_len, 1)
        
        # Apply mask to attention weights (set padded timesteps to -inf before softmax ideally, 
        # but here we'll just zero them out after softmax and renormalize)
        # This is a bit simpler/faster and still valid
        attn_weights = attn_weights.squeeze(-1) * mask.float()
        attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)
        attn_weights = attn_weights.unsqueeze(-1)

        # Context vector: weighted sum over time steps
        # context: (batch, 2 * hidden_dim)
        context = torch.sum(lstm_out * attn_weights, dim=1)

        # Concatenate context with static features
        combined = torch.cat([context, static], dim=1)

        # Output probability (sigmoid is applied outside or via BCEWithLogitsLoss)
        logits = self.classifier(combined).squeeze(-1)
        return logits


class LSTMAttentionModel:
    """Wrapper that manages PyTorch training loop, batching, and early stopping."""
    
    def __init__(
        self,
        params: dict[str, Any],
        device: str = "cpu",
        random_seed: int = 42,
    ):
        self.params = params
        self.device = torch.device(device)
        self.random_seed = random_seed
        self.model: LSTMAttentionNetwork | None = None
        
        # Hyperparams
        self.epochs = params.get("epochs", 20)
        self.batch_size = params.get("batch_size", 256)
        self.lr = params.get("learning_rate", 1e-3)
        self.weight_decay = params.get("weight_decay", 1e-4)
        self.patience = params.get("patience", 5)

    def _prepare_data(
        self,
        X_seq: np.ndarray,
        y: np.ndarray,
        masks: np.ndarray,
        static: np.ndarray,
        shuffle: bool = False
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(masks, dtype=torch.bool),
            torch.tensor(static, dtype=torch.float32)
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def fit(
        self,
        X_train_seq: np.ndarray,
        y_train: np.ndarray,
        train_masks: np.ndarray,
        train_static: np.ndarray,
        X_val_seq: np.ndarray,
        y_val: np.ndarray,
        val_masks: np.ndarray,
        val_static: np.ndarray,
        config: dict = None,
        **kwargs
    ) -> "LSTMAttentionModel":
        
        torch.manual_seed(self.random_seed)
        
        # Instantiate networking knowing input shapes
        input_dim = X_train_seq.shape[2]
        static_dim = train_static.shape[1]
        
        self.model = LSTMAttentionNetwork(
            input_dim=input_dim,
            hidden_dim=self.params.get("hidden_dim", 64),
            num_layers=self.params.get("num_layers", 2),
            dropout=self.params.get("dropout", 0.2),
            static_dim=static_dim
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # positive weight for imbalanced target
        pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        train_loader = self._prepare_data(X_train_seq, y_train, train_masks, train_static, True)
        val_loader = self._prepare_data(X_val_seq, y_val, val_masks, val_static, False)
        
        logger.info(f"Training LSTM with {input_dim} seq features, {static_dim} static...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y, batch_mask, batch_static in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                batch_mask, batch_static = batch_mask.to(self.device), batch_static.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(batch_x, batch_mask, batch_static)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * batch_x.size(0)
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
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                logger.info(f"Epoch {epoch+1:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} [Best]")
            else:
                patience_counter += 1
                logger.info(f"Epoch {epoch+1:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} [Patience {patience_counter}/{self.patience}]")
                if patience_counter >= self.patience:
                    logger.info("Early stopping triggered.")
                    break
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            
        logger.info(f"Training completed. Best Val Loss: {best_val_loss:.4f}")
        return self

    def predict_proba(
        self,
        X_seq: np.ndarray,
        masks: np.ndarray,
        static: np.ndarray
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        loader = self._prepare_data(X_seq, np.zeros(len(X_seq)), masks, static, False)
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
                
        return np.concatenate(all_probs)
        
    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save an untrained model.")
        torch.save(self.model.state_dict(), path)
        logger.info(f"LSTM model saved to {path}")
        
    def load(self, path: str, input_dim: int, static_dim: int = 2) -> "LSTMAttentionModel":
        self.model = LSTMAttentionNetwork(
            input_dim=input_dim,
            hidden_dim=self.params.get("hidden_dim", 64),
            num_layers=self.params.get("num_layers", 2),
            dropout=self.params.get("dropout", 0.2),
            static_dim=static_dim
        ).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
        logger.info(f"LSTM model loaded from {path}")
        return self

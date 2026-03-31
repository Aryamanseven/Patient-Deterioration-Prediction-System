"""
Sequence-Level Data Augmentation for Domain Generalization.

Provides functions to inject realistic physiological noise, temporal jitter, 
and sensor dropout to improve robustness of the temporal deep learning models.
"""
from __future__ import annotations

import numpy as np
import torch

from core.logger import get_logger

logger = get_logger("dg_augmentation")

class SequenceAugmenter:
    """Applies stochastic augmentations to multivariate time series."""
    def __init__(self, jitter_std: float = 0.05, dropout_prob: float = 0.1):
        self.jitter_std = jitter_std
        self.dropout_prob = dropout_prob
        
    def inject_noise(self, X_seq: torch.Tensor | np.ndarray) -> tuple[torch.Tensor | np.ndarray]:
        """
        Add Gaussian jitter to continuous features to simulate sensor variations.
        X_seq shape: (batch_size, seq_len, num_features)
        """
        is_tensor = isinstance(X_seq, torch.Tensor)
        original = X_seq.clone() if is_tensor else X_seq.copy()
        
        # We only want to add noise to non-zero padded values (in practice we assume X_seq is already padded but valid values are present)
        if is_tensor:
            noise = (torch.randn(original.shape, device="cpu") * self.jitter_std).to(original.device)
            original += noise
        else:
            noise = np.random.normal(0, self.jitter_std, size=original.shape)
            original += noise
            
        return original
        
    def simulate_sensor_dropout(self, X_seq: torch.Tensor | np.ndarray, masks: torch.Tensor | np.ndarray) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
        """
        Randomly drops out certain time steps to simulate disconnected sensors.
        Updates the masks accordingly.
        """
        is_tensor = isinstance(X_seq, torch.Tensor)
        aug_X = X_seq.clone() if is_tensor else X_seq.copy()
        aug_masks = masks.clone() if is_tensor else masks.copy()
        
        if is_tensor:
            # Generate on CPU for DML compatibility and then move to correct device
            drop_mask = (torch.rand(aug_masks.shape, device="cpu") > self.dropout_prob).to(aug_masks.device)
            aug_masks = aug_masks * drop_mask
            aug_X = aug_X * aug_masks.to(aug_X.dtype).unsqueeze(-1)
        else:
            drop_mask = np.random.rand(*aug_masks.shape) > self.dropout_prob
            aug_masks = aug_masks * drop_mask
            aug_X = aug_X * np.expand_dims(aug_masks, axis=-1)
            
        return aug_X, aug_masks

    def __call__(self, X_seq: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.jitter_std > 0:
            X_seq = self.inject_noise(X_seq)
        if self.dropout_prob > 0:
            X_seq, masks = self.simulate_sensor_dropout(X_seq, masks)
        return X_seq, masks

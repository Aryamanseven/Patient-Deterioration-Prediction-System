"""
Reproducibility — deterministic seed management.

Sets seeds for: Python stdlib, NumPy, PyTorch (CPU + CUDA).
Called ONCE at pipeline start.
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed — skip

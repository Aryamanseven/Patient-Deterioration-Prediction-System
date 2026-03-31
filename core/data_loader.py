"""
Centralized Data Loading — The ONLY place where the CSV is read.

Arguments are passed from config. Processes raw data, engineers
features (via features.py), handles train/val splitting (group-aware),
and prepares sequences for deep learning.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .features import (
    CATEGORICAL_FEATURE_COLUMNS,
    EPISODE_COLUMN,
    TARGET_COLUMN,
    engineer_all_features,
    get_feature_columns,
    get_numeric_feature_columns,
)
from .logger import get_logger

logger = get_logger("data_loader")


def generate_synthetic_data(num_patients: int = 50, max_seq_len: int = 48) -> pd.DataFrame:
    """Generate dummy data mimicking MIMIC III for dry-run testing."""
    logger.info(f"Generating synthetic subset for {num_patients} patients...")
    np.random.seed(42)
    rows = []
    
    for pid in range(num_patients):
        seq_len = np.random.randint(12, max_seq_len)
        age = np.random.randint(40, 90)
        is_deteriorating = np.random.rand() > 0.8
        
        # Base vitals
        hr = np.random.normal(70, 10, seq_len)
        sbp = np.random.normal(120, 15, seq_len)
        resp = np.random.normal(16, 3, seq_len)
        temp = np.random.normal(37.0, 0.5, seq_len)
        spo2 = np.random.normal(98, 2, seq_len)
        
        if is_deteriorating:
            # Add deterioration trend towards the end
            hr[-6:] += np.linspace(0, 30, 6)
            sbp[-6:] -= np.linspace(0, 40, 6)
            resp[-6:] += np.linspace(0, 10, 6)
            spo2[-6:] -= np.linspace(0, 8, 6)
            
        for t in range(seq_len):
            target = 1 if is_deteriorating and t >= seq_len - 6 else 0
            rows.append({
                "patient_id": pid,
                "episode_id": f"ep_{pid}",
                "time_hr": t,
                "heart_rate": hr[t],
                "sbp": sbp[t],
                "resp_rate": resp[t],
                "temp": temp[t],
                "spo2": spo2[t],
                "age": age,
                "comorbidity_index": np.random.choice([0, 1, 2, 3]),
                "unit_type": np.random.choice(["MICU", "SICU", "CCU"]),
                TARGET_COLUMN: target
            })
            
    return pd.DataFrame(rows)



def load_and_split_data(
    data_path: str | Path,
    test_size: float = 0.2,
    max_rows: int | None = None,
    use_advanced_features: bool = True,
    use_clinical_scores: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Load data, engineer features, and split into train/val.

    Returns:
        raw_df: Original data (subsampled if max_rows)
        train_df: Training set with all features
        val_df: Validation set with all features
        feature_cols: List of column names used as features
    """
    logger.info(f"Loading data from {data_path}...")
    
    # Load raw
    if not Path(data_path).exists() and ("synthetic" in str(data_path) or "quick_test" in str(data_path)):
        logger.warning(f"Data path {data_path} not found. Generating synthetic fallback data for dry-run testing.")
        raw_df = generate_synthetic_data()
    elif max_rows is not None:
        raw_df = pd.read_csv(data_path, nrows=max_rows)
    else:
        raw_df = pd.read_csv(data_path)
    
    logger.info(f"Loaded {len(raw_df)} rows. Positive rate: {raw_df[TARGET_COLUMN].mean():.4f}")

    # Feature engineering
    logger.info(f"Engineering features (advanced={use_advanced_features}, clinical={use_clinical_scores})...")
    featured_df = engineer_all_features(
        raw_df,
        use_advanced=use_advanced_features,
        use_clinical_scores=use_clinical_scores,
    )
    
    feature_cols = get_feature_columns(featured_df)
    logger.info(f"Generated {len(feature_cols)} features.")

    # Group-aware split
    logger.info(f"Splitting data (test_size={test_size}, seed={random_state})...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = featured_df[EPISODE_COLUMN]
    
    train_idx, val_idx = next(splitter.split(featured_df, featured_df[TARGET_COLUMN], groups))
    
    train_df = featured_df.iloc[train_idx].reset_index(drop=True)
    val_df = featured_df.iloc[val_idx].reset_index(drop=True)
    
    logger.info(f"Train set: {len(train_df)} rows. Val set: {len(val_df)} rows.")
    return raw_df, train_df, val_df, feature_cols


import torch
from torch.utils.data import Dataset

class MemoryEfficientSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list[str], max_seq_len: int = 24):
        self.max_seq_len = max_seq_len
        numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_FEATURE_COLUMNS]
        static_cols = ["age", "comorbidity_index"]
        
        grouped = df.groupby(EPISODE_COLUMN, sort=False)
        
        self.windows = []
        self.masks = []
        self.targets = []
        self.statics = []
        
        from tqdm import tqdm
        from numpy.lib.stride_tricks import sliding_window_view
        
        logger.info("Indexing sequences for Dataset (Zero-Copy View)...")
        for _, group in tqdm(grouped, desc="Processing Patients", unit="patient"):
            values = group[numeric_cols].values.astype(np.float32)
            target = group[TARGET_COLUMN].values
            
            static = np.zeros(len(static_cols), dtype=np.float32)
            for i, sc in enumerate(static_cols):
                if sc in group.columns:
                    static[i] = group[sc].iloc[0]
                    
            pad_width = max_seq_len - 1
            padded_values = np.pad(values, ((pad_width, 0), (0, 0)), mode='constant', constant_values=0)
            
            w = sliding_window_view(padded_values, window_shape=(max_seq_len, values.shape[1])).squeeze(axis=1)
            
            mask_arr = np.ones(values.shape[0], dtype=np.bool_)
            padded_mask = np.pad(mask_arr, (pad_width, 0), mode='constant', constant_values=0)
            mw = sliding_window_view(padded_mask, window_shape=max_seq_len)
            
            static_rep = np.tile(static, (len(target), 1))
            
            self.windows.append(w)
            self.targets.append(target.astype(np.float32))
            self.masks.append(mw)
            self.statics.append(static_rep)
            
        self.cum_lengths = np.cumsum([len(t) for t in self.targets])
        
        # For compatibility with legacy dimension checks
        self.input_dim = len(numeric_cols)
        self.static_dim = len(static_cols)
        
    def __len__(self):
        return self.cum_lengths[-1]
        
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return tuple(map(torch.stack, zip(*[self[i] for i in range(*idx.indices(len(self)))])))
            
        patient_idx = np.searchsorted(self.cum_lengths, idx, side='right')
        local_idx = idx if patient_idx == 0 else idx - self.cum_lengths[patient_idx - 1]
        
        # The memory pointers stay at the base numpy memory map. Convert to torch lazily.
        # Ensure we make a contiguous copy so PyTorch has a clean memory layout for the tensor backward pass.
        w_tensor = torch.from_numpy(np.ascontiguousarray(self.windows[patient_idx][local_idx]))
        m_tensor = torch.from_numpy(np.ascontiguousarray(self.masks[patient_idx][local_idx]))
        s_tensor = torch.from_numpy(np.ascontiguousarray(self.statics[patient_idx][local_idx]))
        y_tensor = torch.tensor(self.targets[patient_idx][local_idx], dtype=torch.float32)
        
        return w_tensor, y_tensor, m_tensor, s_tensor

def prepare_sequences(df: pd.DataFrame, feature_cols: list[str], max_seq_len: int = 24) -> Dataset:
    """Wrapper to remain compatible with older calls, but now directly returning PyTorch Dataset."""
    return MemoryEfficientSequenceDataset(df, feature_cols, max_seq_len)

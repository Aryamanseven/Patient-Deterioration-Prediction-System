"""
Feature engineering — the core of model performance.

Produces 257 features from 22 raw columns:

BASE FEATURES (inherited from original):
  - 8 derived clinical columns (MAP, pulse pressure, shock index, etc.)
  - 90 rolling statistics (15 vitals × 3 windows × 2 stats)
  - 45 lag features (15 vitals × 3 lags)
  - 45 delta features (15 vitals × 3 deltas)
  - 3 previous-state features

ADVANCED FEATURES (novel, for winning edge):
  - 3 clinical scores (NEWS, MEWS, qSOFA) as features
  - 1 NEWS delta (rate of change of clinical score)
  - 5 trend accelerations (2nd derivative of key vitals)
  - 10 variability indices (CV over 6h and 12h windows)
  - 5 physiological cross-correlations
  - 2 cumulative abnormality features
  - 2 time-aware features (admission phase, hours in high risk)
  - 10 min/max range features
  - 5 EWMA features (exponentially weighted moving averages)

Column naming convention:
  {vital}_{transform}_{window}
  e.g., heart_rate_roll_mean_6 = rolling mean of heart rate over 6h window

NO DATA LEAKAGE: All features use only past/current values via groupby+shift/rolling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .clinical_scores import compute_scores_vectorized

# ============================================================
# Constants
# ============================================================

TARGET_COLUMN = "deterioration_next_12h"
EPISODE_COLUMN = "episode_id"
HOUR_COLUMN = "hour_from_admission"

RAW_CATEGORICAL_COLUMNS = ("oxygen_device", "gender", "admission_type")
CATEGORICAL_FEATURE_COLUMNS = ("oxygen_device", "gender", "admission_type", "oxygen_device_prev")

RAW_NUMERIC_COLUMNS = (
    "hour_from_admission", "heart_rate", "respiratory_rate", "spo2_pct",
    "temperature_c", "systolic_bp", "diastolic_bp", "oxygen_flow",
    "mobility_score", "nurse_alert", "wbc_count", "lactate",
    "creatinine", "crp_level", "hemoglobin", "sepsis_risk_score",
    "age", "comorbidity_index",
)

SEQUENCE_SOURCE_COLUMNS = (
    "heart_rate", "respiratory_rate", "spo2_pct", "temperature_c",
    "systolic_bp", "diastolic_bp", "oxygen_flow", "wbc_count",
    "lactate", "creatinine", "crp_level", "hemoglobin",
    "sepsis_risk_score", "mean_arterial_pressure", "shock_index",
)

KEY_VITALS = ("heart_rate", "respiratory_rate", "spo2_pct", "systolic_bp", "lactate")

LAG_WINDOWS = (1, 3, 6)
ROLLING_WINDOWS = (3, 6, 12)


# ============================================================
# Episode ID reconstruction
# ============================================================

def add_episode_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct episode boundaries from hour-from-admission resets."""
    working = df.copy()
    if EPISODE_COLUMN in working.columns:
        return working
    resets = working[HOUR_COLUMN].diff().fillna(-1).le(0)
    working[EPISODE_COLUMN] = resets.cumsum().astype("int32") - 1
    return working


# ============================================================
# Derived clinical columns
# ============================================================

def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived clinical columns from raw vitals."""
    derived = {
        "mean_arterial_pressure": ((2.0 * df["diastolic_bp"]) + df["systolic_bp"]) / 3.0,
        "pulse_pressure": df["systolic_bp"] - df["diastolic_bp"],
        "shock_index": df["heart_rate"] / df["systolic_bp"].clip(lower=1.0),
        "spo2_deficit": 100.0 - df["spo2_pct"],
        "fever_excess": (df["temperature_c"] - 37.5).clip(lower=0.0),
        "hypothermia_gap": (36.0 - df["temperature_c"]).clip(lower=0.0),
        "tachypnea_excess": (df["respiratory_rate"] - 20.0).clip(lower=0.0),
        "tachycardia_excess": (df["heart_rate"] - 100.0).clip(lower=0.0),
    }
    return pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)


# ============================================================
# Validation
# ============================================================

def _validate_columns(df: pd.DataFrame, include_target: bool) -> None:
    """Ensure all required columns are present."""
    required = list(RAW_NUMERIC_COLUMNS) + list(RAW_CATEGORICAL_COLUMNS)
    if include_target:
        required.append(TARGET_COLUMN)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


# ============================================================
# Main feature engineering
# ============================================================

def engineer_all_features(
    df: pd.DataFrame,
    use_advanced: bool = True,
    use_clinical_scores: bool = True,
) -> pd.DataFrame:
    """
    Produce all features from raw data.

    Args:
        df: Raw dataset with 22 columns.
        use_advanced: If True, produce 257 features. If False, 212 (base only).
        use_clinical_scores: If True, add NEWS/MEWS/qSOFA as features.

    Returns:
        DataFrame with all features, episode IDs, and target (if present).
    """
    include_target = TARGET_COLUMN in df.columns
    _validate_columns(df, include_target=include_target)

    base = _compute_derived(add_episode_ids(df))
    grouped = base.groupby(EPISODE_COLUMN, sort=False)
    generated: dict[str, pd.Series] = {}

    # ---- Base rolling/lag/delta features ----
    for column in SEQUENCE_SOURCE_COLUMNS:
        series = base[column]
        for lag in LAG_WINDOWS:
            lagged = grouped[column].shift(lag)
            generated[f"{column}_lag_{lag}"] = lagged.fillna(series)
            generated[f"{column}_delta_{lag}"] = (series - lagged).fillna(0.0)
        for window in ROLLING_WINDOWS:
            rolling = grouped[column].rolling(window=window, min_periods=1)
            generated[f"{column}_roll_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
            generated[f"{column}_roll_std_{window}"] = (
                rolling.std().reset_index(level=0, drop=True).fillna(0.0)
            )

    # ---- Previous state features ----
    generated["nurse_alert_prev"] = grouped["nurse_alert"].shift(1).fillna(base["nurse_alert"])
    generated["mobility_score_prev"] = grouped["mobility_score"].shift(1).fillna(base["mobility_score"])
    generated["oxygen_device_prev"] = grouped["oxygen_device"].shift(1).fillna(base["oxygen_device"])

    if not use_advanced:
        # BASE ONLY — assemble and return
        engineered = pd.concat([base, pd.DataFrame(generated, index=base.index)], axis=1)
        return _finalize_types(engineered, include_target)

    # ==========================================
    # ADVANCED FEATURES — the winning edge
    # ==========================================

    # ---- Clinical scores as features ----
    if use_clinical_scores:
        scored = compute_scores_vectorized(base)
        generated["news_score"] = scored["news_score"]
        generated["mews_score"] = scored["mews_score"]
        generated["qsofa_score"] = scored["qsofa_score"]
        # NEWS rate of change
        generated["news_delta_1"] = (
            generated["news_score"] - grouped.apply(
                lambda g: compute_scores_vectorized(g)["news_score"].shift(1),
                include_groups=False
            ).reset_index(level=0, drop=True)
        ).fillna(0).astype("float32")

    # ---- Trend acceleration (2nd derivative) ----
    for col in KEY_VITALS:
        delta_1 = grouped[col].diff().fillna(0)
        generated[f"{col}_accel"] = delta_1.diff().fillna(0).astype("float32")

    # ---- Variability index (coefficient of variation) ----
    for col in KEY_VITALS:
        for w in (6, 12):
            rolling = grouped[col].rolling(window=w, min_periods=2)
            roll_mean = rolling.mean().reset_index(level=0, drop=True)
            roll_std = rolling.std().reset_index(level=0, drop=True).fillna(0)
            generated[f"{col}_cv_{w}"] = (roll_std / roll_mean.clip(lower=0.001)).fillna(0).astype("float32")

    # ---- Physiological cross-correlations ----
    if "shock_index" in base.columns:
        generated["shock_index_accel"] = grouped["shock_index"].diff().diff().fillna(0).astype("float32")
    generated["cardiac_output_proxy"] = (
        base["mean_arterial_pressure"] * base["heart_rate"] / 1000.0
    ).astype("float32")
    generated["resp_compensation"] = (
        base["respiratory_rate"] * (100.0 - base["spo2_pct"]).clip(lower=0)
    ).astype("float32")
    generated["inflammatory_burden"] = (
        (base["wbc_count"] / 10.0) * (base["crp_level"] / 50.0) * (base["lactate"] / 2.0)
    ).clip(lower=0).astype("float32")
    generated["renal_metabolic_stress"] = (base["creatinine"] * base["lactate"]).astype("float32")

    # ---- Cumulative abnormality score ----
    abnormal_flags = pd.DataFrame(index=base.index)
    abnormal_flags["hr"] = ((base["heart_rate"] < 50) | (base["heart_rate"] > 110)).astype(float)
    abnormal_flags["rr"] = ((base["respiratory_rate"] < 9) | (base["respiratory_rate"] > 24)).astype(float)
    abnormal_flags["spo2"] = (base["spo2_pct"] < 93).astype(float)
    abnormal_flags["temp"] = ((base["temperature_c"] < 35.5) | (base["temperature_c"] > 38.5)).astype(float)
    abnormal_flags["sbp"] = ((base["systolic_bp"] < 90) | (base["systolic_bp"] > 180)).astype(float)
    abnormal_flags["lactate"] = (base["lactate"] > 2.0).astype(float)
    abnormal_flags["wbc"] = ((base["wbc_count"] < 4) | (base["wbc_count"] > 12)).astype(float)
    generated["num_abnormal_vitals"] = abnormal_flags.sum(axis=1).astype("float32")
    generated["cum_abnormal_exposure"] = (
        generated["num_abnormal_vitals"].groupby(base[EPISODE_COLUMN]).cumsum()
    ).astype("float32")

    # ---- Time-aware features ----
    hours = base[HOUR_COLUMN]
    generated["admission_phase"] = pd.cut(
        hours, bins=[-1, 6, 24, 72, 999], labels=[0, 1, 2, 3]
    ).astype("float32")
    if "news_score" in generated:
        generated["hours_high_risk"] = (
            (generated["news_score"] >= 5).groupby(base[EPISODE_COLUMN]).cumsum()
        ).astype("float32")

    # ---- Rolling min/max range ----
    for col in KEY_VITALS:
        for w in (6, 12):
            rolling = grouped[col].rolling(window=w, min_periods=1)
            r_min = rolling.min().reset_index(level=0, drop=True)
            r_max = rolling.max().reset_index(level=0, drop=True)
            generated[f"{col}_range_{w}"] = (r_max - r_min).fillna(0).astype("float32")

    # ---- EWMA features ----
    for col in KEY_VITALS:
        ewm = grouped[col].apply(lambda x: x.ewm(halflife=3, min_periods=1).mean(), include_groups=False)
        if hasattr(ewm, "reset_index"):
            generated[f"{col}_ewma_3"] = ewm.reset_index(level=0, drop=True).astype("float32")
        else:
            generated[f"{col}_ewma_3"] = ewm.astype("float32")

    # ==========================================
    # Assemble
    # ==========================================
    engineered = pd.concat([base, pd.DataFrame(generated, index=base.index)], axis=1)
    return _finalize_types(engineered, include_target)


def _finalize_types(df: pd.DataFrame, include_target: bool) -> pd.DataFrame:
    """Cast all columns to proper types. Replace any remaining NaN with 0."""
    categorical_columns = {EPISODE_COLUMN, TARGET_COLUMN, *CATEGORICAL_FEATURE_COLUMNS}
    numeric_columns = [c for c in df.columns if c not in categorical_columns]
    df[numeric_columns] = df[numeric_columns].fillna(0).astype("float32")
    df[EPISODE_COLUMN] = df[EPISODE_COLUMN].astype("int32")
    if include_target and TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int8")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all columns that are model features (excluding target and episode ID)."""
    return [c for c in df.columns if c not in (TARGET_COLUMN, EPISODE_COLUMN)]


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return only numeric feature columns (excluding categoricals)."""
    return [
        c for c in df.columns
        if c not in (TARGET_COLUMN, EPISODE_COLUMN) and c not in CATEGORICAL_FEATURE_COLUMNS
    ]

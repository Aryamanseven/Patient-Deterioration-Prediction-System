from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET_COLUMN = "deterioration_next_12h"
EPISODE_COLUMN = "episode_id"
HOUR_COLUMN = "hour_from_admission"

RAW_CATEGORICAL_COLUMNS = (
    "oxygen_device",
    "gender",
    "admission_type",
)

RAW_NUMERIC_COLUMNS = (
    "hour_from_admission",
    "heart_rate",
    "respiratory_rate",
    "spo2_pct",
    "temperature_c",
    "systolic_bp",
    "diastolic_bp",
    "oxygen_flow",
    "mobility_score",
    "nurse_alert",
    "wbc_count",
    "lactate",
    "creatinine",
    "crp_level",
    "hemoglobin",
    "sepsis_risk_score",
    "age",
    "comorbidity_index",
)

DERIVED_NUMERIC_COLUMNS = (
    "mean_arterial_pressure",
    "pulse_pressure",
    "shock_index",
    "spo2_deficit",
    "fever_excess",
    "hypothermia_gap",
    "tachypnea_excess",
    "tachycardia_excess",
)

SEQUENCE_SOURCE_COLUMNS = (
    "heart_rate",
    "respiratory_rate",
    "spo2_pct",
    "temperature_c",
    "systolic_bp",
    "diastolic_bp",
    "oxygen_flow",
    "wbc_count",
    "lactate",
    "creatinine",
    "crp_level",
    "hemoglobin",
    "sepsis_risk_score",
    "mean_arterial_pressure",
    "shock_index",
)

FEATURED_PREVIOUS_STATE_COLUMNS = (
    "nurse_alert_prev",
    "mobility_score_prev",
    "oxygen_device_prev",
)

CATEGORICAL_FEATURE_COLUMNS = (
    "oxygen_device",
    "gender",
    "admission_type",
    "oxygen_device_prev",
)

LAG_WINDOWS = (1, 3, 6)
ROLLING_WINDOWS = (3, 6, 12)


def required_columns(include_target: bool = False) -> list[str]:
    columns = list(RAW_NUMERIC_COLUMNS) + list(RAW_CATEGORICAL_COLUMNS)
    if include_target:
        columns.append(TARGET_COLUMN)
    return columns


def _validate_columns(df: pd.DataFrame, include_target: bool) -> None:
    missing = [column for column in required_columns(include_target) if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")


def add_episode_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct episode boundaries using hour resets."""
    working = df.copy()
    if EPISODE_COLUMN in working.columns:
        return working

    resets = working[HOUR_COLUMN].diff().fillna(-1).le(0)
    working[EPISODE_COLUMN] = resets.cumsum().astype("int32") - 1
    return working


def _base_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
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
    return pd.concat([df.copy(), pd.DataFrame(derived, index=df.index)], axis=1)


def add_derived_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    include_target = TARGET_COLUMN in df.columns
    _validate_columns(df, include_target=include_target)
    return _base_feature_frame(add_episode_ids(df))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe rolling and trend features using historical measurements only."""
    include_target = TARGET_COLUMN in df.columns
    _validate_columns(df, include_target=include_target)

    base = add_derived_clinical_features(df)
    grouped = base.groupby(EPISODE_COLUMN, sort=False)
    generated: dict[str, pd.Series] = {}

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

    generated["nurse_alert_prev"] = grouped["nurse_alert"].shift(1).fillna(base["nurse_alert"])
    generated["mobility_score_prev"] = grouped["mobility_score"].shift(1).fillna(base["mobility_score"])
    generated["oxygen_device_prev"] = grouped["oxygen_device"].shift(1).fillna(base["oxygen_device"])

    engineered = pd.concat([base, pd.DataFrame(generated, index=base.index)], axis=1)

    categorical_columns = {EPISODE_COLUMN, TARGET_COLUMN, *CATEGORICAL_FEATURE_COLUMNS}
    numeric_columns = [column for column in engineered.columns if column not in categorical_columns]
    engineered[numeric_columns] = engineered[numeric_columns].astype("float32")
    engineered[EPISODE_COLUMN] = engineered[EPISODE_COLUMN].astype("int32")
    if TARGET_COLUMN in engineered.columns:
        engineered[TARGET_COLUMN] = engineered[TARGET_COLUMN].astype("int8")
    return engineered


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in (TARGET_COLUMN, EPISODE_COLUMN)]


def assign_risk_band(
    scores: pd.Series | np.ndarray,
    watch_threshold: float,
    alert_threshold: float,
) -> pd.Series:
    score_array = np.asarray(scores, dtype="float32")
    bands = np.select(
        [score_array >= alert_threshold, score_array >= watch_threshold],
        ["High", "Moderate"],
        default="Low",
    )
    return pd.Series(bands, index=getattr(scores, "index", None), name="risk_band")


def save_metadata(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

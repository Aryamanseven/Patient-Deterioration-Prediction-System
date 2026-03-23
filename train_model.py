from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physio_warning import (  # noqa: E402
    CATEGORICAL_FEATURE_COLUMNS,
    EPISODE_COLUMN,
    HOUR_COLUMN,
    TARGET_COLUMN,
    add_episode_ids,
    assign_risk_band,
    engineer_features,
    get_model_feature_columns,
    save_metadata,
)


DEFAULT_MODEL_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 800,
    "depth": 7,
    "learning_rate": 0.05,
    "auto_class_weights": "Balanced",
    "random_strength": 0.5,
    "l2_leaf_reg": 5.0,
}


def _safe_float(value: float | np.floating) -> float:
    return round(float(value), 6)


def compute_thresholds(y_true: pd.Series, scores: np.ndarray) -> tuple[float, float, dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        default_threshold = 0.5
        return default_threshold, default_threshold, {
            "watch_precision": 0.0,
            "watch_recall": 0.0,
            "alert_precision": 0.0,
            "alert_recall": 0.0,
            "alert_f1": 0.0,
        }

    precision_points = precision[1:]
    recall_points = recall[1:]
    f1_points = 2 * precision_points * recall_points / np.clip(
        precision_points + recall_points,
        1e-9,
        None,
    )

    alert_index = int(np.nanargmax(f1_points))
    watch_candidates = np.where(recall_points >= 0.85)[0]
    if len(watch_candidates) > 0:
        watch_index = int(watch_candidates[np.argmax(precision_points[watch_candidates])])
    else:
        watch_index = int(np.nanargmax(recall_points))

    alert_threshold = float(thresholds[alert_index])
    watch_threshold = float(thresholds[watch_index])
    if watch_threshold > alert_threshold:
        watch_threshold = min(alert_threshold * 0.7, 0.35)

    summary = {
        "watch_precision": _safe_float(precision_points[watch_index]),
        "watch_recall": _safe_float(recall_points[watch_index]),
        "alert_precision": _safe_float(precision_points[alert_index]),
        "alert_recall": _safe_float(recall_points[alert_index]),
        "alert_f1": _safe_float(f1_points[alert_index]),
    }
    return watch_threshold, alert_threshold, summary


def classification_summary(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
    prefix: str,
) -> dict[str, float]:
    predictions = (scores >= threshold).astype("int8")
    return {
        f"{prefix}_threshold": _safe_float(threshold),
        f"{prefix}_precision": _safe_float(precision_score(y_true, predictions, zero_division=0)),
        f"{prefix}_recall": _safe_float(recall_score(y_true, predictions, zero_division=0)),
        f"{prefix}_f1": _safe_float(f1_score(y_true, predictions, zero_division=0)),
        f"{prefix}_positive_rate": _safe_float(predictions.mean()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the patient deterioration early warning model and save artifacts.",
    )
    parser.add_argument("--train-path", default="dataset/train.csv")
    parser.add_argument("--val-path", default="dataset/val_no_labels.csv")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(train_path)
    train_with_ids = add_episode_ids(train_raw)
    train_featured = engineer_features(train_with_ids)

    feature_columns = get_model_feature_columns(train_featured)
    X = train_featured[feature_columns]
    y = train_featured[TARGET_COLUMN].astype("int8")
    groups = train_featured[EPISODE_COLUMN]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_index, holdout_index = next(splitter.split(X, y, groups))

    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_holdout = X.iloc[holdout_index]
    y_holdout = y.iloc[holdout_index]

    eval_params = {
        **DEFAULT_MODEL_PARAMS,
        "random_seed": args.random_state,
        "verbose": 100,
        "od_type": "Iter",
        "od_wait": 60,
    }
    eval_model = CatBoostClassifier(**eval_params)
    eval_model.fit(
        X_train,
        y_train,
        cat_features=list(CATEGORICAL_FEATURE_COLUMNS),
        eval_set=(X_holdout, y_holdout),
        use_best_model=True,
    )

    holdout_scores = eval_model.predict_proba(X_holdout)[:, 1]
    watch_threshold, alert_threshold, threshold_summary = compute_thresholds(y_holdout, holdout_scores)

    metrics = {
        "roc_auc": _safe_float(roc_auc_score(y_holdout, holdout_scores)),
        "pr_auc": _safe_float(average_precision_score(y_holdout, holdout_scores)),
        "brier_score": _safe_float(brier_score_loss(y_holdout, holdout_scores)),
        "holdout_positive_rate": _safe_float(y_holdout.mean()),
        **threshold_summary,
        **classification_summary(y_holdout, holdout_scores, watch_threshold, "watch"),
        **classification_summary(y_holdout, holdout_scores, alert_threshold, "alert"),
    }

    holdout_predictions = train_featured.iloc[holdout_index][
        [EPISODE_COLUMN, HOUR_COLUMN, TARGET_COLUMN]
    ].copy()
    holdout_predictions["risk_score"] = holdout_scores
    holdout_predictions["risk_band"] = assign_risk_band(
        holdout_scores,
        watch_threshold=watch_threshold,
        alert_threshold=alert_threshold,
    )
    holdout_predictions["predicted_alert"] = (
        holdout_predictions["risk_score"] >= alert_threshold
    ).astype("int8")
    holdout_predictions.to_csv(artifact_dir / "holdout_predictions.csv", index=False)

    importance = eval_model.get_feature_importance(prettified=True)
    importance.columns = ["feature", "importance"]
    importance.to_csv(artifact_dir / "feature_importance.csv", index=False)

    best_iteration = eval_model.get_best_iteration()
    final_iterations = max(int(best_iteration) + 1, 250)
    final_params = {
        **DEFAULT_MODEL_PARAMS,
        "iterations": final_iterations,
        "random_seed": args.random_state,
        "verbose": False,
    }

    final_model = CatBoostClassifier(**final_params)
    final_model.fit(X, y, cat_features=list(CATEGORICAL_FEATURE_COLUMNS), verbose=False)
    final_model.save_model(str(artifact_dir / "deterioration_model.cbm"))

    episode_lengths = train_with_ids.groupby(EPISODE_COLUMN, sort=False).size()
    metadata = {
        "train_path": str(train_path.as_posix()),
        "validation_path": str(val_path.as_posix()),
        "model_path": str((artifact_dir / "deterioration_model.cbm").as_posix()),
        "feature_columns": feature_columns,
        "categorical_features": list(CATEGORICAL_FEATURE_COLUMNS),
        "thresholds": {
            "watch": _safe_float(watch_threshold),
            "alert": _safe_float(alert_threshold),
        },
        "metrics": metrics,
        "model_params": {
            **DEFAULT_MODEL_PARAMS,
            "iterations": final_iterations,
            "random_seed": args.random_state,
        },
        "dataset_summary": {
            "train_rows": int(len(train_with_ids)),
            "train_episodes": int(train_with_ids[EPISODE_COLUMN].nunique()),
            "holdout_rows": int(len(holdout_index)),
            "holdout_episodes": int(train_with_ids.iloc[holdout_index][EPISODE_COLUMN].nunique()),
            "positive_rate": _safe_float(train_with_ids[TARGET_COLUMN].mean()),
            "episode_length_min": int(episode_lengths.min()),
            "episode_length_median": _safe_float(episode_lengths.median()),
            "episode_length_max": int(episode_lengths.max()),
        },
        "top_features": importance.head(15).to_dict(orient="records"),
    }
    save_metadata(artifact_dir / "metadata.json", metadata)

    if val_path.exists():
        val_raw = pd.read_csv(val_path)
        val_featured = engineer_features(add_episode_ids(val_raw))
        val_scores = final_model.predict_proba(val_featured[feature_columns])[:, 1]
        val_output = val_raw.copy()
        val_output[EPISODE_COLUMN] = val_featured[EPISODE_COLUMN]
        val_output["deterioration_risk"] = val_scores
        val_output["risk_band"] = assign_risk_band(
            val_scores,
            watch_threshold=watch_threshold,
            alert_threshold=alert_threshold,
        )
        val_output["predicted_alert"] = (val_output["deterioration_risk"] >= alert_threshold).astype("int8")
        val_output.to_csv(artifact_dir / "val_predictions.csv", index=False)

    print("Training complete.")
    print(f"ROC-AUC: {metrics['roc_auc']}")
    print(f"PR-AUC: {metrics['pr_auc']}")
    print(f"Watch threshold: {metadata['thresholds']['watch']}")
    print(f"Alert threshold: {metadata['thresholds']['alert']}")
    print(f"Artifacts saved to: {artifact_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
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

from physio_warning.features import (  # noqa: E402
    CATEGORICAL_FEATURE_COLUMNS,
    EPISODE_COLUMN,
    HOUR_COLUMN,
    TARGET_COLUMN,
    add_episode_ids,
    engineer_features,
)


CATBOOST_CANDIDATES = (
    {
        "name": "catboost_gpu_base",
        "params": {
            "iterations": 1200,
            "depth": 7,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5.0,
            "random_strength": 0.5,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.5,
            "border_count": 254,
        },
    },
    {
        "name": "catboost_gpu_deep",
        "params": {
            "iterations": 1500,
            "depth": 8,
            "learning_rate": 0.035,
            "l2_leaf_reg": 7.0,
            "random_strength": 1.0,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 1.0,
            "border_count": 254,
        },
    },
    {
        "name": "catboost_gpu_fast",
        "params": {
            "iterations": 900,
            "depth": 6,
            "learning_rate": 0.07,
            "l2_leaf_reg": 4.0,
            "random_strength": 0.25,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.3,
            "border_count": 128,
        },
    },
    {
        "name": "catboost_gpu_regularized",
        "params": {
            "iterations": 1600,
            "depth": 7,
            "learning_rate": 0.04,
            "l2_leaf_reg": 9.0,
            "random_strength": 1.5,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 1.5,
            "border_count": 254,
        },
    },
    {
        "name": "catboost_gpu_subsample",
        "params": {
            "iterations": 1300,
            "depth": 8,
            "learning_rate": 0.05,
            "l2_leaf_reg": 6.0,
            "random_strength": 0.7,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.85,
            "border_count": 254,
        },
    },
    {
        "name": "catboost_gpu_low_lr",
        "params": {
            "iterations": 2000,
            "depth": 8,
            "learning_rate": 0.025,
            "l2_leaf_reg": 6.0,
            "random_strength": 0.4,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.8,
            "border_count": 254,
        },
    },
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune CatBoost on GPU and search validation-based ensembles with deep models.",
    )
    parser.add_argument("--train-path", default="dataset/train.csv")
    parser.add_argument("--deep-artifact-dir", default="artifacts/deep_models")
    parser.add_argument("--artifact-dir", default="artifacts/model_search")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def _safe_round(value: float) -> float:
    return round(float(value), 6)


def split_frame(
    frame: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = frame[EPISODE_COLUMN]
    labels = frame[TARGET_COLUMN]

    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_val_index, holdout_index = next(outer_split.split(frame, labels, groups))
    train_val_frame = frame.iloc[train_val_index].copy()
    holdout_frame = frame.iloc[holdout_index].copy()

    inner_groups = train_val_frame[EPISODE_COLUMN]
    inner_labels = train_val_frame[TARGET_COLUMN]
    relative_val_size = val_size / max(1.0 - test_size, 1e-6)
    inner_split = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_val_size,
        random_state=random_state,
    )
    train_index, val_index = next(inner_split.split(train_val_frame, inner_labels, inner_groups))
    train_frame = train_val_frame.iloc[train_index].copy()
    val_frame = train_val_frame.iloc[val_index].copy()
    return train_frame, val_frame, holdout_frame


def compute_binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "brier_score": float(brier_score_loss(y_true, scores)),
    }


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        threshold = 0.5
        predictions = (scores >= threshold).astype(int)
        return {
            "watch_threshold": threshold,
            "watch_precision": float(precision_score(y_true, predictions, zero_division=0)),
            "watch_recall": float(recall_score(y_true, predictions, zero_division=0)),
            "watch_f1": float(f1_score(y_true, predictions, zero_division=0)),
            "alert_threshold": threshold,
            "alert_precision": float(precision_score(y_true, predictions, zero_division=0)),
            "alert_recall": float(recall_score(y_true, predictions, zero_division=0)),
            "alert_f1": float(f1_score(y_true, predictions, zero_division=0)),
        }

    precision_points = precision[1:]
    recall_points = recall[1:]
    f1_points = 2 * precision_points * recall_points / np.clip(
        precision_points + recall_points,
        1e-9,
        None,
    )
    alert_index = int(np.nanargmax(f1_points))
    alert_threshold = float(thresholds[alert_index])

    watch_candidates = np.where((recall_points >= 0.85) & (thresholds <= alert_threshold))[0]
    if len(watch_candidates) == 0:
        watch_candidates = np.where(thresholds <= alert_threshold)[0]
    if len(watch_candidates) == 0:
        watch_candidates = np.array([alert_index])
    watch_index = int(watch_candidates[np.argmax(precision_points[watch_candidates])])
    watch_threshold = float(thresholds[watch_index])

    watch_predictions = (scores >= watch_threshold).astype(int)
    alert_predictions = (scores >= alert_threshold).astype(int)
    return {
        "watch_threshold": watch_threshold,
        "watch_precision": float(precision_score(y_true, watch_predictions, zero_division=0)),
        "watch_recall": float(recall_score(y_true, watch_predictions, zero_division=0)),
        "watch_f1": float(f1_score(y_true, watch_predictions, zero_division=0)),
        "alert_threshold": alert_threshold,
        "alert_precision": float(precision_score(y_true, alert_predictions, zero_division=0)),
        "alert_recall": float(recall_score(y_true, alert_predictions, zero_division=0)),
        "alert_f1": float(f1_score(y_true, alert_predictions, zero_division=0)),
    }


def compute_detailed_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": _safe_round(roc_auc_score(y_true, scores)),
        "pr_auc": _safe_round(average_precision_score(y_true, scores)),
        "brier_score": _safe_round(brier_score_loss(y_true, scores)),
        "holdout_positive_rate": _safe_round(float(np.mean(y_true))),
        **{key: _safe_round(value) for key, value in threshold_metrics(y_true, scores).items()},
    }


def write_metric_summary(rows: list[dict[str, object]], artifact_dir: Path) -> None:
    summary_rows = []
    markdown_lines = [
        "# Best Model Search Summary",
        "",
        "Thresholds below are computed from each model's own evaluation scores.",
        "",
    ]
    for row in rows:
        prediction_frame = pd.read_csv(row["holdout_prediction_path"])
        y_true = prediction_frame[TARGET_COLUMN].to_numpy(dtype=int)
        scores = prediction_frame[row["score_column"]].to_numpy(dtype=float)
        metrics = compute_detailed_metrics(y_true, scores)
        summary_rows.append({**row, **metrics})
        markdown_lines.extend(
            [
                f"## {row['model']}",
                f"- Family: `{row['family']}`",
                f"- ROC-AUC: `{metrics['roc_auc']:.4f}`",
                f"- PR-AUC: `{metrics['pr_auc']:.4f}`",
                f"- Brier score: `{metrics['brier_score']:.4f}`",
                f"- Holdout positive rate: `{metrics['holdout_positive_rate']:.4f}`",
                (
                    f"- Watch threshold: `{metrics['watch_threshold']:.4f}` "
                    f"with precision `{metrics['watch_precision']:.4f}` "
                    f"and recall `{metrics['watch_recall']:.4f}`"
                ),
                (
                    f"- Alert threshold: `{metrics['alert_threshold']:.4f}` "
                    f"with precision `{metrics['alert_precision']:.4f}`, "
                    f"recall `{metrics['alert_recall']:.4f}`, and F1 `{metrics['alert_f1']:.4f}`"
                ),
            ],
        )
        if row.get("validation_pr_auc") is not None:
            markdown_lines.append(f"- Validation PR-AUC: `{float(row['validation_pr_auc']):.4f}`")
        if row.get("config"):
            markdown_lines.append(f"- Config: `{row['config']}`")
        markdown_lines.append("")

    pd.DataFrame(summary_rows).to_csv(artifact_dir / "best_model_metric_details.csv", index=False)
    (artifact_dir / "best_model_metric_details.json").write_text(
        json.dumps(summary_rows, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "best_model_metric_summary.md").write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )


def _save_prediction_frame(
    source_frame: pd.DataFrame,
    scores: np.ndarray,
    output_path: Path,
) -> pd.DataFrame:
    prediction_frame = source_frame[[EPISODE_COLUMN, HOUR_COLUMN, TARGET_COLUMN]].copy()
    prediction_frame["risk_score"] = scores
    prediction_frame.to_csv(output_path, index=False)
    return prediction_frame


def retrain_catboost_on_development(
    model_name: str,
    params: dict[str, object],
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    feature_columns: list[str],
    artifact_dir: Path,
    random_state: int,
) -> dict[str, object]:
    development_frame = pd.concat([train_frame, val_frame], axis=0).sort_index()
    full_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "auto_class_weights": "Balanced",
        "random_seed": random_state,
        "task_type": "GPU",
        "devices": "0",
        "verbose": 100,
        **params,
    }
    model = CatBoostClassifier(**full_params)
    model.fit(
        development_frame[feature_columns],
        development_frame[TARGET_COLUMN],
        cat_features=list(CATEGORICAL_FEATURE_COLUMNS),
    )
    holdout_scores = model.predict_proba(holdout_frame[feature_columns])[:, 1]
    holdout_metrics = compute_binary_metrics(holdout_frame[TARGET_COLUMN].to_numpy(dtype=int), holdout_scores)
    holdout_path = artifact_dir / f"{model_name}_holdout_predictions.csv"
    _save_prediction_frame(holdout_frame, holdout_scores, holdout_path)
    model_path = artifact_dir / f"{model_name}.cbm"
    model.save_model(str(model_path))
    metrics_path = artifact_dir / f"{model_name}_metrics.json"
    metrics_path.write_text(json.dumps(holdout_metrics, indent=2), encoding="utf-8")
    return {
        "model": model_name,
        "family": "catboost_retrain80",
        "validation_pr_auc": None,
        "roc_auc": _safe_round(holdout_metrics["roc_auc"]),
        "pr_auc": _safe_round(holdout_metrics["pr_auc"]),
        "brier_score": _safe_round(holdout_metrics["brier_score"]),
        "checkpoint_path": str(model_path.as_posix()),
        "val_prediction_path": "",
        "holdout_prediction_path": str(holdout_path.as_posix()),
        "score_column": "risk_score",
        "config": json.dumps(params),
    }


def load_deep_model_predictions(deep_artifact_dir: Path) -> tuple[list[dict[str, object]], list[str]]:
    comparison = pd.read_csv(deep_artifact_dir / "model_comparison.csv")
    deep_models = comparison.loc[comparison["family"] == "deep"].copy()
    deep_models = deep_models.sort_values("validation_pr_auc", ascending=False)
    model_rows = []
    model_names = []
    for _, row in deep_models.iterrows():
        model_name = str(row["model"])
        model_rows.append(
            {
                "model": model_name,
                "family": "deep_existing",
                "validation_pr_auc": _safe_round(float(row["validation_pr_auc"])),
                "roc_auc": _safe_round(float(row["roc_auc"])),
                "pr_auc": _safe_round(float(row["pr_auc"])),
                "brier_score": _safe_round(float(row["brier_score"])),
                "holdout_prediction_path": str(row["holdout_prediction_path"]),
                "val_prediction_path": str(deep_artifact_dir / f"{model_name}_val_predictions.csv"),
                "score_column": "risk_score",
                "config": str(row["config"]),
            },
        )
        model_names.append(model_name)
    return model_rows, model_names


def merge_prediction_frames(
    base_frame: pd.DataFrame,
    prediction_sources: list[tuple[str, Path]],
) -> pd.DataFrame:
    merged = base_frame.copy()
    for model_name, path in prediction_sources:
        frame = pd.read_csv(path)
        merged = merged.merge(
            frame[[EPISODE_COLUMN, HOUR_COLUMN, "risk_score"]].rename(columns={"risk_score": model_name}),
            on=[EPISODE_COLUMN, HOUR_COLUMN],
            how="inner",
        )
    return merged


def search_pair_blend(
    val_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    primary_model: str,
    secondary_model: str,
    weights: np.ndarray,
) -> dict[str, object]:
    best = None
    y_val = val_frame[TARGET_COLUMN].to_numpy(dtype=int)
    y_holdout = holdout_frame[TARGET_COLUMN].to_numpy(dtype=int)

    for primary_weight in weights:
        blend = (primary_weight * val_frame[primary_model]) + ((1.0 - primary_weight) * val_frame[secondary_model])
        pr_auc = average_precision_score(y_val, blend.to_numpy(dtype=float))
        if best is None or pr_auc > best["validation_pr_auc"]:
            holdout_scores = (
                (primary_weight * holdout_frame[primary_model]) + ((1.0 - primary_weight) * holdout_frame[secondary_model])
            ).to_numpy(dtype=float)
            best = {
                "weights": {primary_model: _safe_round(primary_weight), secondary_model: _safe_round(1.0 - primary_weight)},
                "validation_pr_auc": float(pr_auc),
                "holdout_metrics": compute_binary_metrics(y_holdout, holdout_scores),
                "holdout_scores": holdout_scores,
            }
    return best


def search_triplet_blend(
    val_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    models: list[str],
    cat_weight_values: np.ndarray,
    split_values: np.ndarray,
) -> dict[str, object]:
    best = None
    y_val = val_frame[TARGET_COLUMN].to_numpy(dtype=int)
    y_holdout = holdout_frame[TARGET_COLUMN].to_numpy(dtype=int)

    primary, second, third = models
    for primary_weight in cat_weight_values:
        remaining = 1.0 - primary_weight
        for second_share in split_values:
            second_weight = remaining * second_share
            third_weight = remaining - second_weight
            weights = np.array([primary_weight, second_weight, third_weight], dtype=float)
            blend = (
                (weights[0] * val_frame[primary])
                + (weights[1] * val_frame[second])
                + (weights[2] * val_frame[third])
            )
            pr_auc = average_precision_score(y_val, blend.to_numpy(dtype=float))
            if best is None or pr_auc > best["validation_pr_auc"]:
                holdout_scores = (
                    (weights[0] * holdout_frame[primary])
                    + (weights[1] * holdout_frame[second])
                    + (weights[2] * holdout_frame[third])
                ).to_numpy(dtype=float)
                best = {
                    "weights": {
                        primary: _safe_round(weights[0]),
                        second: _safe_round(weights[1]),
                        third: _safe_round(weights[2]),
                    },
                    "validation_pr_auc": float(pr_auc),
                    "holdout_metrics": compute_binary_metrics(y_holdout, holdout_scores),
                    "holdout_scores": holdout_scores,
                }
    return best


def search_logistic_stack(
    val_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    model_columns: list[str],
) -> dict[str, object]:
    X_val = val_frame[model_columns].to_numpy(dtype=float)
    y_val = val_frame[TARGET_COLUMN].to_numpy(dtype=int)
    X_holdout = holdout_frame[model_columns].to_numpy(dtype=float)
    y_holdout = holdout_frame[TARGET_COLUMN].to_numpy(dtype=int)

    stacker = LogisticRegression(
        C=0.5,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    stacker.fit(X_val, y_val)
    val_scores = stacker.predict_proba(X_val)[:, 1]
    holdout_scores = stacker.predict_proba(X_holdout)[:, 1]
    return {
        "validation_pr_auc": float(average_precision_score(y_val, val_scores)),
        "holdout_metrics": compute_binary_metrics(y_holdout, holdout_scores),
        "holdout_scores": holdout_scores,
        "weights": {
            column: _safe_round(weight)
            for column, weight in zip(model_columns, stacker.coef_[0].tolist())
        },
        "intercept": _safe_round(stacker.intercept_[0]),
    }


def main() -> None:
    args = build_parser().parse_args()
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    deep_artifact_dir = Path(args.deep_artifact_dir)

    raw_train = pd.read_csv(args.train_path)
    featured = engineer_features(add_episode_ids(raw_train))
    feature_columns = [column for column in featured.columns if column not in (TARGET_COLUMN, EPISODE_COLUMN)]

    train_frame, val_frame, holdout_frame = split_frame(
        featured,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    X_train = train_frame[feature_columns]
    y_train = train_frame[TARGET_COLUMN]
    X_val = val_frame[feature_columns]
    y_val = val_frame[TARGET_COLUMN]
    X_holdout = holdout_frame[feature_columns]
    y_holdout = holdout_frame[TARGET_COLUMN]

    comparison_rows: list[dict[str, object]] = []
    catboost_rows: list[dict[str, object]] = []

    for candidate in CATBOOST_CANDIDATES:
        model_name = candidate["name"]
        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "auto_class_weights": "Balanced",
            "random_seed": args.random_state,
            "task_type": "GPU",
            "devices": "0",
            "verbose": 100,
            "od_type": "Iter",
            "od_wait": 80,
            **candidate["params"],
        }
        model = CatBoostClassifier(**params)
        model.fit(
            X_train,
            y_train,
            cat_features=list(CATEGORICAL_FEATURE_COLUMNS),
            eval_set=(X_val, y_val),
            use_best_model=True,
        )
        val_scores = model.predict_proba(X_val)[:, 1]
        holdout_scores = model.predict_proba(X_holdout)[:, 1]
        val_pr_auc = average_precision_score(y_val, val_scores)
        holdout_metrics = compute_binary_metrics(y_holdout.to_numpy(dtype=int), holdout_scores)

        val_path = artifact_dir / f"{model_name}_val_predictions.csv"
        holdout_path = artifact_dir / f"{model_name}_holdout_predictions.csv"
        _save_prediction_frame(val_frame, val_scores, val_path)
        _save_prediction_frame(holdout_frame, holdout_scores, holdout_path)
        model.save_model(str(artifact_dir / f"{model_name}.cbm"))

        row = {
            "model": model_name,
            "family": "catboost_search",
            "validation_pr_auc": _safe_round(val_pr_auc),
            "roc_auc": _safe_round(holdout_metrics["roc_auc"]),
            "pr_auc": _safe_round(holdout_metrics["pr_auc"]),
            "brier_score": _safe_round(holdout_metrics["brier_score"]),
            "checkpoint_path": str((artifact_dir / f"{model_name}.cbm").as_posix()),
            "val_prediction_path": str(val_path.as_posix()),
            "holdout_prediction_path": str(holdout_path.as_posix()),
            "score_column": "risk_score",
            "config": json.dumps(candidate["params"]),
        }
        catboost_rows.append(row)
        comparison_rows.append(row)

    deep_rows, deep_model_names = load_deep_model_predictions(deep_artifact_dir)
    comparison_rows.extend(deep_rows)

    best_catboost = max(catboost_rows, key=lambda row: float(row["validation_pr_auc"]))
    top_deep_rows = sorted(deep_rows, key=lambda row: float(row["validation_pr_auc"]), reverse=True)[:3]
    top_deep_names = [row["model"] for row in top_deep_rows]

    base_val = val_frame[[EPISODE_COLUMN, HOUR_COLUMN, TARGET_COLUMN]].copy()
    base_holdout = holdout_frame[[EPISODE_COLUMN, HOUR_COLUMN, TARGET_COLUMN]].copy()

    val_prediction_sources = [(best_catboost["model"], Path(best_catboost["val_prediction_path"]))]
    val_prediction_sources.extend(
        [(row["model"], Path(row["val_prediction_path"])) for row in top_deep_rows]
    )
    holdout_prediction_sources = [(best_catboost["model"], Path(best_catboost["holdout_prediction_path"]))]
    holdout_prediction_sources.extend(
        [(row["model"], Path(row["holdout_prediction_path"])) for row in top_deep_rows]
    )

    merged_val = merge_prediction_frames(base_val, val_prediction_sources)
    merged_holdout = merge_prediction_frames(base_holdout, holdout_prediction_sources)

    ensemble_candidates = []
    weight_grid = np.linspace(0.50, 0.98, 49)
    for deep_name in top_deep_names:
        result = search_pair_blend(
            val_frame=merged_val,
            holdout_frame=merged_holdout,
            primary_model=best_catboost["model"],
            secondary_model=deep_name,
            weights=weight_grid,
        )
        ensemble_name = f"{best_catboost['model']}_plus_{deep_name}"
        holdout_path = artifact_dir / f"{ensemble_name}_holdout_predictions.csv"
        ensemble_frame = merged_holdout[[EPISODE_COLUMN, HOUR_COLUMN, TARGET_COLUMN]].copy()
        ensemble_frame[ensemble_name] = result["holdout_scores"]
        ensemble_frame.to_csv(holdout_path, index=False)
        row = {
            "model": ensemble_name,
            "family": "weighted_ensemble",
            "validation_pr_auc": _safe_round(result["validation_pr_auc"]),
            "roc_auc": _safe_round(result["holdout_metrics"]["roc_auc"]),
            "pr_auc": _safe_round(result["holdout_metrics"]["pr_auc"]),
            "brier_score": _safe_round(result["holdout_metrics"]["brier_score"]),
            "checkpoint_path": "",
            "val_prediction_path": "",
            "holdout_prediction_path": str(holdout_path.as_posix()),
            "score_column": ensemble_name,
            "config": json.dumps(result["weights"]),
        }
        ensemble_candidates.append(row)
        comparison_rows.append(row)

    if len(top_deep_names) >= 2:
        triplet_models = [best_catboost["model"], top_deep_names[0], top_deep_names[1]]
        result = search_triplet_blend(
            val_frame=merged_val,
            holdout_frame=merged_holdout,
            models=triplet_models,
            cat_weight_values=np.linspace(0.55, 0.95, 9),
            split_values=np.linspace(0.0, 1.0, 21),
        )
        ensemble_name = f"{best_catboost['model']}_plus_top2_transformers"
        holdout_path = artifact_dir / f"{ensemble_name}_holdout_predictions.csv"
        ensemble_frame = merged_holdout[[EPISODE_COLUMN, HOUR_COLUMN, TARGET_COLUMN]].copy()
        ensemble_frame[ensemble_name] = result["holdout_scores"]
        ensemble_frame.to_csv(holdout_path, index=False)
        row = {
            "model": ensemble_name,
            "family": "weighted_ensemble",
            "validation_pr_auc": _safe_round(result["validation_pr_auc"]),
            "roc_auc": _safe_round(result["holdout_metrics"]["roc_auc"]),
            "pr_auc": _safe_round(result["holdout_metrics"]["pr_auc"]),
            "brier_score": _safe_round(result["holdout_metrics"]["brier_score"]),
            "checkpoint_path": "",
            "val_prediction_path": "",
            "holdout_prediction_path": str(holdout_path.as_posix()),
            "score_column": ensemble_name,
            "config": json.dumps(result["weights"]),
        }
        ensemble_candidates.append(row)
        comparison_rows.append(row)

    stack_models = [best_catboost["model"], *top_deep_names[:2]]
    stack_result = search_logistic_stack(
        val_frame=merged_val,
        holdout_frame=merged_holdout,
        model_columns=stack_models,
    )
    stack_name = f"{best_catboost['model']}_logistic_stack"
    stack_holdout_path = artifact_dir / f"{stack_name}_holdout_predictions.csv"
    stack_frame = merged_holdout[[EPISODE_COLUMN, HOUR_COLUMN, TARGET_COLUMN]].copy()
    stack_frame[stack_name] = stack_result["holdout_scores"]
    stack_frame.to_csv(stack_holdout_path, index=False)
    stack_row = {
        "model": stack_name,
        "family": "stacking",
        "validation_pr_auc": _safe_round(stack_result["validation_pr_auc"]),
        "roc_auc": _safe_round(stack_result["holdout_metrics"]["roc_auc"]),
        "pr_auc": _safe_round(stack_result["holdout_metrics"]["pr_auc"]),
        "brier_score": _safe_round(stack_result["holdout_metrics"]["brier_score"]),
        "checkpoint_path": "",
        "val_prediction_path": "",
        "holdout_prediction_path": str(stack_holdout_path.as_posix()),
        "score_column": stack_name,
        "config": json.dumps(
            {
                "weights": stack_result["weights"],
                "intercept": stack_result["intercept"],
                "models": stack_models,
            },
        ),
    }
    comparison_rows.append(stack_row)

    best_catboost_full_row = retrain_catboost_on_development(
        model_name=f"{best_catboost['model']}_train80",
        params=json.loads(best_catboost["config"]),
        train_frame=train_frame,
        val_frame=val_frame,
        holdout_frame=holdout_frame,
        feature_columns=feature_columns,
        artifact_dir=artifact_dir,
        random_state=args.random_state,
    )
    comparison_rows.append(best_catboost_full_row)

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["pr_auc", "roc_auc"],
        ascending=False,
    )
    comparison_df.to_csv(artifact_dir / "best_model_comparison.csv", index=False)
    write_metric_summary(comparison_df.to_dict(orient="records"), artifact_dir)

    best_overall = comparison_df.iloc[0]
    summary = {
        "best_model": str(best_overall["model"]),
        "best_family": str(best_overall["family"]),
        "best_holdout_pr_auc": _safe_round(best_overall["pr_auc"]),
        "best_holdout_roc_auc": _safe_round(best_overall["roc_auc"]),
        "best_catboost_candidate": best_catboost["model"],
        "top_deep_models_by_validation_pr_auc": top_deep_names,
    }
    (artifact_dir / "best_model_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Best model search complete.")
    print(comparison_df[["model", "family", "validation_pr_auc", "roc_auc", "pr_auc", "brier_score"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""Revalidate and retune the best tabular models on the provided dataset only.

This script was added to reproduce the CPU-based holdout search that produced
the cleaned findings under ``artifacts/model_search_revalidated_20260326``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
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


REPO_BASELINE_PARAMS = {
    "iterations": 800,
    "depth": 7,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5.0,
    "random_strength": 0.5,
}

REPO_GPU_CANDIDATES = (
    {
        "name": "repo_catboost_base",
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
        "name": "repo_catboost_deep",
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
        "name": "repo_catboost_fast",
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
        "name": "repo_catboost_regularized",
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
        "name": "repo_catboost_subsample",
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
        "name": "repo_catboost_low_lr",
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

MANUAL_CATBOOST_CANDIDATES = (
    {
        "name": "manual_cpu_conservative",
        "params": {
            "iterations": 1600,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 10.0,
            "random_strength": 1.1,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.82,
            "border_count": 254,
            "rsm": 0.85,
        },
    },
    {
        "name": "manual_cpu_balanced_depth7",
        "params": {
            "iterations": 1400,
            "depth": 7,
            "learning_rate": 0.04,
            "l2_leaf_reg": 8.0,
            "random_strength": 0.8,
            "bootstrap_type": "MVS",
            "subsample": 0.9,
            "border_count": 254,
            "rsm": 0.9,
        },
    },
    {
        "name": "manual_cpu_regularized_depth8",
        "params": {
            "iterations": 1500,
            "depth": 8,
            "learning_rate": 0.03,
            "l2_leaf_reg": 12.0,
            "random_strength": 1.4,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.78,
            "border_count": 254,
            "rsm": 0.8,
        },
    },
    {
        "name": "manual_cpu_strong_columnsample",
        "params": {
            "iterations": 1800,
            "depth": 7,
            "learning_rate": 0.025,
            "l2_leaf_reg": 9.0,
            "random_strength": 0.9,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 1.2,
            "border_count": 128,
            "rsm": 0.72,
        },
    },
)

MANDATORY_FINALIST_NAMES = (
    "repo_train_model_baseline",
    "repo_catboost_base",
    "repo_catboost_subsample",
)


@dataclass
class Candidate:
    name: str
    family: str
    params: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Revalidate and expand the patient deterioration model search on CPU.",
    )
    parser.add_argument("--train-path", default="dataset/train.csv")
    parser.add_argument("--output-dir", default="artifacts/model_search_revalidated_20260326")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--holdout-size", type=float, default=0.2)
    parser.add_argument("--screen-val-size", type=float, default=0.12)
    parser.add_argument("--screen-repeats", type=int, default=2)
    parser.add_argument("--screen-episode-fraction", type=float, default=0.5)
    parser.add_argument("--random-configs", type=int, default=8)
    parser.add_argument("--top-k-finalists", type=int, default=5)
    parser.add_argument("--ensemble-top-k", type=int, default=3)
    parser.add_argument("--final-val-size", type=float, default=0.1)
    parser.add_argument("--thread-count", type=int, default=-1)
    return parser


def _safe_float(value: float | np.floating) -> float:
    return round(float(value), 6)


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True)


def split_outer_holdout(
    frame: pd.DataFrame,
    holdout_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=holdout_size, random_state=random_state)
    train_idx, holdout_idx = next(
        splitter.split(frame, frame[TARGET_COLUMN], frame[EPISODE_COLUMN]),
    )
    return frame.iloc[train_idx].copy(), frame.iloc[holdout_idx].copy()


def sample_episode_subset(
    frame: pd.DataFrame,
    fraction: float,
    random_state: int,
) -> pd.DataFrame:
    if fraction >= 0.999:
        return frame.copy()

    rng = np.random.default_rng(random_state)
    episodes = frame[EPISODE_COLUMN].drop_duplicates().to_numpy()
    sample_count = max(2, int(round(len(episodes) * fraction)))
    chosen = set(rng.choice(episodes, size=sample_count, replace=False).tolist())
    return frame.loc[frame[EPISODE_COLUMN].isin(chosen)].copy()


def split_fit_eval(
    frame: pd.DataFrame,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    fit_idx, val_idx = next(splitter.split(frame, frame[TARGET_COLUMN], frame[EPISODE_COLUMN]))
    return frame.iloc[fit_idx].copy(), frame.iloc[val_idx].copy()


def build_catboost_params(
    raw_params: dict[str, Any],
    random_state: int,
    thread_count: int,
) -> dict[str, Any]:
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "auto_class_weights": "Balanced",
        "random_seed": random_state,
        "task_type": "CPU",
        "verbose": False,
        "od_type": "Iter",
        "od_wait": 60,
        "allow_writing_files": False,
        **raw_params,
    }
    if thread_count > 0:
        params["thread_count"] = thread_count
    return params


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier_score": float(brier_score_loss(y_true, scores)),
    }


def fit_catboost_once(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_columns: list[str],
    params: dict[str, Any],
) -> tuple[CatBoostClassifier, np.ndarray, np.ndarray]:
    model = CatBoostClassifier(**params)
    model.fit(
        train_frame[feature_columns],
        train_frame[TARGET_COLUMN],
        cat_features=list(CATEGORICAL_FEATURE_COLUMNS),
        eval_set=(eval_frame[feature_columns], eval_frame[TARGET_COLUMN]),
        use_best_model=True,
    )
    train_scores = model.predict_proba(train_frame[feature_columns])[:, 1]
    eval_scores = model.predict_proba(eval_frame[feature_columns])[:, 1]
    return model, train_scores, eval_scores


def random_candidate_params(rng: np.random.Generator) -> dict[str, Any]:
    bootstrap_type = rng.choice(["Bayesian", "Bernoulli", "MVS"]).item()
    params: dict[str, Any] = {
        "iterations": int(rng.integers(900, 2201)),
        "depth": int(rng.integers(5, 10)),
        "learning_rate": float(10 ** rng.uniform(math.log10(0.02), math.log10(0.08))),
        "l2_leaf_reg": float(rng.uniform(4.0, 14.0)),
        "random_strength": float(rng.uniform(0.1, 1.8)),
        "bootstrap_type": bootstrap_type,
        "border_count": int(rng.choice([128, 254])),
        "rsm": float(rng.uniform(0.7, 1.0)),
    }
    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = float(rng.uniform(0.2, 1.8))
    else:
        params["subsample"] = float(rng.uniform(0.72, 0.95))
    return params


def build_candidates(random_configs: int, random_state: int) -> list[Candidate]:
    candidates = [Candidate("repo_train_model_baseline", "repo_baseline", REPO_BASELINE_PARAMS)]
    candidates.extend(Candidate(row["name"], "repo_existing_search", row["params"]) for row in REPO_GPU_CANDIDATES)
    candidates.extend(Candidate(row["name"], "manual_expanded", row["params"]) for row in MANUAL_CATBOOST_CANDIDATES)

    rng = np.random.default_rng(random_state)
    seen = {stable_json(candidate.params) for candidate in candidates}
    random_index = 0
    while random_index < random_configs:
        params = random_candidate_params(rng)
        key = stable_json(params)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(Candidate(f"random_catboost_{random_index:02d}", "random_search", params))
        random_index += 1
    return candidates


def evaluate_candidate_screening(
    candidate: Candidate,
    screen_frame: pd.DataFrame,
    feature_columns: list[str],
    val_size: float,
    repeats: int,
    random_state: int,
    thread_count: int,
) -> dict[str, Any]:
    train_pr_aucs: list[float] = []
    val_pr_aucs: list[float] = []
    val_roc_aucs: list[float] = []
    val_briers: list[float] = []
    best_iterations: list[int] = []

    for repeat in range(repeats):
        split_seed = random_state + repeat
        fit_frame, val_frame = split_fit_eval(screen_frame, val_size=val_size, random_state=split_seed)
        params = build_catboost_params(candidate.params, random_state=split_seed, thread_count=thread_count)
        model, train_scores, val_scores = fit_catboost_once(
            train_frame=fit_frame,
            eval_frame=val_frame,
            feature_columns=feature_columns,
            params=params,
        )
        train_metrics = compute_metrics(fit_frame[TARGET_COLUMN].to_numpy(dtype=int), train_scores)
        val_metrics = compute_metrics(val_frame[TARGET_COLUMN].to_numpy(dtype=int), val_scores)
        train_pr_aucs.append(train_metrics["pr_auc"])
        val_pr_aucs.append(val_metrics["pr_auc"])
        val_roc_aucs.append(val_metrics["roc_auc"])
        val_briers.append(val_metrics["brier_score"])
        best_iterations.append(max(1, int(model.get_best_iteration()) + 1))

    mean_train_pr = float(np.mean(train_pr_aucs))
    mean_val_pr = float(np.mean(val_pr_aucs))
    val_std = float(np.std(val_pr_aucs))
    gap = mean_train_pr - mean_val_pr
    rank_score = mean_val_pr - max(0.0, gap) * 0.35 - val_std * 0.02

    return {
        "model": candidate.name,
        "family": candidate.family,
        "config": stable_json(candidate.params),
        "screen_train_pr_auc_mean": _safe_float(mean_train_pr),
        "screen_val_pr_auc_mean": _safe_float(mean_val_pr),
        "screen_val_pr_auc_std": _safe_float(val_std),
        "screen_val_roc_auc_mean": _safe_float(float(np.mean(val_roc_aucs))),
        "screen_val_brier_mean": _safe_float(float(np.mean(val_briers))),
        "screen_overfit_gap": _safe_float(gap),
        "screen_rank_score": _safe_float(rank_score),
        "suggested_iterations": int(round(float(np.mean(best_iterations)))),
    }


def refit_final_candidate(
    candidate_row: dict[str, Any],
    development_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    feature_columns: list[str],
    final_val_size: float,
    random_state: int,
    thread_count: int,
    artifact_dir: Path,
) -> dict[str, Any]:
    fit_frame, val_frame = split_fit_eval(development_frame, val_size=final_val_size, random_state=random_state)
    raw_params = json.loads(candidate_row["config"])
    fit_params = build_catboost_params(raw_params, random_state=random_state, thread_count=thread_count)
    staged_model, fit_scores, val_scores = fit_catboost_once(
        train_frame=fit_frame,
        eval_frame=val_frame,
        feature_columns=feature_columns,
        params=fit_params,
    )

    final_iterations = max(50, int(staged_model.get_best_iteration()) + 1)
    final_params = build_catboost_params(
        {**raw_params, "iterations": final_iterations},
        random_state=random_state,
        thread_count=thread_count,
    )
    final_params.pop("od_type", None)
    final_params.pop("od_wait", None)
    final_model = CatBoostClassifier(**final_params)
    final_model.fit(
        development_frame[feature_columns],
        development_frame[TARGET_COLUMN],
        cat_features=list(CATEGORICAL_FEATURE_COLUMNS),
    )

    holdout_scores = final_model.predict_proba(holdout_frame[feature_columns])[:, 1]
    holdout_metrics = compute_metrics(holdout_frame[TARGET_COLUMN].to_numpy(dtype=int), holdout_scores)

    holdout_path = artifact_dir / f"{candidate_row['model']}_holdout_predictions.csv"
    holdout_output = holdout_frame[[EPISODE_COLUMN, HOUR_COLUMN, TARGET_COLUMN]].copy()
    holdout_output["risk_score"] = holdout_scores
    holdout_output.to_csv(holdout_path, index=False)

    calib_path = artifact_dir / f"{candidate_row['model']}_calibration_predictions.csv"
    calib_output = val_frame[[EPISODE_COLUMN, HOUR_COLUMN, TARGET_COLUMN]].copy()
    calib_output["risk_score"] = val_scores
    calib_output.to_csv(calib_path, index=False)

    model_path = artifact_dir / f"{candidate_row['model']}.cbm"
    final_model.save_model(str(model_path))

    fit_metrics = compute_metrics(fit_frame[TARGET_COLUMN].to_numpy(dtype=int), fit_scores)
    calib_metrics = compute_metrics(val_frame[TARGET_COLUMN].to_numpy(dtype=int), val_scores)

    return {
        **candidate_row,
        "final_iterations": final_iterations,
        "refit_train_pr_auc": _safe_float(fit_metrics["pr_auc"]),
        "calibration_pr_auc": _safe_float(calib_metrics["pr_auc"]),
        "calibration_roc_auc": _safe_float(calib_metrics["roc_auc"]),
        "calibration_brier": _safe_float(calib_metrics["brier_score"]),
        "refit_overfit_gap": _safe_float(fit_metrics["pr_auc"] - calib_metrics["pr_auc"]),
        "holdout_pr_auc": _safe_float(holdout_metrics["pr_auc"]),
        "holdout_roc_auc": _safe_float(holdout_metrics["roc_auc"]),
        "holdout_brier": _safe_float(holdout_metrics["brier_score"]),
        "holdout_prediction_path": str(holdout_path.as_posix()),
        "calibration_prediction_path": str(calib_path.as_posix()),
        "checkpoint_path": str(model_path.as_posix()),
        "score_column": "risk_score",
    }


def merge_predictions(
    frames: list[tuple[str, Path]],
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for model_name, path in frames:
        frame = pd.read_csv(path)
        current = frame[[EPISODE_COLUMN, HOUR_COLUMN, TARGET_COLUMN, "risk_score"]].rename(
            columns={"risk_score": model_name},
        )
        if merged is None:
            merged = current
        else:
            merged = merged.merge(
                current[[EPISODE_COLUMN, HOUR_COLUMN, model_name]],
                on=[EPISODE_COLUMN, HOUR_COLUMN],
                how="inner",
            )
    if merged is None:
        raise ValueError("No prediction frames were provided.")
    return merged


def search_weighted_average(
    calibration_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    model_names: list[str],
) -> dict[str, Any]:
    y_cal = calibration_frame[TARGET_COLUMN].to_numpy(dtype=int)
    y_holdout = holdout_frame[TARGET_COLUMN].to_numpy(dtype=int)
    best: dict[str, Any] | None = None

    if len(model_names) == 2:
        for first_weight in np.linspace(0.05, 0.95, 19):
            weights = np.array([first_weight, 1.0 - first_weight], dtype=float)
            cal_scores = calibration_frame[model_names].to_numpy(dtype=float) @ weights
            pr_auc = average_precision_score(y_cal, cal_scores)
            if best is None or pr_auc > best["calibration_pr_auc"]:
                holdout_scores = holdout_frame[model_names].to_numpy(dtype=float) @ weights
                metrics = compute_metrics(y_holdout, holdout_scores)
                best = {
                    "weights": {name: _safe_float(weight) for name, weight in zip(model_names, weights, strict=True)},
                    "calibration_pr_auc": float(pr_auc),
                    "holdout_pr_auc": float(metrics["pr_auc"]),
                    "holdout_roc_auc": float(metrics["roc_auc"]),
                    "holdout_brier": float(metrics["brier_score"]),
                    "holdout_scores": holdout_scores,
                }
        if best is None:
            raise ValueError("No ensemble result produced for pairwise blend.")
        return best

    if len(model_names) == 3:
        for first_weight in np.linspace(0.5, 0.95, 10):
            remainder = 1.0 - first_weight
            for second_share in np.linspace(0.0, 1.0, 21):
                second_weight = remainder * second_share
                third_weight = remainder - second_weight
                weights = np.array([first_weight, second_weight, third_weight], dtype=float)
                cal_scores = calibration_frame[model_names].to_numpy(dtype=float) @ weights
                pr_auc = average_precision_score(y_cal, cal_scores)
                if best is None or pr_auc > best["calibration_pr_auc"]:
                    holdout_scores = holdout_frame[model_names].to_numpy(dtype=float) @ weights
                    metrics = compute_metrics(y_holdout, holdout_scores)
                    best = {
                        "weights": {
                            name: _safe_float(weight) for name, weight in zip(model_names, weights, strict=True)
                        },
                        "calibration_pr_auc": float(pr_auc),
                        "holdout_pr_auc": float(metrics["pr_auc"]),
                        "holdout_roc_auc": float(metrics["roc_auc"]),
                        "holdout_brier": float(metrics["brier_score"]),
                        "holdout_scores": holdout_scores,
                    }
        if best is None:
            raise ValueError("No ensemble result produced for triplet blend.")
        return best

    raise ValueError("Only 2-model and 3-model weighted averages are supported.")


def search_logistic_stack(
    calibration_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    model_names: list[str],
) -> dict[str, Any]:
    stacker = LogisticRegression(
        C=0.5,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    X_cal = calibration_frame[model_names].to_numpy(dtype=float)
    y_cal = calibration_frame[TARGET_COLUMN].to_numpy(dtype=int)
    X_holdout = holdout_frame[model_names].to_numpy(dtype=float)
    y_holdout = holdout_frame[TARGET_COLUMN].to_numpy(dtype=int)
    stacker.fit(X_cal, y_cal)
    cal_scores = stacker.predict_proba(X_cal)[:, 1]
    holdout_scores = stacker.predict_proba(X_holdout)[:, 1]
    holdout_metrics = compute_metrics(y_holdout, holdout_scores)
    return {
        "weights": {
            name: _safe_float(weight) for name, weight in zip(model_names, stacker.coef_[0].tolist(), strict=True)
        },
        "intercept": _safe_float(float(stacker.intercept_[0])),
        "calibration_pr_auc": _safe_float(float(average_precision_score(y_cal, cal_scores))),
        "holdout_pr_auc": _safe_float(holdout_metrics["pr_auc"]),
        "holdout_roc_auc": _safe_float(holdout_metrics["roc_auc"]),
        "holdout_brier": _safe_float(holdout_metrics["brier_score"]),
        "holdout_scores": holdout_scores,
    }


def save_ensemble_predictions(
    name: str,
    scores: np.ndarray,
    base_frame: pd.DataFrame,
    artifact_dir: Path,
) -> str:
    output = base_frame[[EPISODE_COLUMN, HOUR_COLUMN, TARGET_COLUMN]].copy()
    output[name] = scores
    path = artifact_dir / f"{name}_holdout_predictions.csv"
    output.to_csv(path, index=False)
    return str(path.as_posix())


def load_existing_repo_comparison() -> pd.DataFrame:
    model_search_path = ROOT / "artifacts" / "model_search" / "best_model_metric_details.json"
    deep_path = ROOT / "artifacts" / "deep_models" / "model_comparison.csv"
    rows: list[dict[str, Any]] = []

    if model_search_path.exists():
        saved_rows = json.loads(model_search_path.read_text(encoding="utf-8").replace("NaN", "null"))
        for row in saved_rows:
            rows.append(
                {
                    "model": row["model"],
                    "family": f"repo_saved::{row['family']}",
                    "holdout_pr_auc": row["pr_auc"],
                    "holdout_roc_auc": row["roc_auc"],
                    "holdout_brier": row["brier_score"],
                    "source": "saved_repo_artifacts",
                },
            )

    if deep_path.exists():
        deep_rows = pd.read_csv(deep_path)
        for row in deep_rows.to_dict(orient="records"):
            rows.append(
                {
                    "model": row["model"],
                    "family": f"repo_saved::{row['family']}",
                    "holdout_pr_auc": row["pr_auc"],
                    "holdout_roc_auc": row["roc_auc"],
                    "holdout_brier": row["brier_score"],
                    "source": "saved_repo_artifacts",
                },
            )

    return pd.DataFrame(rows)


def choose_finalists(screening_df: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_rows(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            name = str(row["model"])
            if name in seen:
                continue
            seen.add(name)
            chosen.append(row)

    add_rows(screening_df.head(top_k).to_dict(orient="records"))
    mandatory_rows = screening_df.loc[screening_df["model"].isin(MANDATORY_FINALIST_NAMES)]
    add_rows(mandatory_rows.to_dict(orient="records"))
    high_signal_rows = screening_df.sort_values(["screen_val_pr_auc_mean"], ascending=False).head(top_k)
    add_rows(high_signal_rows.to_dict(orient="records"))
    return chosen


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_train = pd.read_csv(ROOT / args.train_path)
    featured = engineer_features(add_episode_ids(raw_train))
    feature_columns = [column for column in featured.columns if column not in (TARGET_COLUMN, EPISODE_COLUMN)]

    development_frame, holdout_frame = split_outer_holdout(
        featured,
        holdout_size=args.holdout_size,
        random_state=args.random_state,
    )
    screen_frame = sample_episode_subset(
        development_frame,
        fraction=args.screen_episode_fraction,
        random_state=args.random_state,
    )

    candidates = build_candidates(random_configs=args.random_configs, random_state=args.random_state)
    screening_rows = []
    for index, candidate in enumerate(candidates, start=1):
        print(f"[screen {index}/{len(candidates)}] {candidate.name}")
        row = evaluate_candidate_screening(
            candidate=candidate,
            screen_frame=screen_frame,
            feature_columns=feature_columns,
            val_size=args.screen_val_size,
            repeats=args.screen_repeats,
            random_state=args.random_state,
            thread_count=args.thread_count,
        )
        screening_rows.append(row)

    screening_df = pd.DataFrame(screening_rows).sort_values(
        ["screen_rank_score", "screen_val_pr_auc_mean"],
        ascending=False,
    )
    screening_df.to_csv(output_dir / "screening_results.csv", index=False)
    screening_df.to_json(output_dir / "screening_results.json", orient="records", indent=2)

    finalists = choose_finalists(screening_df, top_k=args.top_k_finalists)
    final_rows = []
    for finalist in finalists:
        print(f"[finalist] refitting {finalist['model']}")
        final_rows.append(
            refit_final_candidate(
                candidate_row=finalist,
                development_frame=development_frame,
                holdout_frame=holdout_frame,
                feature_columns=feature_columns,
                final_val_size=args.final_val_size,
                random_state=args.random_state,
                thread_count=args.thread_count,
                artifact_dir=output_dir,
            ),
        )

    final_df = pd.DataFrame(final_rows).sort_values(
        ["holdout_pr_auc", "holdout_roc_auc"],
        ascending=False,
    )
    final_df.to_csv(output_dir / "finalist_holdout_results.csv", index=False)
    final_df.to_json(output_dir / "finalist_holdout_results.json", orient="records", indent=2)

    ensemble_rows: list[dict[str, Any]] = []
    ensemble_top = final_df.head(args.ensemble_top_k)
    if len(ensemble_top) >= 2:
        prediction_inputs = [
            (row["model"], Path(row["calibration_prediction_path"])) for row in ensemble_top.to_dict(orient="records")
        ]
        calibration_merged = merge_predictions(prediction_inputs)
        holdout_merged = merge_predictions(
            [(row["model"], Path(row["holdout_prediction_path"])) for row in ensemble_top.to_dict(orient="records")],
        )
        model_names = ensemble_top["model"].tolist()

        top2 = model_names[:2]
        pair_result = search_weighted_average(calibration_merged, holdout_merged, top2)
        pair_name = f"{top2[0]}_plus_{top2[1]}"
        pair_path = save_ensemble_predictions(pair_name, pair_result["holdout_scores"], holdout_merged, output_dir)
        ensemble_rows.append(
            {
                "model": pair_name,
                "family": "catboost_weighted_average",
                "holdout_pr_auc": _safe_float(pair_result["holdout_pr_auc"]),
                "holdout_roc_auc": _safe_float(pair_result["holdout_roc_auc"]),
                "holdout_brier": _safe_float(pair_result["holdout_brier"]),
                "calibration_pr_auc": _safe_float(pair_result["calibration_pr_auc"]),
                "config": stable_json(pair_result["weights"]),
                "holdout_prediction_path": pair_path,
            },
        )

        if len(model_names) >= 3:
            top3 = model_names[:3]
            triplet_result = search_weighted_average(calibration_merged, holdout_merged, top3)
            triplet_name = f"{top3[0]}_plus_top3"
            triplet_path = save_ensemble_predictions(
                triplet_name,
                triplet_result["holdout_scores"],
                holdout_merged,
                output_dir,
            )
            ensemble_rows.append(
                {
                    "model": triplet_name,
                    "family": "catboost_weighted_average",
                    "holdout_pr_auc": _safe_float(triplet_result["holdout_pr_auc"]),
                    "holdout_roc_auc": _safe_float(triplet_result["holdout_roc_auc"]),
                    "holdout_brier": _safe_float(triplet_result["holdout_brier"]),
                    "calibration_pr_auc": _safe_float(triplet_result["calibration_pr_auc"]),
                    "config": stable_json(triplet_result["weights"]),
                    "holdout_prediction_path": triplet_path,
                },
            )

            stack_result = search_logistic_stack(calibration_merged, holdout_merged, top3)
            stack_name = f"{top3[0]}_logistic_stack"
            stack_path = save_ensemble_predictions(
                stack_name,
                stack_result["holdout_scores"],
                holdout_merged,
                output_dir,
            )
            ensemble_rows.append(
                {
                    "model": stack_name,
                    "family": "catboost_logistic_stack",
                    "holdout_pr_auc": _safe_float(stack_result["holdout_pr_auc"]),
                    "holdout_roc_auc": _safe_float(stack_result["holdout_roc_auc"]),
                    "holdout_brier": _safe_float(stack_result["holdout_brier"]),
                    "calibration_pr_auc": _safe_float(stack_result["calibration_pr_auc"]),
                    "config": stable_json(
                        {
                            "weights": stack_result["weights"],
                            "intercept": stack_result["intercept"],
                        },
                    ),
                    "holdout_prediction_path": stack_path,
                },
            )

    ensemble_df = pd.DataFrame(ensemble_rows).sort_values(
        ["holdout_pr_auc", "holdout_roc_auc"],
        ascending=False,
    ) if ensemble_rows else pd.DataFrame()
    ensemble_df.to_csv(output_dir / "ensemble_results.csv", index=False)
    ensemble_df.to_json(output_dir / "ensemble_results.json", orient="records", indent=2)

    repo_comparison_df = load_existing_repo_comparison()
    current_rows = [
        {
            "model": row["model"],
            "family": row["family"],
            "holdout_pr_auc": row["holdout_pr_auc"],
            "holdout_roc_auc": row["holdout_roc_auc"],
            "holdout_brier": row["holdout_brier"],
            "source": "revalidated_cpu_search",
        }
        for row in final_df.to_dict(orient="records")
    ]
    current_rows.extend(ensemble_df.assign(source="revalidated_cpu_search").to_dict(orient="records"))
    combined = pd.concat(
        [pd.DataFrame(current_rows), repo_comparison_df],
        ignore_index=True,
        sort=False,
    ).sort_values(["holdout_pr_auc", "holdout_roc_auc"], ascending=False)
    combined.to_csv(output_dir / "combined_comparison.csv", index=False)
    combined.to_json(output_dir / "combined_comparison.json", orient="records", indent=2)

    best_row = combined.iloc[0].to_dict()
    summary = {
        "search_runtime_environment": {
            "device": "cpu",
            "thread_count": args.thread_count,
        },
        "dataset_summary": {
            "rows": int(len(featured)),
            "episodes": int(featured[EPISODE_COLUMN].nunique()),
            "positive_rate": _safe_float(float(featured[TARGET_COLUMN].mean())),
            "development_rows": int(len(development_frame)),
            "holdout_rows": int(len(holdout_frame)),
            "development_episodes": int(development_frame[EPISODE_COLUMN].nunique()),
            "holdout_episodes": int(holdout_frame[EPISODE_COLUMN].nunique()),
        },
        "screening": {
            "candidate_count": len(candidates),
            "screen_repeats": args.screen_repeats,
            "screen_episode_fraction": args.screen_episode_fraction,
        },
        "best_observed_model": {
            "model": best_row["model"],
            "family": best_row["family"],
            "source": best_row["source"],
            "holdout_pr_auc": _safe_float(float(best_row["holdout_pr_auc"])),
            "holdout_roc_auc": _safe_float(float(best_row["holdout_roc_auc"])),
            "holdout_brier": _safe_float(float(best_row["holdout_brier"])),
        },
        "best_revalidated_single_model": final_df.iloc[0].to_dict() if not final_df.empty else None,
        "best_revalidated_ensemble": ensemble_df.iloc[0].to_dict() if not ensemble_df.empty else None,
        "note": (
            "This search reduces overfitting risk with repeated group-aware screening and an untouched outer holdout, "
            "but it cannot prove a true global optimum or guarantee zero overfitting."
        ),
    }
    (output_dir / "search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nTop combined models:")
    print(combined[["model", "family", "source", "holdout_pr_auc", "holdout_roc_auc", "holdout_brier"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physio_warning.deep_learning import (  # noqa: E402
    SequenceWindowDataset,
    build_episode_store,
    build_loader,
    build_model,
    compute_binary_metrics,
    fit_sequence_preprocessor,
    prepare_sequence_frame,
    resolve_device,
    save_preprocessor,
    seed_everything,
    train_sequence_model,
)
from physio_warning.features import EPISODE_COLUMN, HOUR_COLUMN, TARGET_COLUMN, load_metadata  # noqa: E402

MODEL_SPECS = (
    {
        "name": "tcn",
        "builder": "tcn",
    },
    {
        "name": "gru_attention",
        "builder": "gru_attention",
    },
    {
        "name": "transformer_encoder",
        "builder": "transformer_encoder",
    },
    {
        "name": "transformer_encoder_wide",
        "builder": "transformer_encoder",
        "config": {
            "d_model": 128,
            "num_heads": 8,
            "num_layers": 3,
            "dropout": 0.10,
        },
        "max_seq_len": 36,
        "batch_size": 192,
        "epochs": 12,
        "patience": 3,
        "learning_rate": 0.0005,
        "weight_decay": 0.00005,
        "scheduler_name": "cosine",
    },
    {
        "name": "transformer_encoder_long",
        "builder": "transformer_encoder",
        "config": {
            "d_model": 160,
            "num_heads": 8,
            "num_layers": 4,
            "dropout": 0.10,
        },
        "max_seq_len": 48,
        "batch_size": 128,
        "epochs": 14,
        "patience": 4,
        "learning_rate": 0.00035,
        "weight_decay": 0.00005,
        "scheduler_name": "cosine",
    },
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and compare GPU-ready deep sequence models for physiological deterioration.",
    )
    parser.add_argument("--train-path", default="dataset/train.csv")
    parser.add_argument("--artifact-dir", default="artifacts/deep_models")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def _safe_round(value: float) -> float:
    return round(float(value), 6)


def split_frame(
    prepared_frame: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = prepared_frame[EPISODE_COLUMN]
    labels = prepared_frame[TARGET_COLUMN]

    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_val_index, holdout_index = next(outer_split.split(prepared_frame, labels, groups))

    train_val_frame = prepared_frame.iloc[train_val_index].copy()
    holdout_frame = prepared_frame.iloc[holdout_index].copy()

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


def load_catboost_holdout() -> tuple[pd.DataFrame | None, dict | None]:
    catboost_predictions_path = Path("artifacts/holdout_predictions.csv")
    catboost_metadata_path = Path("artifacts/metadata.json")
    if not catboost_predictions_path.exists() or not catboost_metadata_path.exists():
        return None, None

    predictions = pd.read_csv(catboost_predictions_path)
    metadata = load_metadata(catboost_metadata_path)
    return predictions, metadata


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
    base_metrics = compute_binary_metrics(y_true, scores)
    threshold_detail = threshold_metrics(y_true, scores)
    return {
        "roc_auc": _safe_round(base_metrics["roc_auc"]),
        "pr_auc": _safe_round(base_metrics["pr_auc"]),
        "brier_score": _safe_round(base_metrics["brier_score"]),
        "holdout_positive_rate": _safe_round(float(np.mean(y_true))),
        **{key: _safe_round(value) for key, value in threshold_detail.items()},
    }


def load_prediction_scores(path: str | Path, score_column: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    return frame[TARGET_COLUMN].to_numpy(dtype=int), frame[score_column].to_numpy(dtype=float)


def write_metric_summary(
    comparison_df: pd.DataFrame,
    artifact_dir: Path,
) -> None:
    detailed_rows: list[dict[str, object]] = []
    markdown_lines = [
        "# Detailed Model Metrics",
        "",
        "Thresholds below are computed from each model's own holdout precision-recall curve.",
        "",
    ]

    for row in comparison_df.to_dict(orient="records"):
        y_true, scores = load_prediction_scores(row["holdout_prediction_path"], row["score_column"])
        metrics = compute_detailed_metrics(y_true, scores)
        detail_row = {**row, **metrics}
        detailed_rows.append(detail_row)

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
        if pd.notna(row.get("best_epoch")):
            markdown_lines.append(f"- Best epoch: `{int(row['best_epoch'])}`")
        if pd.notna(row.get("validation_pr_auc")):
            markdown_lines.append(f"- Validation PR-AUC: `{float(row['validation_pr_auc']):.4f}`")
        if row.get("config"):
            markdown_lines.append(f"- Config: `{row['config']}`")
        markdown_lines.append("")

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df.to_csv(artifact_dir / "model_metric_details.csv", index=False)
    (artifact_dir / "model_metric_details.json").write_text(
        json.dumps(detailed_rows, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "model_metric_summary.md").write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.random_state)
    device = resolve_device(args.device)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    prepared_frame = prepare_sequence_frame(pd.read_csv(args.train_path))
    train_frame, val_frame, holdout_frame = split_frame(
        prepared_frame=prepared_frame,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    preprocessor = fit_sequence_preprocessor(train_frame)
    save_preprocessor(artifact_dir / "sequence_preprocessor.json", preprocessor)

    train_store = build_episode_store(train_frame, preprocessor)
    val_store = build_episode_store(val_frame, preprocessor)
    holdout_store = build_episode_store(holdout_frame, preprocessor)

    example_payload = next(iter(train_store.values()))
    input_dim = int(example_payload["dynamic"].shape[1])
    static_dim = int(example_payload["static"].shape[0])
    positive_count = max(float(train_frame[TARGET_COLUMN].sum()), 1.0)
    negative_count = max(float((train_frame[TARGET_COLUMN] == 0).sum()), 1.0)
    pos_weight = negative_count / positive_count

    split_manifest = {
        "device": str(device),
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_frame)),
        "holdout_rows": int(len(holdout_frame)),
        "train_episodes": int(train_frame[EPISODE_COLUMN].nunique()),
        "val_episodes": int(val_frame[EPISODE_COLUMN].nunique()),
        "holdout_episodes": int(holdout_frame[EPISODE_COLUMN].nunique()),
        "default_max_seq_len": args.max_seq_len,
        "default_batch_size": args.batch_size,
        "pos_weight": _safe_round(pos_weight),
    }
    (artifact_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    loader_cache: dict[tuple[int, int], tuple[object, object, object]] = {}

    def get_loaders(max_seq_len: int, batch_size: int):
        cache_key = (max_seq_len, batch_size)
        if cache_key not in loader_cache:
            train_dataset = SequenceWindowDataset(train_store, max_seq_len=max_seq_len)
            val_dataset = SequenceWindowDataset(val_store, max_seq_len=max_seq_len)
            holdout_dataset = SequenceWindowDataset(holdout_store, max_seq_len=max_seq_len)
            loader_cache[cache_key] = (
                build_loader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    device=device,
                    num_workers=args.num_workers,
                ),
                build_loader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    device=device,
                    num_workers=args.num_workers,
                ),
                build_loader(
                    holdout_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    device=device,
                    num_workers=args.num_workers,
                ),
            )
        return loader_cache[cache_key]

    comparison_rows: list[dict[str, object]] = []
    deep_validation_ranking: list[tuple[str, float]] = []
    deep_model_names: list[str] = []

    for spec in MODEL_SPECS:
        model_name = spec["name"]
        builder_name = spec["builder"]
        max_seq_len = int(spec.get("max_seq_len", args.max_seq_len))
        batch_size = int(spec.get("batch_size", args.batch_size))
        learning_rate = float(spec.get("learning_rate", args.learning_rate))
        weight_decay = float(spec.get("weight_decay", args.weight_decay))
        epochs = int(spec.get("epochs", args.epochs))
        patience = int(spec.get("patience", args.patience))
        scheduler_name = spec.get("scheduler_name")

        train_loader, val_loader, holdout_loader = get_loaders(max_seq_len=max_seq_len, batch_size=batch_size)

        print(f"Training {model_name} on {device}...")
        model, config = build_model(
            model_name=builder_name,
            input_dim=input_dim,
            static_dim=static_dim,
            max_seq_len=max_seq_len,
            config_override=spec.get("config"),
        )
        training_result = train_sequence_model(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            holdout_loader=holdout_loader,
            device=device,
            output_dir=artifact_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=epochs,
            patience=patience,
            pos_weight=pos_weight,
            scheduler_name=scheduler_name,
        )

        holdout_metrics = training_result["holdout_metrics"]
        validation_pr_auc = _safe_round(training_result["best_val_metrics"]["pr_auc"])
        deep_validation_ranking.append((model_name, validation_pr_auc))
        deep_model_names.append(model_name)

        comparison_rows.append(
            {
                "model": model_name,
                "family": "deep",
                "best_epoch": int(training_result["best_epoch"]),
                "validation_pr_auc": validation_pr_auc,
                "roc_auc": _safe_round(holdout_metrics["roc_auc"]),
                "pr_auc": _safe_round(holdout_metrics["pr_auc"]),
                "brier_score": _safe_round(holdout_metrics["brier_score"]),
                "checkpoint_path": training_result["checkpoint_path"],
                "history_path": training_result["history_path"],
                "holdout_prediction_path": training_result["holdout_prediction_path"],
                "score_column": "risk_score",
                "config": json.dumps(
                    {
                        **config,
                        "max_seq_len": max_seq_len,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "epochs": epochs,
                        "patience": patience,
                        "scheduler_name": scheduler_name,
                    },
                ),
            },
        )

    catboost_predictions, catboost_metadata = load_catboost_holdout()
    if catboost_predictions is not None and catboost_metadata is not None:
        comparison_rows.append(
            {
                "model": "catboost",
                "family": "baseline",
                "best_epoch": None,
                "validation_pr_auc": None,
                "roc_auc": _safe_round(catboost_metadata["metrics"]["roc_auc"]),
                "pr_auc": _safe_round(catboost_metadata["metrics"]["pr_auc"]),
                "brier_score": _safe_round(catboost_metadata["metrics"]["brier_score"]),
                "checkpoint_path": catboost_metadata["model_path"],
                "history_path": "",
                "holdout_prediction_path": "artifacts/holdout_predictions.csv",
                "score_column": "risk_score",
                "config": json.dumps(catboost_metadata["model_params"]),
            },
        )

        merged_scores = catboost_predictions[
            [EPISODE_COLUMN, HOUR_COLUMN, TARGET_COLUMN, "risk_score"]
        ].rename(columns={"risk_score": "catboost"})

        for model_name in deep_model_names:
            deep_predictions = pd.read_csv(artifact_dir / f"{model_name}_holdout_predictions.csv")
            merged_scores = merged_scores.merge(
                deep_predictions[[EPISODE_COLUMN, HOUR_COLUMN, "risk_score"]].rename(
                    columns={"risk_score": model_name},
                ),
                on=[EPISODE_COLUMN, HOUR_COLUMN],
                how="inner",
            )

        ranked_deep_models = [name for name, _ in sorted(deep_validation_ranking, key=lambda item: item[1], reverse=True)]
        top_two_models = ranked_deep_models[:2]
        top_three_models = ranked_deep_models[:3]
        best_deep_model = ranked_deep_models[0]

        merged_scores["deep_average_all"] = merged_scores[deep_model_names].mean(axis=1)
        merged_scores["deep_average_top2"] = merged_scores[top_two_models].mean(axis=1)
        merged_scores["deep_average_top3"] = merged_scores[top_three_models].mean(axis=1)
        merged_scores["catboost_best_deep_average"] = merged_scores[["catboost", best_deep_model]].mean(axis=1)
        merged_scores["catboost_top2_deep_average"] = merged_scores[["catboost", *top_two_models]].mean(axis=1)
        merged_scores.to_csv(artifact_dir / "ensemble_holdout_predictions.csv", index=False)

        for ensemble_name in (
            "deep_average_all",
            "deep_average_top2",
            "deep_average_top3",
            "catboost_best_deep_average",
            "catboost_top2_deep_average",
        ):
            metrics = compute_binary_metrics(
                merged_scores[TARGET_COLUMN].to_numpy(),
                merged_scores[ensemble_name].to_numpy(),
            )
            comparison_rows.append(
                {
                    "model": ensemble_name,
                    "family": "ensemble",
                    "best_epoch": None,
                    "validation_pr_auc": None,
                    "roc_auc": _safe_round(metrics["roc_auc"]),
                    "pr_auc": _safe_round(metrics["pr_auc"]),
                    "brier_score": _safe_round(metrics["brier_score"]),
                    "checkpoint_path": "",
                    "history_path": "",
                    "holdout_prediction_path": str((artifact_dir / "ensemble_holdout_predictions.csv").as_posix()),
                    "score_column": ensemble_name,
                    "config": "",
                },
            )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["pr_auc", "roc_auc"],
        ascending=False,
    )
    comparison_df.to_csv(artifact_dir / "model_comparison.csv", index=False)
    write_metric_summary(comparison_df, artifact_dir)

    summary = {
        "device": str(device),
        "best_model_by_pr_auc": comparison_df.iloc[0]["model"],
        "best_model_pr_auc": _safe_round(comparison_df.iloc[0]["pr_auc"]),
        "models_compared": comparison_df["model"].tolist(),
        "top_deep_model_by_validation_pr_auc": max(deep_validation_ranking, key=lambda item: item[1])[0],
    }
    (artifact_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Deep model comparison complete.")
    print(comparison_df[["model", "family", "roc_auc", "pr_auc", "brier_score"]].to_string(index=False))


if __name__ == "__main__":
    main()

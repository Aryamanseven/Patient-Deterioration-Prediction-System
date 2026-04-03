from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(y_true) else None,
        "roc_auc": None,
        "brier_score": float(brier_score_loss(y_true, y_prob)) if len(y_true) else None,
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else None,
    }
    if len(y_true) and np.unique(y_true).size > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return out


def _stratified_indices(y_true: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    n_pos = max(1, int(round(len(pos_idx) * fraction))) if len(pos_idx) else 0
    n_neg = max(1, int(round(len(neg_idx) * fraction))) if len(neg_idx) else 0

    sampled_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False) if n_pos else np.array([], dtype=int)
    sampled_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False) if n_neg else np.array([], dtype=int)

    idx = np.concatenate([sampled_pos, sampled_neg])
    rng.shuffle(idx)
    return idx


def _load_sources() -> list[dict[str, Any]]:
    return [
        {
            "source_name": "run_20260401_095900_stage2",
            "path": PROJECT_ROOT / "artifacts" / "run_20260401_095900_876038_stage2_fl10_from_recovered_dl" / "predictions.csv",
            "label_col": "y_true",
            "score_cols": {
                "ensemble": "y_proba_ensemble",
                "catboost": "y_proba_catboost",
                "deep_learning": "y_proba_dl",
            },
        },
        {
            "source_name": "run_20260401_025010_ssl_retrain",
            "path": PROJECT_ROOT / "artifacts" / "run_20260401_025010_804106_retrain_from_ssl_strict_save" / "predictions.csv",
            "label_col": "y_true",
            "score_cols": {
                "ensemble": "y_proba_ensemble",
                "catboost": "y_proba_catboost",
                "deep_learning": "y_proba_dl",
            },
        },
        {
            "source_name": "artifacts_holdout_current",
            "path": PROJECT_ROOT / "artifacts" / "holdout_predictions.csv",
            "label_col": "deterioration_next_12h",
            "score_cols": {
                "catboost_holdout": "risk_score",
            },
        },
        {
            "source_name": "archive_focused_subsample_lr0048_iter1450",
            "path": PROJECT_ROOT
            / "archive"
            / "legacy_outputs"
            / "artifacts_model_search_revalidated_20260326"
            / "focused_subsample_lr0048_iter1450_best_holdout_predictions.csv",
            "label_col": "deterioration_next_12h",
            "score_cols": {
                "focused_subsample_lr0048_iter1450": "risk_score",
            },
        },
        {
            "source_name": "archive_catboost_gpu_subsample_train80",
            "path": PROJECT_ROOT
            / "archive"
            / "legacy_outputs"
            / "artifacts_model_search"
            / "catboost_gpu_subsample_train80_holdout_predictions.csv",
            "label_col": "deterioration_next_12h",
            "score_cols": {
                "catboost_gpu_subsample_train80": "risk_score",
            },
        },
    ]


def _evaluate_source(
    *,
    source_name: str,
    df: pd.DataFrame,
    label_col: str,
    score_cols: dict[str, str],
    fractions: list[float],
    repeats: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y_true_all = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).to_numpy()

    detailed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model_name, col in score_cols.items():
        y_prob_all = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y_prob_all = np.clip(y_prob_all, 0.0, 1.0)

        # Full-sample row (fraction = 1.0, repeat = -1)
        full_metrics = _compute_metrics(y_true_all, y_prob_all)
        detailed_rows.append(
            {
                "source": source_name,
                "model": model_name,
                "fraction": 1.0,
                "repeat": -1,
                **full_metrics,
            }
        )

        for frac in fractions:
            frac_metrics: list[dict[str, float | None]] = []
            for rep in range(repeats):
                rng = np.random.default_rng(seed + rep + int(frac * 1000))
                idx = _stratified_indices(y_true_all, frac, rng)
                y_t = y_true_all[idx]
                y_p = y_prob_all[idx]
                m = _compute_metrics(y_t, y_p)
                frac_metrics.append(m)
                detailed_rows.append(
                    {
                        "source": source_name,
                        "model": model_name,
                        "fraction": frac,
                        "repeat": rep,
                        **m,
                    }
                )

            for metric in ["pr_auc", "roc_auc", "brier_score"]:
                vals = [x[metric] for x in frac_metrics if x[metric] is not None]
                summary_rows.append(
                    {
                        "source": source_name,
                        "model": model_name,
                        "fraction": frac,
                        "metric": metric,
                        "mean": float(np.mean(vals)) if vals else None,
                        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "min": float(np.min(vals)) if vals else None,
                        "max": float(np.max(vals)) if vals else None,
                        "repeats": repeats,
                    }
                )

    return detailed_rows, summary_rows


def run_subsample_benchmark(
    *,
    output_dir: Path,
    fractions: list[float],
    repeats: int,
    seed: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = _load_sources()
    detailed_all: list[dict[str, Any]] = []
    summary_all: list[dict[str, Any]] = []
    source_status: list[dict[str, Any]] = []

    for src in sources:
        path = src["path"]
        if not path.exists():
            source_status.append({"source": src["source_name"], "path": str(path), "status": "missing"})
            continue

        try:
            df = pd.read_csv(path)
            if src["label_col"] not in df.columns:
                source_status.append(
                    {
                        "source": src["source_name"],
                        "path": str(path),
                        "status": "missing_label_column",
                        "label_col": src["label_col"],
                    }
                )
                continue

            missing_scores = [c for c in src["score_cols"].values() if c not in df.columns]
            if missing_scores:
                source_status.append(
                    {
                        "source": src["source_name"],
                        "path": str(path),
                        "status": "missing_score_columns",
                        "missing": ",".join(missing_scores),
                    }
                )
                continue

            detail_rows, summary_rows = _evaluate_source(
                source_name=src["source_name"],
                df=df,
                label_col=src["label_col"],
                score_cols=src["score_cols"],
                fractions=fractions,
                repeats=repeats,
                seed=seed,
            )
            detailed_all.extend(detail_rows)
            summary_all.extend(summary_rows)
            source_status.append({"source": src["source_name"], "path": str(path), "status": "ok", "rows": len(df)})
        except Exception as exc:
            source_status.append({"source": src["source_name"], "path": str(path), "status": f"error:{exc}"})

    detailed_df = pd.DataFrame(detailed_all)
    summary_df = pd.DataFrame(summary_all)
    status_df = pd.DataFrame(source_status)

    detailed_df.to_csv(output_dir / "subsample_metrics_detailed.csv", index=False)
    summary_df.to_csv(output_dir / "subsample_metrics_summary.csv", index=False)
    status_df.to_csv(output_dir / "source_status.csv", index=False)

    full_rows = detailed_df[detailed_df["repeat"] == -1].copy() if not detailed_df.empty else pd.DataFrame()
    if not full_rows.empty:
        full_rows.to_csv(output_dir / "full_sample_metrics.csv", index=False)

    h2h_rows: list[dict[str, Any]] = []
    if not summary_df.empty:
        target_a_source = "run_20260401_095900_stage2"
        target_a_model = "ensemble"
        target_b_source = "archive_focused_subsample_lr0048_iter1450"
        target_b_model = "focused_subsample_lr0048_iter1450"

        for frac in sorted(summary_df["fraction"].dropna().unique().tolist()):
            a_pr = summary_df[
                (summary_df["fraction"] == frac)
                & (summary_df["metric"] == "pr_auc")
                & (summary_df["source"] == target_a_source)
                & (summary_df["model"] == target_a_model)
            ]
            b_pr = summary_df[
                (summary_df["fraction"] == frac)
                & (summary_df["metric"] == "pr_auc")
                & (summary_df["source"] == target_b_source)
                & (summary_df["model"] == target_b_model)
            ]
            a_brier = summary_df[
                (summary_df["fraction"] == frac)
                & (summary_df["metric"] == "brier_score")
                & (summary_df["source"] == target_a_source)
                & (summary_df["model"] == target_a_model)
            ]
            b_brier = summary_df[
                (summary_df["fraction"] == frac)
                & (summary_df["metric"] == "brier_score")
                & (summary_df["source"] == target_b_source)
                & (summary_df["model"] == target_b_model)
            ]
            if a_pr.empty or b_pr.empty or a_brier.empty or b_brier.empty:
                continue

            h2h_rows.append(
                {
                    "fraction": frac,
                    "ensemble_pr_mean": float(a_pr.iloc[0]["mean"]),
                    "focused_pr_mean": float(b_pr.iloc[0]["mean"]),
                    "delta_pr_focused_minus_ensemble": float(b_pr.iloc[0]["mean"]) - float(a_pr.iloc[0]["mean"]),
                    "ensemble_brier_mean": float(a_brier.iloc[0]["mean"]),
                    "focused_brier_mean": float(b_brier.iloc[0]["mean"]),
                    "delta_brier_focused_minus_ensemble": float(b_brier.iloc[0]["mean"]) - float(a_brier.iloc[0]["mean"]),
                }
            )

    h2h_df = pd.DataFrame(h2h_rows)
    h2h_df.to_csv(output_dir / "head_to_head_ensemble_vs_focused.csv", index=False)

    top_rows = []
    if not summary_df.empty:
        for frac in sorted(summary_df["fraction"].dropna().unique().tolist()):
            pr_slice = summary_df[(summary_df["fraction"] == frac) & (summary_df["metric"] == "pr_auc")].dropna(subset=["mean"])
            if not pr_slice.empty:
                best_pr = pr_slice.sort_values("mean", ascending=False).iloc[0]
                top_rows.append(
                    {
                        "fraction": frac,
                        "winner_metric": "pr_auc",
                        "source": best_pr["source"],
                        "model": best_pr["model"],
                        "mean": best_pr["mean"],
                    }
                )

            brier_slice = summary_df[(summary_df["fraction"] == frac) & (summary_df["metric"] == "brier_score")].dropna(subset=["mean"])
            if not brier_slice.empty:
                best_brier = brier_slice.sort_values("mean", ascending=True).iloc[0]
                top_rows.append(
                    {
                        "fraction": frac,
                        "winner_metric": "brier_score",
                        "source": best_brier["source"],
                        "model": best_brier["model"],
                        "mean": best_brier["mean"],
                    }
                )

    pd.DataFrame(top_rows).to_csv(output_dir / "winner_by_fraction.csv", index=False)

    readme_lines = [
        "# Subsample Benchmark",
        "",
        "This package benchmarks labeled prediction artifacts using stratified subsamples.",
        "",
        "## Configuration",
        "",
        f"1. Fractions: {', '.join(f'{x:.2f}' for x in fractions)}",
        f"2. Repeats per fraction: {repeats}",
        f"3. Seed base: {seed}",
        "",
        "## Outputs",
        "",
        "1. source_status.csv",
        "2. full_sample_metrics.csv",
        "3. subsample_metrics_detailed.csv",
        "4. subsample_metrics_summary.csv",
        "5. winner_by_fraction.csv",
        "6. head_to_head_ensemble_vs_focused.csv",
    ]

    if not full_rows.empty:
        e_full = full_rows[
            (full_rows["source"] == "run_20260401_095900_stage2")
            & (full_rows["model"] == "ensemble")
        ]
        f_full = full_rows[
            (full_rows["source"] == "archive_focused_subsample_lr0048_iter1450")
            & (full_rows["model"] == "focused_subsample_lr0048_iter1450")
        ]
        if not e_full.empty and not f_full.empty:
            e_row = e_full.iloc[0]
            f_row = f_full.iloc[0]
            readme_lines.extend(
                [
                    "",
                    "## Key Full-Sample Head-to-Head",
                    "",
                    f"1. Ensemble PR-AUC: {float(e_row['pr_auc']):.6f}",
                    f"2. Focused PR-AUC: {float(f_row['pr_auc']):.6f}",
                    f"3. PR delta (Focused - Ensemble): {float(f_row['pr_auc']) - float(e_row['pr_auc']):.6f}",
                    f"4. Ensemble Brier: {float(e_row['brier_score']):.6f}",
                    f"5. Focused Brier: {float(f_row['brier_score']):.6f}",
                    f"6. Brier delta (Focused - Ensemble): {float(f_row['brier_score']) - float(e_row['brier_score']):.6f}",
                ]
            )
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "fractions": fractions,
        "repeats": repeats,
        "seed": seed,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run stratified subsample benchmarks on labeled prediction artifacts.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/subsample_benchmark_latest",
        help="Output directory for benchmark package",
    )
    parser.add_argument(
        "--fractions",
        default="0.05,0.1,0.2,0.3,0.5,0.8",
        help="Comma-separated subsample fractions in (0,1]",
    )
    parser.add_argument("--repeats", type=int, default=25, help="Repeats per fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    fractions = [float(x.strip()) for x in str(args.fractions).split(",") if x.strip()]
    fractions = [x for x in fractions if 0.0 < x <= 1.0]
    fractions = sorted(set(fractions))
    if not fractions:
        raise ValueError("No valid fractions provided")

    out_dir = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    run_subsample_benchmark(output_dir=out_dir, fractions=fractions, repeats=int(args.repeats), seed=int(args.seed))
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
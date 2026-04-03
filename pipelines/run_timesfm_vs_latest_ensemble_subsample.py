from __future__ import annotations

import argparse
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_loader import load_and_split_data
from core.features import EPISODE_COLUMN, HOUR_COLUMN, TARGET_COLUMN


TIMESFM_SRC = PROJECT_ROOT.parent / "timesfm_upstream" / "src"
if str(TIMESFM_SRC) not in sys.path:
    sys.path.insert(0, str(TIMESFM_SRC))

import timesfm  # type: ignore[reportMissingImports]  # noqa: E402


def _patch_timesfm_constructor_for_hf_kwargs() -> bool:
    """Allow `from_pretrained` to ignore hub kwargs unsupported by upstream constructor."""
    original_init = timesfm.TimesFM_2p5_200M_torch.__init__
    signature = inspect.signature(original_init)
    has_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if has_var_kwargs:
        return False

    def _patched(self, torch_compile: bool = True, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        _ = kwargs
        original_init(self, torch_compile=torch_compile, config=config)

    timesfm.TimesFM_2p5_200M_torch.__init__ = _patched
    return True


def _parse_fractions(raw: str) -> list[float]:
    values = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    values = [x for x in values if 0.0 < x <= 1.0]
    values = sorted(set(values))
    if not values:
        raise ValueError("No valid fractions provided")
    return values


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    clean_prob = np.nan_to_num(np.asarray(y_prob, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    clean_prob = np.clip(clean_prob, 0.0, 1.0)
    out: dict[str, float | None] = {
        "pr_auc": float(average_precision_score(y_true, clean_prob)) if len(y_true) else None,
        "roc_auc": None,
        "brier_score": float(brier_score_loss(y_true, clean_prob)) if len(y_true) else None,
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else None,
    }
    if len(y_true) and np.unique(y_true).size > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, clean_prob))
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


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax <= vmin:
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def _build_episode_series(featured_df: pd.DataFrame, feature_col: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    needed = [EPISODE_COLUMN, HOUR_COLUMN, feature_col]
    for col in needed:
        if col not in featured_df.columns:
            raise KeyError(f"Missing required column: {col}")

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    grouped = featured_df[needed].groupby(EPISODE_COLUMN, sort=False)

    for episode_id, group in grouped:
        sg = group.sort_values(HOUR_COLUMN)
        hours = pd.to_numeric(sg[HOUR_COLUMN], errors="coerce").to_numpy(dtype=float)
        values = pd.to_numeric(sg[feature_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(hours) & np.isfinite(values)
        if int(np.sum(mask)) < 2:
            continue
        out[int(episode_id)] = (hours[mask], values[mask])

    return out


def _context_for_row(
    *,
    hours: np.ndarray,
    values: np.ndarray,
    current_hour: float,
    max_context: int,
) -> tuple[np.ndarray, float]:
    pos = int(np.searchsorted(hours, current_hour, side="right"))
    if pos < 2:
        pos = min(2, len(values))
    hist = values[:pos]
    if hist.size > max_context:
        hist = hist[-max_context:]
    if hist.size == 0:
        hist = np.array([0.0, 0.0], dtype=float)
    elif hist.size == 1:
        hist = np.array([hist[0], hist[0]], dtype=float)
    return hist.astype(float), float(hist[-1])


def _load_eval_data(
    *,
    run_dir: Path,
    data_path: Path,
    test_size: float,
    random_state: int,
    use_advanced_features: bool,
    use_clinical_scores: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pred_path = run_dir / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    pred_df = pd.read_csv(pred_path)
    required_pred = {"y_true", "y_proba_ensemble"}
    missing_pred = sorted(required_pred - set(pred_df.columns))
    if missing_pred:
        raise ValueError(f"predictions.csv missing required columns: {missing_pred}")

    _, _, val_df, _ = load_and_split_data(
        data_path=data_path,
        test_size=test_size,
        max_rows=None,
        use_advanced_features=use_advanced_features,
        use_clinical_scores=use_clinical_scores,
        random_state=random_state,
    )

    if len(val_df) != len(pred_df):
        raise ValueError(
            f"Row count mismatch: val_df={len(val_df)} vs predictions.csv={len(pred_df)}"
        )

    y_val = pd.to_numeric(val_df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int).to_numpy()
    y_pred = pd.to_numeric(pred_df["y_true"], errors="coerce").fillna(0).astype(int).to_numpy()
    labels_match = bool(np.array_equal(y_val, y_pred))
    label_match_rate = float(np.mean(y_val == y_pred))
    if not labels_match:
        raise ValueError(
            "Validation labels do not align with predictions row order "
            f"(match_rate={label_match_rate:.6f})."
        )

    eval_df = pd.DataFrame(
        {
            EPISODE_COLUMN: pd.to_numeric(val_df[EPISODE_COLUMN], errors="coerce").fillna(-1).astype(int),
            HOUR_COLUMN: pd.to_numeric(val_df[HOUR_COLUMN], errors="coerce").fillna(0.0).astype(float),
            TARGET_COLUMN: y_val,
            "latest_ensemble_risk": np.clip(
                pd.to_numeric(pred_df["y_proba_ensemble"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                0.0,
                1.0,
            ),
        }
    )

    alignment = {
        "rows": int(len(eval_df)),
        "labels_match": labels_match,
        "label_match_rate": label_match_rate,
        "positive_rate": float(np.mean(y_val)),
        "run_predictions_path": str(pred_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    return eval_df, val_df, alignment


def _generate_timesfm_proxy_scores(
    *,
    eval_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_col: str,
    max_context: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if feature_col not in val_df.columns:
        raise ValueError(f"Feature column not found in validation data: {feature_col}")

    series_by_episode = _build_episode_series(val_df, feature_col)

    patched = _patch_timesfm_constructor_for_hf_kwargs()
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(
        timesfm.ForecastConfig(
            max_context=int(max_context),
            max_horizon=1,
            normalize_inputs=True,
            per_core_batch_size=max(1, int(batch_size)),
        )
    )

    contexts: list[np.ndarray] = []
    last_vals: list[float] = []
    missing_episode_rows = 0

    for row in eval_df.itertuples(index=False):
        episode_id = int(getattr(row, EPISODE_COLUMN))
        current_hour = float(getattr(row, HOUR_COLUMN))
        payload = series_by_episode.get(episode_id)
        if payload is None:
            missing_episode_rows += 1
            contexts.append(np.array([0.0, 0.0], dtype=float))
            last_vals.append(0.0)
            continue

        hours, values = payload
        hist, last_val = _context_for_row(
            hours=hours,
            values=values,
            current_hour=current_hour,
            max_context=max_context,
        )
        contexts.append(hist)
        last_vals.append(last_val)

    forecast_vals: list[float] = []
    for start in range(0, len(contexts), int(batch_size)):
        batch = contexts[start : start + int(batch_size)]
        point_forecast, _ = model.forecast(horizon=1, inputs=batch)
        batch_pred = np.asarray(point_forecast, dtype=float).reshape(-1)
        forecast_vals.extend(batch_pred.tolist())

    last_arr = np.asarray(last_vals, dtype=float)
    pred_arr = np.asarray(forecast_vals, dtype=float)
    pred_arr = np.where(np.isfinite(pred_arr), pred_arr, last_arr)
    last_arr = np.nan_to_num(last_arr, nan=0.0, posinf=1.0, neginf=0.0)
    raw_timesfm = 0.6 * last_arr + 0.4 * pred_arr
    timesfm_risk = np.clip(_minmax_scale(np.nan_to_num(raw_timesfm, nan=0.0, posinf=1.0, neginf=0.0)), 0.0, 1.0)

    coverage = {
        "rows": int(len(eval_df)),
        "episodes_with_series": int(len(series_by_episode)),
        "missing_episode_rows": int(missing_episode_rows),
        "timesfm_constructor_patched": bool(patched),
    }
    return timesfm_risk, coverage


def _run_subsample_eval(
    *,
    y_true: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    fractions: list[float],
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detailed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model_name, probs in scores_by_model.items():
        full_metrics = _compute_metrics(y_true, probs)
        detailed_rows.append(
            {
                "model": model_name,
                "fraction": 1.0,
                "repeat": -1,
                **full_metrics,
            }
        )

    for frac in fractions:
        for rep in range(repeats):
            rng = np.random.default_rng(seed + rep + int(frac * 1000))
            idx = _stratified_indices(y_true, frac, rng)
            y_sub = y_true[idx]
            for model_name, probs in scores_by_model.items():
                metrics = _compute_metrics(y_sub, probs[idx])
                detailed_rows.append(
                    {
                        "model": model_name,
                        "fraction": frac,
                        "repeat": rep,
                        **metrics,
                    }
                )

    detailed_df = pd.DataFrame(detailed_rows)

    if detailed_df.empty:
        return detailed_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    summary_group = detailed_df[detailed_df["repeat"] >= 0].groupby(["model", "fraction"], as_index=False)
    for _, group in summary_group:
        model_name = str(group.iloc[0]["model"])
        frac = float(group.iloc[0]["fraction"])
        for metric in ["pr_auc", "roc_auc", "brier_score"]:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            summary_rows.append(
                {
                    "model": model_name,
                    "fraction": frac,
                    "metric": metric,
                    "mean": float(np.mean(vals)) if len(vals) else None,
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "min": float(np.min(vals)) if len(vals) else None,
                    "max": float(np.max(vals)) if len(vals) else None,
                    "repeats": int(repeats),
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    h2h_rows: list[dict[str, Any]] = []
    for frac in sorted(summary_df["fraction"].dropna().unique().tolist()):
        def _slice(model: str, metric: str) -> pd.DataFrame:
            return summary_df[
                (summary_df["fraction"] == frac)
                & (summary_df["model"] == model)
                & (summary_df["metric"] == metric)
            ]

        e_pr = _slice("latest_ensemble", "pr_auc")
        t_pr = _slice("timesfm_proxy", "pr_auc")
        e_roc = _slice("latest_ensemble", "roc_auc")
        t_roc = _slice("timesfm_proxy", "roc_auc")
        e_brier = _slice("latest_ensemble", "brier_score")
        t_brier = _slice("timesfm_proxy", "brier_score")

        if e_pr.empty or t_pr.empty or e_roc.empty or t_roc.empty or e_brier.empty or t_brier.empty:
            continue

        h2h_rows.append(
            {
                "fraction": float(frac),
                "ensemble_pr_mean": float(e_pr.iloc[0]["mean"]),
                "timesfm_pr_mean": float(t_pr.iloc[0]["mean"]),
                "delta_pr_ensemble_minus_timesfm": float(e_pr.iloc[0]["mean"]) - float(t_pr.iloc[0]["mean"]),
                "ensemble_roc_mean": float(e_roc.iloc[0]["mean"]),
                "timesfm_roc_mean": float(t_roc.iloc[0]["mean"]),
                "delta_roc_ensemble_minus_timesfm": float(e_roc.iloc[0]["mean"]) - float(t_roc.iloc[0]["mean"]),
                "ensemble_brier_mean": float(e_brier.iloc[0]["mean"]),
                "timesfm_brier_mean": float(t_brier.iloc[0]["mean"]),
                "delta_brier_timesfm_minus_ensemble": float(t_brier.iloc[0]["mean"]) - float(e_brier.iloc[0]["mean"]),
            }
        )

    h2h_df = pd.DataFrame(h2h_rows)

    winner_rows: list[dict[str, Any]] = []
    for frac in sorted(summary_df["fraction"].dropna().unique().tolist()):
        pr_slice = summary_df[(summary_df["fraction"] == frac) & (summary_df["metric"] == "pr_auc")].dropna(subset=["mean"])
        if not pr_slice.empty:
            pr_best = pr_slice.sort_values("mean", ascending=False).iloc[0]
            winner_rows.append(
                {
                    "fraction": float(frac),
                    "metric": "pr_auc",
                    "winner_model": pr_best["model"],
                    "winner_mean": float(pr_best["mean"]),
                }
            )

        brier_slice = summary_df[(summary_df["fraction"] == frac) & (summary_df["metric"] == "brier_score")].dropna(subset=["mean"])
        if not brier_slice.empty:
            brier_best = brier_slice.sort_values("mean", ascending=True).iloc[0]
            winner_rows.append(
                {
                    "fraction": float(frac),
                    "metric": "brier_score",
                    "winner_model": brier_best["model"],
                    "winner_mean": float(brier_best["mean"]),
                }
            )

    winner_df = pd.DataFrame(winner_rows)
    full_df = detailed_df[detailed_df["repeat"] == -1].copy()
    return detailed_df, summary_df, h2h_df, winner_df, full_df


def run_benchmark(
    *,
    run_dir: Path,
    data_path: Path,
    feature_col: str,
    max_context: int,
    batch_size: int,
    fractions: list[float],
    repeats: int,
    seed: int,
    max_eval_rows: int | None,
    test_size: float,
    random_state: int,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "artifacts" / f"timesfm_vs_latest_ensemble_subsample_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_df, val_df, alignment = _load_eval_data(
        run_dir=run_dir,
        data_path=data_path,
        test_size=test_size,
        random_state=random_state,
        use_advanced_features=True,
        use_clinical_scores=True,
    )

    eval_sampling: dict[str, Any] = {
        "max_eval_rows": int(max_eval_rows) if max_eval_rows is not None else None,
        "used_stratified_downsample": False,
    }

    if max_eval_rows is not None and max_eval_rows > 0 and len(eval_df) > max_eval_rows:
        y_all = eval_df[TARGET_COLUMN].to_numpy(dtype=int)
        frac = float(max_eval_rows) / float(len(eval_df))
        idx = _stratified_indices(y_all, frac, np.random.default_rng(seed))
        eval_df = eval_df.iloc[idx].reset_index(drop=True)
        eval_sampling["used_stratified_downsample"] = True
        eval_sampling["rows_after_sampling"] = int(len(eval_df))
        eval_sampling["sampling_fraction"] = float(frac)
    else:
        eval_sampling["rows_after_sampling"] = int(len(eval_df))
        eval_sampling["sampling_fraction"] = 1.0

    timesfm_scores, coverage = _generate_timesfm_proxy_scores(
        eval_df=eval_df,
        val_df=val_df,
        feature_col=feature_col,
        max_context=max_context,
        batch_size=batch_size,
    )

    y_true = eval_df[TARGET_COLUMN].to_numpy(dtype=int)
    ensemble_scores = np.clip(eval_df["latest_ensemble_risk"].to_numpy(dtype=float), 0.0, 1.0)

    scores_by_model = {
        "latest_ensemble": ensemble_scores,
        "timesfm_proxy": timesfm_scores,
    }

    detailed_df, summary_df, h2h_df, winner_df, full_df = _run_subsample_eval(
        y_true=y_true,
        scores_by_model=scores_by_model,
        fractions=fractions,
        repeats=repeats,
        seed=seed,
    )

    out_eval = eval_df.copy()
    out_eval["timesfm_proxy_risk"] = timesfm_scores
    out_eval.to_csv(out_dir / "evaluation_rows_with_scores.csv", index=False)

    detailed_df.to_csv(out_dir / "subsample_metrics_detailed.csv", index=False)
    summary_df.to_csv(out_dir / "subsample_metrics_summary.csv", index=False)
    h2h_df.to_csv(out_dir / "head_to_head_by_fraction.csv", index=False)
    winner_df.to_csv(out_dir / "winner_by_fraction.csv", index=False)
    full_df.to_csv(out_dir / "full_sample_metrics.csv", index=False)

    full_ens = full_df[full_df["model"] == "latest_ensemble"]
    full_tf = full_df[full_df["model"] == "timesfm_proxy"]
    if full_ens.empty or full_tf.empty:
        raise RuntimeError("Missing full-sample rows for latest_ensemble or timesfm_proxy")

    e_row = full_ens.iloc[0]
    t_row = full_tf.iloc[0]

    delta_full = {
        "pr_auc_ensemble_minus_timesfm": float(e_row["pr_auc"]) - float(t_row["pr_auc"]),
        "roc_auc_ensemble_minus_timesfm": float(e_row["roc_auc"]) - float(t_row["roc_auc"]),
        "brier_timesfm_minus_ensemble": float(t_row["brier_score"]) - float(e_row["brier_score"]),
    }

    summary_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "strict_two_model_latest_ensemble_vs_timesfm_proxy",
        "inputs": {
            "run_dir": str(run_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "data_path": str(data_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "feature_col": feature_col,
            "max_context": int(max_context),
            "batch_size": int(batch_size),
            "fractions": fractions,
            "repeats": int(repeats),
            "seed": int(seed),
            "test_size": float(test_size),
            "random_state": int(random_state),
        },
        "alignment": alignment,
        "evaluation_sampling": eval_sampling,
        "timesfm_coverage": coverage,
        "full_sample_metrics": {
            "latest_ensemble": {
                "pr_auc": float(e_row["pr_auc"]),
                "roc_auc": float(e_row["roc_auc"]),
                "brier_score": float(e_row["brier_score"]),
                "n": int(e_row["n"]),
                "positive_rate": float(e_row["positive_rate"]),
            },
            "timesfm_proxy": {
                "pr_auc": float(t_row["pr_auc"]),
                "roc_auc": float(t_row["roc_auc"]),
                "brier_score": float(t_row["brier_score"]),
                "n": int(t_row["n"]),
                "positive_rate": float(t_row["positive_rate"]),
            },
            "delta": delta_full,
        },
        "notes": [
            "TimeSFM scores here are proxy clinical risk scores derived from one-step univariate forecasts.",
            "Latest ensemble scores are loaded from run_20260401_095900... predictions and aligned by exact row order check.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    readme_lines = [
        "# TimeSFM vs Latest Ensemble (Subsample Benchmark)",
        "",
        "Strict two-model benchmark only:",
        "1. latest_ensemble (from latest run predictions)",
        "2. timesfm_proxy (generated now)",
        "",
        "## Full-Sample Metrics (Evaluated Rows)",
        "",
        f"1. Latest Ensemble PR-AUC: {float(e_row['pr_auc']):.6f}",
        f"2. TimeSFM Proxy PR-AUC: {float(t_row['pr_auc']):.6f}",
        f"3. PR-AUC delta (Ensemble - TimeSFM): {delta_full['pr_auc_ensemble_minus_timesfm']:.6f}",
        f"4. Latest Ensemble ROC-AUC: {float(e_row['roc_auc']):.6f}",
        f"5. TimeSFM Proxy ROC-AUC: {float(t_row['roc_auc']):.6f}",
        f"6. ROC-AUC delta (Ensemble - TimeSFM): {delta_full['roc_auc_ensemble_minus_timesfm']:.6f}",
        f"7. Latest Ensemble Brier: {float(e_row['brier_score']):.6f}",
        f"8. TimeSFM Proxy Brier: {float(t_row['brier_score']):.6f}",
        f"9. Brier delta (TimeSFM - Ensemble): {delta_full['brier_timesfm_minus_ensemble']:.6f}",
        "",
        "## Evaluation Coverage",
        "",
        f"1. Total aligned rows available: {alignment['rows']}",
        f"2. Rows evaluated in this benchmark: {eval_sampling['rows_after_sampling']}",
        f"3. Stratified downsample applied: {eval_sampling['used_stratified_downsample']}",
        "",
        "## Output Files",
        "",
        "1. summary.json",
        "2. full_sample_metrics.csv",
        "3. subsample_metrics_detailed.csv",
        "4. subsample_metrics_summary.csv",
        "5. head_to_head_by_fraction.csv",
        "6. winner_by_fraction.csv",
        "7. evaluation_rows_with_scores.csv",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    pointer = {
        "generated_utc": summary_payload["generated_utc"],
        "path": str(out_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    (PROJECT_ROOT / "artifacts" / "timesfm_vs_latest_ensemble_latest.json").write_text(
        json.dumps(pointer, indent=2),
        encoding="utf-8",
    )

    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run strict latest-ensemble vs TimeSFM subsample benchmark on the same validation split."
    )
    parser.add_argument(
        "--run-dir",
        default="artifacts/run_20260401_095900_876038_stage2_fl10_from_recovered_dl",
        help="Run artifact directory containing predictions.csv for latest ensemble",
    )
    parser.add_argument("--data-path", default="dataset/train.csv", help="Training CSV used to reconstruct split")
    parser.add_argument("--feature-col", default="sepsis_risk_score", help="Univariate feature for TimeSFM forecast")
    parser.add_argument("--max-context", type=int, default=128, help="TimeSFM max context length")
    parser.add_argument("--batch-size", type=int, default=128, help="TimeSFM inference batch size")
    parser.add_argument("--fractions", default="0.05,0.1,0.2,0.3,0.5,0.8", help="Subsample fractions")
    parser.add_argument("--repeats", type=int, default=25, help="Repeats per fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=12000,
        help="Optional stratified cap on evaluated rows for faster TimeSFM inference (<=0 disables cap)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Validation split seed")
    args = parser.parse_args()

    run_dir = (PROJECT_ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    data_path = (PROJECT_ROOT / args.data_path).resolve() if not Path(args.data_path).is_absolute() else Path(args.data_path)
    fractions = _parse_fractions(str(args.fractions))

    max_eval_rows = int(args.max_eval_rows)
    if max_eval_rows <= 0:
        max_eval_rows = None

    out_dir = run_benchmark(
        run_dir=run_dir,
        data_path=data_path,
        feature_col=str(args.feature_col),
        max_context=int(args.max_context),
        batch_size=int(args.batch_size),
        fractions=fractions,
        repeats=int(args.repeats),
        seed=int(args.seed),
        max_eval_rows=max_eval_rows,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
    )
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

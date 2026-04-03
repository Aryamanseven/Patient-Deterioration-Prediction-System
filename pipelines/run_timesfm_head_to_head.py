from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMESFM_SRC = PROJECT_ROOT.parent / "timesfm_upstream" / "src"
if str(TIMESFM_SRC) not in sys.path:
    sys.path.insert(0, str(TIMESFM_SRC))

import timesfm  # noqa: E402


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_hat = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_hat,
        average="binary",
        zero_division=0,
    )
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if np.unique(y_true).size > 1 else float("nan"),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax <= vmin:
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def _load_latest_suite_path() -> Path:
    pointer = PROJECT_ROOT / "artifacts" / "benchmark_suite_latest.json"
    if not pointer.exists():
        raise FileNotFoundError(f"Missing suite pointer: {pointer}")
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    suite_rel = payload.get("suite_path")
    if not suite_rel:
        raise ValueError("benchmark_suite_latest.json missing suite_path")
    suite = PROJECT_ROOT / suite_rel
    if not suite.exists():
        raise FileNotFoundError(f"Suite path does not exist: {suite}")
    return suite


def _build_episode_series(train_df: pd.DataFrame, feature_col: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    needed = ["episode_id", "hour_from_admission", feature_col]
    for col in needed:
        if col not in train_df.columns:
            raise KeyError(f"Missing required train column: {col}")

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    grouped = train_df[needed].groupby("episode_id", sort=False)
    for episode_id, g in grouped:
        sg = g.sort_values("hour_from_admission")
        hours = pd.to_numeric(sg["hour_from_admission"], errors="coerce").to_numpy(dtype=float)
        values = pd.to_numeric(sg[feature_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(hours) & np.isfinite(values)
        if np.sum(mask) < 2:
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


def run_head_to_head(
    *,
    feature_col: str,
    max_context: int,
    batch_size: int,
) -> Path:
    artifacts_dir = PROJECT_ROOT / "artifacts"
    holdout_path = artifacts_dir / "holdout_predictions.csv"

    if not holdout_path.exists():
        raise FileNotFoundError(f"Missing holdout predictions: {holdout_path}")

    holdout_df = pd.read_csv(holdout_path)

    required_holdout = {"episode_id", "hour_from_admission", "deterioration_next_12h", "risk_score"}
    if not required_holdout.issubset(holdout_df.columns):
        missing = sorted(required_holdout - set(holdout_df.columns))
        raise ValueError(f"holdout_predictions.csv missing columns: {missing}")

    if feature_col not in holdout_df.columns:
        raise ValueError(
            f"feature_col '{feature_col}' not found in holdout_predictions.csv. "
            f"Available columns include: {list(holdout_df.columns)}"
        )

    # Build univariate trajectories directly from holdout artifact.
    series_source = holdout_df[["episode_id", "hour_from_admission", feature_col]].rename(
        columns={feature_col: "_ts_feature"}
    )
    series_by_episode = _build_episode_series(series_source.rename(columns={"_ts_feature": feature_col}), feature_col)

    model = timesfm.TimesFM_2p5_200M_torch._from_pretrained(
        model_id="google/timesfm-2.5-200m-pytorch",
        revision=None,
        cache_dir=None,
        force_download=False,
        local_files_only=False,
        token=None,
        config=None,
    )
    model.compile(timesfm.ForecastConfig(max_context=max_context, max_horizon=1, normalize_inputs=True))

    contexts: list[np.ndarray] = []
    last_vals: list[float] = []
    missing_episode_rows = 0

    for row in holdout_df.itertuples(index=False):
        episode_id = int(getattr(row, "episode_id"))
        current_hour = float(getattr(row, "hour_from_admission"))
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
    for start in range(0, len(contexts), batch_size):
        batch = contexts[start : start + batch_size]
        point_forecast, _ = model.forecast(horizon=1, inputs=batch)
        batch_pred = np.asarray(point_forecast, dtype=float).reshape(-1)
        forecast_vals.extend(batch_pred.tolist())

    last_arr = np.asarray(last_vals, dtype=float)
    pred_arr = np.asarray(forecast_vals, dtype=float)
    # Blend current state and forecast to map univariate forecasting signal to a risk score.
    raw_timesfm = 0.6 * last_arr + 0.4 * pred_arr
    timesfm_risk = _minmax_scale(raw_timesfm)

    y_true = pd.to_numeric(holdout_df["deterioration_next_12h"], errors="coerce").fillna(0).astype(int).to_numpy()
    physioguard_risk = pd.to_numeric(holdout_df["risk_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    physioguard_risk = np.clip(physioguard_risk, 0.0, 1.0)

    m_pg = _metrics(y_true, physioguard_risk)
    m_tf = _metrics(y_true, timesfm_risk)

    delta = {
        "pr_auc": float(m_pg["pr_auc"] - m_tf["pr_auc"]),
        "roc_auc": float(m_pg["roc_auc"] - m_tf["roc_auc"]),
        "brier_improvement": float(m_tf["brier_score"] - m_pg["brier_score"]),
        "f1": float(m_pg["f1"] - m_tf["f1"]),
    }

    suite_dir = _load_latest_suite_path()
    out_dir = suite_dir / "external_competitor" / "timesfm_head_to_head"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pred = holdout_df.copy()
    out_pred["timesfm_proxy_risk"] = timesfm_risk
    out_pred.to_csv(out_dir / "timesfm_holdout_predictions.csv", index=False)

    summary_payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": "physioguard_holdout_model_vs_timesfm_proxy",
        "timesfm_proxy_definition": {
            "model": "TimesFM_2p5_200M_torch",
            "loader": "_from_pretrained workaround (hub-mixin compatibility issue)",
            "forecast_feature": feature_col,
            "horizon": 1,
            "trajectory_source": "artifacts/holdout_predictions.csv grouped by episode_id and hour_from_admission",
            "risk_mapping": "risk = minmax(0.6 * last_value + 0.4 * one_step_forecast)",
        },
        "coverage": {
            "rows": int(len(holdout_df)),
            "missing_episode_rows": int(missing_episode_rows),
        },
        "metrics": {
            "physioguard_holdout_model": m_pg,
            "timesfm_proxy": m_tf,
        },
        "delta_physioguard_minus_timesfm": delta,
        "notes": [
            "This is a computed TimeSFM proxy baseline generated from available repository data.",
            "Use same-split direct TimeSFM clinical outputs for final superiority claims if available later.",
        ],
    }
    (out_dir / "head_to_head_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    pd.DataFrame(
        [
            {"model": "physioguard_holdout_model", **m_pg},
            {"model": "timesfm_proxy", **m_tf},
        ]
    ).to_csv(out_dir / "head_to_head_metrics.csv", index=False)

    pd.DataFrame([delta]).to_csv(out_dir / "head_to_head_deltas.csv", index=False)

    readme_lines = [
        "# TimeSFM Head-to-Head (Computed)",
        "",
        "## Compared Models",
        "",
        "1. physioguard_holdout_model (risk_score from artifacts/holdout_predictions.csv)",
        "2. timesfm_proxy (freshly generated from TimeSFM one-step forecasts)",
        "",
        "## Core Metrics",
        "",
        f"1. PhysioGuard PR-AUC: {m_pg['pr_auc']:.6f}",
        f"2. TimeSFM Proxy PR-AUC: {m_tf['pr_auc']:.6f}",
        f"3. PhysioGuard ROC-AUC: {m_pg['roc_auc']:.6f}",
        f"4. TimeSFM Proxy ROC-AUC: {m_tf['roc_auc']:.6f}",
        f"5. PhysioGuard Brier: {m_pg['brier_score']:.6f}",
        f"6. TimeSFM Proxy Brier: {m_tf['brier_score']:.6f}",
        "",
        "## Delta (PhysioGuard minus TimeSFM Proxy)",
        "",
        f"1. PR-AUC delta: {delta['pr_auc']:.6f}",
        f"2. ROC-AUC delta: {delta['roc_auc']:.6f}",
        f"3. Brier improvement: {delta['brier_improvement']:.6f}",
        f"4. F1 delta: {delta['f1']:.6f}",
        "",
        "## Output Files",
        "",
        "1. head_to_head_summary.json",
        "2. head_to_head_metrics.csv",
        "3. head_to_head_deltas.csv",
        "4. timesfm_holdout_predictions.csv",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    latest_pointer = {
        "generated_utc": summary_payload["generated_utc"],
        "path": str(out_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    (PROJECT_ROOT / "artifacts" / "timesfm_head_to_head_latest.json").write_text(
        json.dumps(latest_pointer, indent=2),
        encoding="utf-8",
    )

    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TimeSFM vs PhysioGuard head-to-head on holdout labels.")
    parser.add_argument("--feature-col", default="sepsis_risk_score", help="Feature used as univariate TimeSFM forecast signal")
    parser.add_argument("--max-context", type=int, default=128, help="Max context length for TimeSFM compile")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    args = parser.parse_args()

    out_dir = run_head_to_head(
        feature_col=str(args.feature_col),
        max_context=int(args.max_context),
        batch_size=int(args.batch_size),
    )
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
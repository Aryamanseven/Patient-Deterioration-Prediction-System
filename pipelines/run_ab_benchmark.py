from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def _bootstrap_ci(values: np.ndarray, alpha: float = 0.95) -> dict[str, float]:
    lo_q = (1.0 - alpha) / 2.0
    hi_q = 1.0 - lo_q
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "ci_lower": float(np.quantile(values, lo_q)),
        "ci_upper": float(np.quantile(values, hi_q)),
    }


def _run_bootstrap(
    *,
    y_true: np.ndarray,
    model_probs: dict[str, np.ndarray],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y_true)

    pr_samples: dict[str, list[float]] = {name: [] for name in model_probs}
    roc_samples: dict[str, list[float]] = {name: [] for name in model_probs}
    brier_samples: dict[str, list[float]] = {name: [] for name in model_probs}

    delta_pr_ens_vs_cat: list[float] = []
    delta_roc_ens_vs_cat: list[float] = []
    delta_brier_ens_vs_cat: list[float] = []

    delta_pr_ens_vs_dl: list[float] = []
    delta_roc_ens_vs_dl: list[float] = []
    delta_brier_ens_vs_dl: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]

        # Skip invalid draws with a single class.
        if np.unique(y_b).size < 2:
            continue

        metrics_by_model: dict[str, dict[str, float]] = {}
        for name, probs in model_probs.items():
            m = _metrics(y_b, probs[idx])
            metrics_by_model[name] = m
            pr_samples[name].append(m["pr_auc"])
            roc_samples[name].append(m["roc_auc"])
            brier_samples[name].append(m["brier_score"])

        ens = metrics_by_model["ensemble"]
        cat = metrics_by_model["catboost"]
        dl = metrics_by_model["deep_learning"]

        delta_pr_ens_vs_cat.append(ens["pr_auc"] - cat["pr_auc"])
        delta_roc_ens_vs_cat.append(ens["roc_auc"] - cat["roc_auc"])
        delta_brier_ens_vs_cat.append(cat["brier_score"] - ens["brier_score"])

        delta_pr_ens_vs_dl.append(ens["pr_auc"] - dl["pr_auc"])
        delta_roc_ens_vs_dl.append(ens["roc_auc"] - dl["roc_auc"])
        delta_brier_ens_vs_dl.append(dl["brier_score"] - ens["brier_score"])

    def summarize(values: list[float]) -> dict[str, float]:
        arr = np.array(values, dtype=float)
        return {
            **_bootstrap_ci(arr),
            "p_gt_zero": float(np.mean(arr > 0.0)),
            "samples": int(len(arr)),
        }

    return {
        "models": {
            name: {
                "pr_auc": _bootstrap_ci(np.array(pr_samples[name], dtype=float)),
                "roc_auc": _bootstrap_ci(np.array(roc_samples[name], dtype=float)),
                "brier_score": _bootstrap_ci(np.array(brier_samples[name], dtype=float)),
            }
            for name in model_probs
        },
        "deltas": {
            "ensemble_minus_catboost": {
                "pr_auc": summarize(delta_pr_ens_vs_cat),
                "roc_auc": summarize(delta_roc_ens_vs_cat),
                "brier_improvement": summarize(delta_brier_ens_vs_cat),
            },
            "ensemble_minus_deep_learning": {
                "pr_auc": summarize(delta_pr_ens_vs_dl),
                "roc_auc": summarize(delta_roc_ens_vs_dl),
                "brier_improvement": summarize(delta_brier_ens_vs_dl),
            },
        },
    }


def build_benchmark(run_dir: Path, n_bootstrap: int, seed: int) -> dict[str, Any]:
    pred_path = run_dir / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    df = pd.read_csv(pred_path)
    required = ["y_true", "y_proba_catboost", "y_proba_dl", "y_proba_ensemble"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in predictions.csv: {missing}")

    y_true = df["y_true"].to_numpy(dtype=int)
    model_probs = {
        "catboost": df["y_proba_catboost"].to_numpy(dtype=float),
        "deep_learning": df["y_proba_dl"].to_numpy(dtype=float),
        "ensemble": df["y_proba_ensemble"].to_numpy(dtype=float),
    }

    point_metrics = {name: _metrics(y_true, probs) for name, probs in model_probs.items()}
    bootstrap = _run_bootstrap(y_true=y_true, model_probs=model_probs, n_bootstrap=n_bootstrap, seed=seed)

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_dir.name,
        "run_path": str(run_dir),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "point_metrics": point_metrics,
        "bootstrap_summary": bootstrap,
        "external_reference": {
            "timesfm": {
                "version": "2.5",
                "parameters": "200M",
                "max_context": 16384,
                "source_local_readme": "modules/ssl/timesfm_upstream/README.md",
                "source_google_blog": "https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/",
                "note": "Google Research blog states pretraining corpus of 100B real-world time-points. This is not a direct clinical deterioration benchmark result.",
            }
        },
        "validation_policy": {
            "head_to_head_claim_allowed": "Only if compared on the same patient split and same binary outcome labels.",
            "current_status": "Internal A/B validated (CatBoost vs DL vs Ensemble). External TimeSFM remains reference-only until same-split clinical run is executed.",
        },
    }


def write_outputs(payload: dict[str, Any], run_dir: Path) -> None:
    run_out = run_dir / "ab_benchmark_report.json"
    latest_out = PROJECT_ROOT / "artifacts" / "ab_benchmark_latest.json"

    run_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    pm = payload["point_metrics"]
    ens = pm["ensemble"]
    cat = pm["catboost"]
    dl = pm["deep_learning"]

    lines = [
        "# A/B Benchmark Validation Report",
        "",
        f"Generated UTC: {payload['generated_utc']}",
        f"Run: {payload['run_name']}",
        "",
        "## Point Metrics",
        "",
        f"- Ensemble PR-AUC: {ens['pr_auc']:.6f}",
        f"- CatBoost PR-AUC: {cat['pr_auc']:.6f}",
        f"- Deep Learning PR-AUC: {dl['pr_auc']:.6f}",
        f"- Ensemble ROC-AUC: {ens['roc_auc']:.6f}",
        f"- CatBoost ROC-AUC: {cat['roc_auc']:.6f}",
        f"- Deep Learning ROC-AUC: {dl['roc_auc']:.6f}",
        f"- Ensemble Brier: {ens['brier_score']:.6f}",
        f"- CatBoost Brier: {cat['brier_score']:.6f}",
        f"- Deep Learning Brier: {dl['brier_score']:.6f}",
        "",
        "## Claim Guardrail",
        "",
        "- Internal model claims validated by same-split A/B testing.",
        "- External TimeSFM claims are reference-only unless run on same clinical split.",
    ]

    md_out = PROJECT_ROOT / "docs" / "AB_BENCHMARK_VALIDATION.md"
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run A/B benchmark validation for CatBoost, DL, and Ensemble outputs.")
    parser.add_argument(
        "--run-dir",
        default="artifacts/run_20260401_095900_876038_stage2_fl10_from_recovered_dl",
        help="Path to run directory containing predictions.csv",
    )
    parser.add_argument("--n-bootstrap", type=int, default=400, help="Bootstrap rounds for confidence intervals.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    run_dir = (PROJECT_ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    payload = build_benchmark(run_dir=run_dir, n_bootstrap=int(args.n_bootstrap), seed=int(args.seed))
    write_outputs(payload, run_dir)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

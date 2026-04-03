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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipelines.run_ab_benchmark import build_benchmark




def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_readme(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float | None]:
    y_hat = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_hat,
        average="binary",
        zero_division=0,
    )

    out: dict[str, float | None] = {
        "pr_auc": None,
        "roc_auc": None,
        "brier_score": None,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "n_samples": float(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else None,
    }

    if len(y_true) > 0:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        out["brier_score"] = float(brier_score_loss(y_true, y_prob))
        if np.unique(y_true).size > 1:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob))

    return out


def _flatten_delta_rows(run_name: str, deltas: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_name, pair_payload in deltas.items():
        for metric_name, stats in pair_payload.items():
            rows.append(
                {
                    "run_name": run_name,
                    "comparison": pair_name,
                    "metric": metric_name,
                    "mean": _safe_float(stats.get("mean")),
                    "ci_lower": _safe_float(stats.get("ci_lower")),
                    "ci_upper": _safe_float(stats.get("ci_upper")),
                    "p_gt_zero": _safe_float(stats.get("p_gt_zero")),
                    "samples": stats.get("samples"),
                }
            )
    return rows


def benchmark_run_predictions(
    *,
    run_dir: Path,
    out_dir: Path,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = build_benchmark(run_dir=run_dir, n_bootstrap=n_bootstrap, seed=seed)
    point = payload["point_metrics"]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "benchmark_report.json", payload)

    point_rows = []
    for model_name, metrics in point.items():
        point_rows.append(
            {
                "run_name": run_dir.name,
                "model": model_name,
                "pr_auc": metrics.get("pr_auc"),
                "roc_auc": metrics.get("roc_auc"),
                "brier_score": metrics.get("brier_score"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "n_samples": metrics.get("n_samples"),
                "positive_rate": metrics.get("positive_rate"),
            }
        )

    point_df = pd.DataFrame(point_rows)
    point_df.to_csv(out_dir / "point_metrics.csv", index=False)

    deltas = payload.get("bootstrap_summary", {}).get("deltas", {})
    delta_rows = _flatten_delta_rows(run_dir.name, deltas)
    pd.DataFrame(delta_rows).to_csv(out_dir / "bootstrap_deltas.csv", index=False)

    ens = point.get("ensemble", {})
    cat = point.get("catboost", {})
    dl = point.get("deep_learning", {})
    ens_cat_pr = deltas.get("ensemble_minus_catboost", {}).get("pr_auc", {})
    ens_cat_brier = deltas.get("ensemble_minus_catboost", {}).get("brier_improvement", {})

    readme_lines = [
        f"# Run Benchmark: {run_dir.name}",
        "",
        "## Source",
        "",
        f"1. Predictions: artifacts/{run_dir.name}/predictions.csv",
        f"2. Metrics: artifacts/{run_dir.name}/metrics.json",
        "",
        "## Point Metrics",
        "",
        f"1. Ensemble PR-AUC: {ens.get('pr_auc'):.6f}",
        f"2. CatBoost PR-AUC: {cat.get('pr_auc'):.6f}",
        f"3. Deep Learning PR-AUC: {dl.get('pr_auc'):.6f}",
        f"4. Ensemble ROC-AUC: {ens.get('roc_auc'):.6f}",
        f"5. CatBoost ROC-AUC: {cat.get('roc_auc'):.6f}",
        f"6. Deep Learning ROC-AUC: {dl.get('roc_auc'):.6f}",
        "",
        "## Bootstrap Validation",
        "",
        f"1. Ensemble minus CatBoost PR-AUC mean delta: {ens_cat_pr.get('mean', 0.0):.6f}",
        f"2. Ensemble minus CatBoost PR-AUC 95% CI: [{ens_cat_pr.get('ci_lower', 0.0):.6f}, {ens_cat_pr.get('ci_upper', 0.0):.6f}]",
        f"3. Ensemble minus CatBoost Brier improvement mean: {ens_cat_brier.get('mean', 0.0):.6f}",
        f"4. Bootstrap rounds: {payload.get('n_bootstrap')}",
        "",
        "## Output Files",
        "",
        "1. benchmark_report.json",
        "2. point_metrics.csv",
        "3. bootstrap_deltas.csv",
    ]
    _write_readme(out_dir / "README.md", readme_lines)

    summary_row = {
        "run_name": run_dir.name,
        "ensemble_pr_auc": ens.get("pr_auc"),
        "catboost_pr_auc": cat.get("pr_auc"),
        "deep_learning_pr_auc": dl.get("pr_auc"),
        "ensemble_roc_auc": ens.get("roc_auc"),
        "catboost_roc_auc": cat.get("roc_auc"),
        "deep_learning_roc_auc": dl.get("roc_auc"),
        "ensemble_brier_score": ens.get("brier_score"),
        "catboost_brier_score": cat.get("brier_score"),
        "deep_learning_brier_score": dl.get("brier_score"),
    }
    return summary_row, delta_rows


def benchmark_holdout_csv(file_path: Path, out_dir: Path) -> dict[str, Any]:
    df = pd.read_csv(file_path)
    required = {"deterioration_next_12h", "risk_score"}
    if not required.issubset(df.columns):
        raise ValueError(f"Holdout file missing required columns: {sorted(required - set(df.columns))}")

    y_true = pd.to_numeric(df["deterioration_next_12h"], errors="coerce").fillna(0).astype(int).to_numpy()
    y_prob = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    metrics = _binary_metrics(y_true, y_prob)
    payload = {
        "source_file": str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "schema": "flat_holdout_with_labels",
        "metrics": metrics,
        "row_count": int(len(df)),
        "risk_band_counts": df["risk_band"].value_counts(dropna=False).to_dict() if "risk_band" in df.columns else {},
        "predicted_alert_counts": df["predicted_alert"].value_counts(dropna=False).to_dict()
        if "predicted_alert" in df.columns
        else {},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "benchmark_report.json", payload)
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)

    readme_lines = [
        "# Holdout Benchmark",
        "",
        "## Source",
        "",
        f"1. {payload['source_file']}",
        "",
        "## Metrics",
        "",
        f"1. PR-AUC: {metrics.get('pr_auc', 0.0):.6f}",
        f"2. ROC-AUC: {metrics.get('roc_auc', 0.0):.6f}" if metrics.get("roc_auc") is not None else "2. ROC-AUC: N/A (single class)",
        f"3. Brier Score: {metrics.get('brier_score', 0.0):.6f}",
        f"4. Precision: {metrics.get('precision', 0.0):.6f}",
        f"5. Recall: {metrics.get('recall', 0.0):.6f}",
        f"6. F1: {metrics.get('f1', 0.0):.6f}",
        "",
        "## Output Files",
        "",
        "1. benchmark_report.json",
        "2. metrics.csv",
    ]
    _write_readme(out_dir / "README.md", readme_lines)
    return payload


def summarize_unlabeled_csv(file_path: Path, out_dir: Path) -> dict[str, Any]:
    df = pd.read_csv(file_path)
    risk_col = "deterioration_risk" if "deterioration_risk" in df.columns else None
    if risk_col is None:
        for col in ["risk_score", "y_proba_ensemble", "prediction", "pred"]:
            if col in df.columns:
                risk_col = col
                break

    if risk_col is None:
        raise ValueError(f"No recognized risk/probability column found in {file_path}")

    s = pd.to_numeric(df[risk_col], errors="coerce").dropna()
    payload = {
        "source_file": str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "schema": "unlabeled_prediction_distribution",
        "risk_column": risk_col,
        "row_count": int(len(df)),
        "scored_count": int(len(s)),
        "distribution": {
            "mean": float(s.mean()) if len(s) else None,
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "min": float(s.min()) if len(s) else None,
            "q25": float(s.quantile(0.25)) if len(s) else None,
            "median": float(s.quantile(0.5)) if len(s) else None,
            "q75": float(s.quantile(0.75)) if len(s) else None,
            "max": float(s.max()) if len(s) else None,
        },
        "risk_band_counts": df["risk_band"].value_counts(dropna=False).to_dict() if "risk_band" in df.columns else {},
        "predicted_alert_counts": df["predicted_alert"].value_counts(dropna=False).to_dict()
        if "predicted_alert" in df.columns
        else {},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "distribution_report.json", payload)
    pd.DataFrame([payload["distribution"]]).to_csv(out_dir / "distribution_summary.csv", index=False)

    readme_lines = [
        "# Unlabeled Prediction Distribution",
        "",
        "## Source",
        "",
        f"1. {payload['source_file']}",
        f"2. Risk column used: {risk_col}",
        "",
        "## Distribution",
        "",
        f"1. Mean: {payload['distribution'].get('mean', 0.0):.6f}" if payload["distribution"].get("mean") is not None else "1. Mean: N/A",
        f"2. Std: {payload['distribution'].get('std', 0.0):.6f}",
        f"3. Min: {payload['distribution'].get('min', 0.0):.6f}" if payload["distribution"].get("min") is not None else "3. Min: N/A",
        f"4. Median: {payload['distribution'].get('median', 0.0):.6f}" if payload["distribution"].get("median") is not None else "4. Median: N/A",
        f"5. Max: {payload['distribution'].get('max', 0.0):.6f}" if payload["distribution"].get("max") is not None else "5. Max: N/A",
        "",
        "## Output Files",
        "",
        "1. distribution_report.json",
        "2. distribution_summary.csv",
    ]
    _write_readme(out_dir / "README.md", readme_lines)
    return payload


def _write_root_readme(
    root: Path,
    *,
    generated_utc: str,
    n_bootstrap: int,
    seed: int,
    run_rows: list[dict[str, Any]],
    holdout_payload: dict[str, Any] | None,
    unlabeled_payload: dict[str, Any] | None,
    external_payload: dict[str, Any] | None,
) -> None:
    lines = [
        "# Full Benchmark Suite",
        "",
        "This folder contains the complete benchmark execution across all available artifact sources.",
        "",
        "## Execution Metadata",
        "",
        f"1. Generated UTC: {generated_utc}",
        f"2. Bootstrap rounds (run-level): {n_bootstrap}",
        f"3. Random seed: {seed}",
        "",
        "## Folder Structure",
        "",
        "1. run_level/ - per-run model-vs-model benchmark outputs with bootstrap validation",
        "2. flat_files/holdout_predictions/ - holdout-level benchmark for risk_score",
        "3. flat_files/val_predictions/ - unlabeled distribution summary",
        "4. summary/ - consolidated CSV and JSON indexes",
        "5. provenance/ - exact source files and pipeline references",
        "6. external_competitor/ - TimeSFM reference comparison and dominance strategy",
        "",
        "## Run-Level Sources Benchmarked",
        "",
    ]

    if run_rows:
        for idx, row in enumerate(run_rows, start=1):
            lines.append(
                f"{idx}. {row['run_name']} | Ensemble PR-AUC={row['ensemble_pr_auc']:.6f} | CatBoost PR-AUC={row['catboost_pr_auc']:.6f}"
            )
    else:
        lines.append("1. None")

    lines.extend(["", "## Flat File Sources", ""])
    if holdout_payload is not None:
        lines.append(
            f"1. holdout_predictions.csv scored with labels | PR-AUC={holdout_payload['metrics'].get('pr_auc', 0.0):.6f}"
        )
    else:
        lines.append("1. holdout_predictions.csv not benchmarked")

    if unlabeled_payload is not None:
        lines.append(
            f"2. val_predictions.csv distribution only | mean risk={unlabeled_payload['distribution'].get('mean', 0.0):.6f}"
        )
    else:
        lines.append("2. val_predictions.csv not summarized")

    lines.extend(
        [
            "",
            "## External Competitor Positioning",
            "",
        ]
    )

    if external_payload is not None:
        head_to_head = external_payload.get("direct_head_to_head", {})
        best_run = external_payload.get("our_best_internal_run", {})
        lines.append(f"1. Direct same-split TimeSFM head-to-head available: {head_to_head.get('available')}")
        lines.append(
            f"2. Our best internal run: {best_run.get('run_name', 'N/A')} | Ensemble PR-AUC={best_run.get('ensemble_pr_auc', 'N/A')}"
        )
        lines.append(
            f"3. TimeSFM reference: {external_payload.get('timesfm_reference', {}).get('model_version', 'N/A')} ({external_payload.get('timesfm_reference', {}).get('parameters', 'N/A')})"
        )
    else:
        lines.append("1. External competitor positioning not generated")

    lines.extend(
        [
            "",
            "## Exact Pipelines and Scripts",
            "",
            "1. Training artifact producer: pipelines/run_full_pipeline.py",
            "2. Run-level benchmark logic: pipelines/run_ab_benchmark.py",
            "3. Full benchmark suite orchestrator: pipelines/run_benchmark_suite.py",
        ]
    )
    _write_readme(root / "README.md", lines)


def _build_external_competitor_payload(
    *,
    generated_utc: str,
    run_rows: list[dict[str, Any]],
    holdout_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    best_run = None
    if run_rows:
        best_run = max(
            run_rows,
            key=lambda row: float(row.get("ensemble_pr_auc") or float("-inf")),
        )

    return {
        "generated_utc": generated_utc,
        "comparison_scope": "external_reference_without_direct_timesfm_predictions",
        "terminology_guardrail": {
            "important_note": "TimeSFM's widely cited 100B figure refers to pretraining time-points, not parameters.",
            "timesfm_parameters": "200M",
            "timesfm_pretraining_corpus": "100B time-points",
        },
        "timesfm_reference": {
            "model_version": "2.5",
            "parameters": "200M",
            "max_context": 16384,
            "local_reference_path": "../timesfm_upstream/README.md",
            "public_reference": "https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/",
        },
        "direct_head_to_head": {
            "available": False,
            "reason": "No same-split TimeSFM prediction artifact with clinical labels is currently present in repository artifacts.",
            "claim_policy": "No direct superiority claim versus TimeSFM without same-split evaluation.",
        },
        "our_best_internal_run": {
            "run_name": best_run.get("run_name") if best_run else None,
            "ensemble_pr_auc": best_run.get("ensemble_pr_auc") if best_run else None,
            "ensemble_roc_auc": best_run.get("ensemble_roc_auc") if best_run else None,
            "ensemble_brier_score": best_run.get("ensemble_brier_score") if best_run else None,
            "catboost_pr_auc": best_run.get("catboost_pr_auc") if best_run else None,
            "deep_learning_pr_auc": best_run.get("deep_learning_pr_auc") if best_run else None,
        },
        "holdout_file_evidence": {
            "available": holdout_payload is not None,
            "pr_auc": holdout_payload.get("metrics", {}).get("pr_auc") if holdout_payload else None,
            "roc_auc": holdout_payload.get("metrics", {}).get("roc_auc") if holdout_payload else None,
            "brier_score": holdout_payload.get("metrics", {}).get("brier_score") if holdout_payload else None,
        },
        "dominance_scorecard": [
            {
                "dimension": "Task-specific clinical classification evidence",
                "our_status": "Measured on current split with saved predictions and labels",
                "timesfm_status": "Not measured in this repository on same split",
                "current_edge": "ours",
            },
            {
                "dimension": "Calibration evidence (Brier)",
                "our_status": "Measured and benchmarked",
                "timesfm_status": "Not available on same split",
                "current_edge": "ours",
            },
            {
                "dimension": "Reproducibility and artifact governance",
                "our_status": "Config-driven runs, audit scripts, traceable outputs",
                "timesfm_status": "Reference model docs only in current workspace",
                "current_edge": "ours",
            },
            {
                "dimension": "Explainability and clinical operation hooks",
                "our_status": "Integrated SHAP and dashboard workflows",
                "timesfm_status": "Not integrated in current repository pipeline",
                "current_edge": "ours",
            },
            {
                "dimension": "Foundation pretraining scale",
                "our_status": "Specialized clinical pipeline",
                "timesfm_status": "Large-scale generic pretraining corpus",
                "current_edge": "timesfm",
            },
            {
                "dimension": "Direct same-split superiority proof",
                "our_status": "Not available yet",
                "timesfm_status": "Not available yet",
                "current_edge": "undetermined",
            },
        ],
        "dominance_playbook": [
            "Win on target task fitness: clinical deterioration classification, not generic forecasting.",
            "Win on calibration and actionability: optimize Brier and threshold utility for triage decisions.",
            "Win on reliability: enforce artifact completeness and reproducibility contracts.",
            "Win on explainability: tie predictions to SHAP drivers and patient-level trend context.",
            "Win on deployment economics: lower complexity, easier integration, and faster clinical iteration.",
            "When TimeSFM outputs are available, execute strict same-split head-to-head benchmark before making superiority claims.",
        ],
    }


def _write_external_competitor_docs(
    *,
    external_dir: Path,
    payload: dict[str, Any],
) -> None:
    external_dir.mkdir(parents=True, exist_ok=True)
    _write_json(external_dir / "timesfm_positioning.json", payload)

    scorecard_rows = payload.get("dominance_scorecard", [])
    pd.DataFrame(scorecard_rows).to_csv(external_dir / "dominance_scorecard.csv", index=False)

    ref = payload.get("timesfm_reference", {})
    h2h = payload.get("direct_head_to_head", {})
    best = payload.get("our_best_internal_run", {})
    holdout = payload.get("holdout_file_evidence", {})

    readme_lines = [
        "# External Competitor Comparison: TimeSFM",
        "",
        "## Scope",
        "",
        "1. This is a strategic evidence-based comparison pack using available repository artifacts.",
        "2. No same-split TimeSFM prediction artifact with labels is currently present.",
        "",
        "## TimeSFM Reference",
        "",
        f"1. Model version: {ref.get('model_version', 'N/A')}",
        f"2. Parameters: {ref.get('parameters', 'N/A')}",
        f"3. Max context: {ref.get('max_context', 'N/A')}",
        f"4. Public reference: {ref.get('public_reference', 'N/A')}",
        "5. Guardrail note: the 100B figure is pretraining time-points, not parameter count.",
        "",
        "## Direct Head-to-Head Status",
        "",
        f"1. Available: {h2h.get('available')}",
        f"2. Reason: {h2h.get('reason', 'N/A')}",
        f"3. Claim policy: {h2h.get('claim_policy', 'N/A')}",
        "",
        "## Our Best Internal Evidence",
        "",
        f"1. Run: {best.get('run_name', 'N/A')}",
        f"2. Ensemble PR-AUC: {best.get('ensemble_pr_auc', 'N/A')}",
        f"3. Ensemble ROC-AUC: {best.get('ensemble_roc_auc', 'N/A')}",
        f"4. Ensemble Brier: {best.get('ensemble_brier_score', 'N/A')}",
        f"5. Holdout PR-AUC file evidence: {holdout.get('pr_auc', 'N/A')}",
        "",
        "## Dominance Strategy",
        "",
    ]

    for i, step in enumerate(payload.get("dominance_playbook", []), start=1):
        readme_lines.append(f"{i}. {step}")

    readme_lines.extend(
        [
            "",
            "## Output Files",
            "",
            "1. timesfm_positioning.json",
            "2. dominance_scorecard.csv",
            "3. README.md",
        ]
    )
    _write_readme(external_dir / "README.md", readme_lines)


def run_suite(n_bootstrap: int, seed: int, out_name: str, invocation: str) -> Path:
    artifacts_dir = PROJECT_ROOT / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = artifacts_dir / f"{out_name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_level_dir = out_dir / "run_level"
    flat_files_dir = out_dir / "flat_files"
    summary_dir = out_dir / "summary"
    provenance_dir = out_dir / "provenance"
    external_dir = out_dir / "external_competitor"
    for p in [run_level_dir, flat_files_dir, summary_dir, provenance_dir, external_dir]:
        p.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    skipped_runs: list[dict[str, str]] = []

    run_dirs = sorted([d for d in artifacts_dir.iterdir() if d.is_dir() and d.name.startswith("run_")], key=lambda p: p.name)
    for run_dir in run_dirs:
        pred_file = run_dir / "predictions.csv"
        if not pred_file.exists():
            skipped_runs.append({"run_name": run_dir.name, "reason": "predictions.csv missing"})
            continue

        try:
            header_df = pd.read_csv(pred_file, nrows=1)
            needed = {"y_true", "y_proba_catboost", "y_proba_dl", "y_proba_ensemble"}
            if not needed.issubset(header_df.columns):
                skipped_runs.append(
                    {
                        "run_name": run_dir.name,
                        "reason": "predictions.csv schema incompatible for model-vs-model benchmark",
                    }
                )
                continue

            run_out = run_level_dir / run_dir.name
            row, drows = benchmark_run_predictions(
                run_dir=run_dir,
                out_dir=run_out,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            run_rows.append(row)
            delta_rows.extend(drows)
        except Exception as exc:
            skipped_runs.append({"run_name": run_dir.name, "reason": f"benchmark failed: {exc}"})

    holdout_payload = None
    holdout_file = artifacts_dir / "holdout_predictions.csv"
    holdout_error = None
    if holdout_file.exists():
        try:
            holdout_payload = benchmark_holdout_csv(holdout_file, flat_files_dir / "holdout_predictions")
        except Exception as exc:
            holdout_error = str(exc)

    unlabeled_payload = None
    unlabeled_file = artifacts_dir / "val_predictions.csv"
    unlabeled_error = None
    if unlabeled_file.exists():
        try:
            unlabeled_payload = summarize_unlabeled_csv(unlabeled_file, flat_files_dir / "val_predictions")
        except Exception as exc:
            unlabeled_error = str(exc)

    run_df = pd.DataFrame(run_rows)
    if not run_df.empty:
        run_df = run_df.sort_values(["ensemble_pr_auc", "catboost_pr_auc"], ascending=False)
    run_df.to_csv(summary_dir / "run_level_overview.csv", index=False)

    run_delta_rows: list[dict[str, Any]] = []
    if len(run_df) >= 2:
        by_name = run_df.sort_values("run_name").reset_index(drop=True)
        for i in range(1, len(by_name)):
            prev_row = by_name.iloc[i - 1]
            curr_row = by_name.iloc[i]
            run_delta_rows.append(
                {
                    "from_run": prev_row["run_name"],
                    "to_run": curr_row["run_name"],
                    "delta_ensemble_pr_auc": _safe_float(curr_row["ensemble_pr_auc"]) - _safe_float(prev_row["ensemble_pr_auc"]),
                    "delta_catboost_pr_auc": _safe_float(curr_row["catboost_pr_auc"]) - _safe_float(prev_row["catboost_pr_auc"]),
                    "delta_deep_learning_pr_auc": _safe_float(curr_row["deep_learning_pr_auc"]) - _safe_float(prev_row["deep_learning_pr_auc"]),
                    "delta_ensemble_brier": _safe_float(curr_row["ensemble_brier_score"]) - _safe_float(prev_row["ensemble_brier_score"]),
                    "delta_catboost_brier": _safe_float(curr_row["catboost_brier_score"]) - _safe_float(prev_row["catboost_brier_score"]),
                    "delta_deep_learning_brier": _safe_float(curr_row["deep_learning_brier_score"]) - _safe_float(prev_row["deep_learning_brier_score"]),
                }
            )
    pd.DataFrame(run_delta_rows).to_csv(summary_dir / "run_to_run_deltas.csv", index=False)

    pd.DataFrame(delta_rows).to_csv(summary_dir / "run_level_bootstrap_deltas.csv", index=False)
    pd.DataFrame(skipped_runs).to_csv(summary_dir / "skipped_runs.csv", index=False)

    generated_utc = datetime.now(timezone.utc).isoformat()
    provenance_payload = {
        "generated_utc": generated_utc,
        "suite_root": str(out_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "invocation": invocation,
        "training_pipeline": "pipelines/run_full_pipeline.py",
        "run_level_benchmark_logic": "pipelines/run_ab_benchmark.py",
        "suite_orchestrator": "pipelines/run_benchmark_suite.py",
        "run_dirs_seen": [d.name for d in run_dirs],
        "run_dirs_benchmarked": [r["run_name"] for r in run_rows],
        "run_dirs_skipped": skipped_runs,
        "flat_files": {
            "holdout_predictions": {
                "exists": holdout_file.exists(),
                "benchmarked": holdout_payload is not None,
                "error": holdout_error,
            },
            "val_predictions": {
                "exists": unlabeled_file.exists(),
                "benchmarked": unlabeled_payload is not None,
                "error": unlabeled_error,
            },
        },
    }
    _write_json(provenance_dir / "exact_sources_and_pipelines.json", provenance_payload)

    external_payload = _build_external_competitor_payload(
        generated_utc=generated_utc,
        run_rows=run_rows,
        holdout_payload=holdout_payload,
    )
    _write_external_competitor_docs(external_dir=external_dir, payload=external_payload)

    actions_lines = [
        "# Exact Changes and Actions",
        "",
        "## Command Used",
        "",
        f"1. {invocation}",
        "",
        "## Exact Pipelines",
        "",
        "1. pipelines/run_full_pipeline.py generated run artifacts",
        "2. pipelines/run_ab_benchmark.py executed run-level model-vs-model scoring and bootstrap",
        "3. pipelines/run_benchmark_suite.py orchestrated full benchmark packaging",
        "",
        "## Exact Outputs Created",
        "",
        f"1. Run-level benchmark folders: {len(run_rows)}",
        f"2. Flat-file benchmark folders: {int(holdout_payload is not None) + int(unlabeled_payload is not None)}",
        f"3. Skipped run folders: {len(skipped_runs)}",
        "4. Summary files: run_level_overview.csv, run_level_bootstrap_deltas.csv, run_to_run_deltas.csv, skipped_runs.csv",
        "5. Provenance file: exact_sources_and_pipelines.json",
        "6. External competitor files: external_competitor/README.md, external_competitor/timesfm_positioning.json, external_competitor/dominance_scorecard.csv",
    ]
    _write_readme(provenance_dir / "exact_changes_and_actions.md", actions_lines)

    _write_root_readme(
        out_dir,
        generated_utc=generated_utc,
        n_bootstrap=n_bootstrap,
        seed=seed,
        run_rows=run_rows,
        holdout_payload=holdout_payload,
        unlabeled_payload=unlabeled_payload,
        external_payload=external_payload,
    )

    latest_pointer = {
        "generated_utc": generated_utc,
        "suite_path": str(out_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    _write_json(artifacts_dir / "benchmark_suite_latest.json", latest_pointer)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full benchmark suite across all artifact sources.")
    parser.add_argument("--n-bootstrap", type=int, default=400, help="Bootstrap rounds for run-level A/B benchmarks.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out-name",
        default="benchmark_suite",
        help="Output prefix under artifacts/; final folder is <out-name>_<timestamp>",
    )
    args = parser.parse_args()

    invocation = f"py -3.10 pipelines/run_benchmark_suite.py --n-bootstrap {int(args.n_bootstrap)} --seed {int(args.seed)} --out-name {str(args.out_name)}"
    out_dir = run_suite(
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
        out_name=str(args.out_name),
        invocation=invocation,
    )
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
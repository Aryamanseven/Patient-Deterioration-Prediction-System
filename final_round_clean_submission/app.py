from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

TITLE = "PS-2 Patient Deterioration Prediction System (Team ANC-052)"

PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_DIR.parent
EVIDENCE_DIR = PACKAGE_DIR / "evidence"

EVIDENCE_JSON = EVIDENCE_DIR / "evidence_latest_run.json"
BENCHMARK_JSON = EVIDENCE_DIR / "benchmark_summary.json"
BENCHMARK_FULL_CSV = EVIDENCE_DIR / "benchmark_full_sample_metrics.csv"
BENCHMARK_SUBSAMPLE_CSV = EVIDENCE_DIR / "benchmark_subsample_summary.csv"


st.set_page_config(
    page_title=TITLE,
    page_icon="H",
    layout="wide",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _load_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse JSON file: {path}") from exc


def _resolve_predictions_path(evidence: dict[str, Any]) -> Path | None:
    run_path_raw = str(evidence.get("run_path", "")).replace("\\", "/").strip()
    if not run_path_raw:
        return None

    run_path = Path(run_path_raw)
    if not run_path.is_absolute():
        run_path = (REPO_ROOT / run_path).resolve()

    pred_path = run_path / "predictions.csv"
    if pred_path.exists():
        return pred_path
    return None


def _load_bundle() -> dict[str, Any]:
    evidence = _load_required_json(EVIDENCE_JSON)
    benchmark = _load_required_json(BENCHMARK_JSON)

    full = benchmark.get("full_sample_metrics", {})
    ens = full.get("latest_ensemble", {})
    delta = full.get("delta", {})
    align = benchmark.get("alignment", {})
    sampling = benchmark.get("evaluation_sampling", {})

    full_df = pd.DataFrame()
    if BENCHMARK_FULL_CSV.exists():
        full_df = pd.read_csv(BENCHMARK_FULL_CSV)

    subsample_df = pd.DataFrame()
    if BENCHMARK_SUBSAMPLE_CSV.exists():
        subsample_df = pd.read_csv(BENCHMARK_SUBSAMPLE_CSV)

    pred_path = _resolve_predictions_path(evidence)
    pred_df = pd.DataFrame()
    if pred_path is not None:
        pred_df = pd.read_csv(pred_path)

    return {
        "run_name": evidence.get("run_name", "N/A"),
        "metrics": evidence.get("metrics", {}),
        "config_highlights": evidence.get("config_highlights", {}),
        "ensemble": ens,
        "delta": delta,
        "alignment": align,
        "sampling": sampling,
        "full_df": full_df,
        "subsample_df": subsample_df,
        "pred_path": pred_path,
        "pred_df": pred_df,
    }


def _render_best_model(bundle: dict[str, Any]) -> None:
    st.title(TITLE)
    st.caption("Best-model-only final dashboard")

    st.success(f"Loaded final run: {bundle['run_name']}")

    metrics = bundle["metrics"]
    ensemble = bundle["ensemble"]
    delta = bundle["delta"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", "Latest Ensemble")
    c2.metric("Ensemble PR-AUC", f"{_safe_float(metrics.get('ensemble_pr_auc')):.6f}")
    c3.metric("Ensemble ROC-AUC", f"{_safe_float(metrics.get('ensemble_roc_auc')):.6f}")

    d1, d2, d3 = st.columns(3)
    d1.metric("Ensemble Brier", f"{_safe_float(ensemble.get('brier_score')):.6f}")
    d2.metric("PR Delta vs TimeSFM Proxy", f"{_safe_float(delta.get('pr_auc_ensemble_minus_timesfm')):.6f}")
    d3.metric("ROC Delta vs TimeSFM Proxy", f"{_safe_float(delta.get('roc_auc_ensemble_minus_timesfm')):.6f}")

    st.subheader("Execution Profile (from final run config highlights)")
    cfg = bundle["config_highlights"]
    profile_rows = [
        {"item": "Python", "value": str(cfg.get("python_version", "N/A"))},
        {"item": "CatBoost iterations", "value": str(cfg.get("catboost_iterations", "N/A"))},
        {"item": "CatBoost learning_rate", "value": str(cfg.get("catboost_learning_rate", "N/A"))},
        {"item": "SSL reuse existing", "value": str(bool(cfg.get("ssl_reuse_existing", False)))},
        {"item": "Federated Learning enabled", "value": str(bool(cfg.get("federated_learning_enabled", False)))},
        {"item": "Domain Generalization enabled", "value": str(bool(cfg.get("domain_generalization_enabled", False)))},
        {"item": "XAI enabled", "value": str(bool(cfg.get("xai_enabled", False)))},
    ]
    st.dataframe(pd.DataFrame(profile_rows), hide_index=True, use_container_width=True)


def _render_inference_trace(bundle: dict[str, Any]) -> None:
    st.title("Inference Trace (Best Model)")
    pred_df = bundle["pred_df"].copy()

    required_cols = ["y_true", "y_proba_ensemble"]
    if not all(col in pred_df.columns for col in required_cols):
        st.error("Inference source file does not contain required columns.")
        return

    pred_df["y_true"] = pd.to_numeric(pred_df["y_true"], errors="coerce").fillna(0).astype(int)
    pred_df["risk_score"] = pd.to_numeric(pred_df["y_proba_ensemble"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    threshold = st.slider("Alert threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.01)
    pred_df["predicted_alert"] = (pred_df["risk_score"] >= threshold).astype(int)

    tp = int(((pred_df["y_true"] == 1) & (pred_df["predicted_alert"] == 1)).sum())
    tn = int(((pred_df["y_true"] == 0) & (pred_df["predicted_alert"] == 0)).sum())
    fp = int(((pred_df["y_true"] == 0) & (pred_df["predicted_alert"] == 1)).sum())
    fn = int(((pred_df["y_true"] == 1) & (pred_df["predicted_alert"] == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("TP", str(tp))
    a2.metric("FP", str(fp))
    a3.metric("TN", str(tn))
    a4.metric("FN", str(fn))

    b1, b2, b3 = st.columns(3)
    b1.metric("Precision", f"{precision:.4f}")
    b2.metric("Recall", f"{recall:.4f}")
    b3.metric("Rows", str(len(pred_df)))

    def _risk_band(score: float) -> str:
        if score >= 0.80:
            return "critical"
        if score >= 0.50:
            return "high"
        if score >= 0.20:
            return "medium"
        return "low"

    pred_df["risk_band"] = pred_df["risk_score"].apply(_risk_band)
    top = pred_df.sort_values("risk_score", ascending=False).head(20)

    st.subheader("Top 20 Highest-Risk Inference Rows")
    st.dataframe(
        top[["y_true", "risk_score", "risk_band", "predicted_alert"]],
        hide_index=True,
        use_container_width=True,
    )


def _render_submission_lock() -> None:
    st.title("Submission Lock")

    required = [
        PACKAGE_DIR / "notebooks" / "Final_Round_Reproducible_Notebook.ipynb",
        PACKAGE_DIR / "presentation" / "AesCodeNexus_Final_Round_Deck.pptx",
        PACKAGE_DIR / "scripts" / "FINAL_DEMO_VIDEO_SCRIPT.md",
        PACKAGE_DIR / "scripts" / "PORTAL_SUBMISSION_CHECKLIST.md",
        PACKAGE_DIR / "evidence" / "evidence_latest_run.json",
        PACKAGE_DIR / "evidence" / "benchmark_summary.json",
    ]

    rows = [{"file": str(p.relative_to(PACKAGE_DIR)).replace("\\", "/"), "exists": p.exists()} for p in required]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if all(row["exists"] for row in rows):
        st.success("Package lock status: READY")
    else:
        st.error("Package lock status: MISSING REQUIRED FILES")

    st.markdown("### Final Portal Rules")
    st.markdown(
        """
1. Notebook link must be public (Kaggle/Colab)
2. PPT link must be public and max 10 slides
3. Demo video link must be public
4. Verify all three links in incognito before submit
        """
    )
    st.markdown("Submission portal: https://docs.google.com/forms/d/e/1FAIpQLSerBwRsj4CpHfX5tyyHJ7yelw-cbImuCDB02gyJWretVqY2bw/viewform?usp=dialog")


def main() -> None:
    try:
        bundle = _load_bundle()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.title("ANC-052 Final")
    st.sidebar.caption("Best-model-only view")

    pages = ["Best Model", "Submission Lock"]
    if not bundle["pred_df"].empty:
        pages.insert(1, "Inference Trace")

    page = st.sidebar.radio("Page", pages)

    if page == "Best Model":
        _render_best_model(bundle)
    elif page == "Inference Trace":
        _render_inference_trace(bundle)
    else:
        _render_submission_lock()


if __name__ == "__main__":
    main()

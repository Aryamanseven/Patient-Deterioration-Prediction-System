"""
PhysioGuard Final Dashboard

Clean final-round Streamlit dashboard that shows only the best available
ensemble model evidence and submission-ready outputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_loader import load_and_split_data
from core.features import EPISODE_COLUMN, HOUR_COLUMN, TARGET_COLUMN

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_PATH = PROJECT_ROOT / "dataset" / "train.csv"
EVIDENCE_PATH = ARTIFACTS_DIR / "evidence_latest_run.json"
FINAL_BENCHMARK_POINTER = ARTIFACTS_DIR / "final_benchmark_latest.json"

st.set_page_config(
    page_title="PhysioGuard Final Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #0f1f38 0%, #112a4a 100%);
    border: 1px solid #1b3a63;
    border-radius: 14px;
    padding: 14px;
}
.metric-title {
    color: #8fa9c6;
    font-size: 0.80rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.metric-value {
    color: #e8f3ff;
    font-size: 1.85rem;
    font-weight: 700;
}
.section-title {
    border-left: 4px solid #1f8fff;
    padding-left: 10px;
    margin-top: 10px;
    margin-bottom: 10px;
    font-weight: 700;
    color: #d9ebff;
}
</style>
""",
    unsafe_allow_html=True,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pick_best_run_dir() -> Path:
    evidence = _read_json(EVIDENCE_PATH)
    run_name = str(evidence.get("run_name", "")).strip()
    if run_name:
        candidate = ARTIFACTS_DIR / run_name
        if candidate.exists() and (candidate / "predictions.csv").exists() and (candidate / "metrics.json").exists():
            return candidate

    run_dirs = [
        d
        for d in ARTIFACTS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("run_") and (d / "predictions.csv").exists() and (d / "metrics.json").exists()
    ]
    if not run_dirs:
        raise FileNotFoundError("No complete run found in artifacts/run_*")
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0]


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    out = {
        "test_size": 0.2,
        "seed": 42,
        "use_advanced": True,
        "use_clinical_scores": True,
    }
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists() or yaml is None:
        return out

    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        out["test_size"] = float(cfg.get("data", {}).get("test_size", out["test_size"]))
        out["seed"] = int(cfg.get("general", {}).get("seed", out["seed"]))
        out["use_advanced"] = bool(cfg.get("features", {}).get("use_advanced", out["use_advanced"]))
        out["use_clinical_scores"] = bool(
            cfg.get("features", {}).get("use_clinical_scores", out["use_clinical_scores"])
        )
    except Exception:
        return out
    return out


def _extract_ensemble_metrics(metrics_payload: dict[str, Any]) -> dict[str, float]:
    m = metrics_payload.get("ensemble", {}) if "ensemble" in metrics_payload else metrics_payload

    def _sf(v: Any) -> float:
        try:
            if pd.isna(v):
                return 0.0
            return float(v)
        except Exception:
            return 0.0

    return {
        "pr_auc": _sf(m.get("pr_auc")),
        "roc_auc": _sf(m.get("roc_auc")),
        "brier_score": _sf(m.get("brier_score")),
    }


@st.cache_data(show_spinner=True)
def load_best_run_bundle() -> dict[str, Any]:
    run_dir = _pick_best_run_dir()
    run_cfg = _load_run_config(run_dir)

    preds_path = run_dir / "predictions.csv"
    pred_df = pd.read_csv(preds_path)
    required_cols = {"y_true", "y_proba_ensemble"}
    missing = sorted(required_cols - set(pred_df.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    _, _, val_df, _ = load_and_split_data(
        data_path=DATA_PATH,
        test_size=run_cfg["test_size"],
        max_rows=None,
        use_advanced_features=run_cfg["use_advanced"],
        use_clinical_scores=run_cfg["use_clinical_scores"],
        random_state=run_cfg["seed"],
    )

    if len(val_df) != len(pred_df):
        raise ValueError(f"Validation rows ({len(val_df)}) do not match predictions rows ({len(pred_df)})")

    y_true_val = pd.to_numeric(val_df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int).to_numpy()
    y_true_pred = pd.to_numeric(pred_df["y_true"], errors="coerce").fillna(0).astype(int).to_numpy()
    if not np.array_equal(y_true_val, y_true_pred):
        raise ValueError("Prediction labels do not align with reconstructed validation split")

    keep_cols = [
        EPISODE_COLUMN,
        HOUR_COLUMN,
        TARGET_COLUMN,
        "age",
        "heart_rate",
        "respiratory_rate",
        "spo2_pct",
        "systolic_bp",
        "temperature_c",
        "lactate",
        "news_score",
        "mean_arterial_pressure",
    ]
    available = [c for c in keep_cols if c in val_df.columns]
    df = val_df[available].copy()
    df["deterioration_risk"] = np.clip(
        pd.to_numeric(pred_df["y_proba_ensemble"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        0.0,
        1.0,
    )

    metrics_payload = _read_json(run_dir / "metrics.json")
    ensemble_metrics = _extract_ensemble_metrics(metrics_payload)

    final_ptr = _read_json(FINAL_BENCHMARK_POINTER)
    final_rel = str(final_ptr.get("path", "")).strip()
    benchmark_dir = PROJECT_ROOT / final_rel if final_rel else None
    benchmark_summary = _read_json(benchmark_dir / "summary.json") if benchmark_dir and benchmark_dir.exists() else {}

    top_features_path = run_dir / "top_features.csv"
    shap_img_path = run_dir / "shap_summary.png"

    return {
        "run_name": run_dir.name,
        "run_dir": run_dir,
        "df": df,
        "ensemble_metrics": ensemble_metrics,
        "benchmark_summary": benchmark_summary,
        "benchmark_dir": benchmark_dir,
        "top_features_path": top_features_path,
        "shap_img_path": shap_img_path,
    }


def _risk_band(score: float) -> str:
    if score >= 0.6:
        return "High"
    if score >= 0.3:
        return "Moderate"
    return "Low"


def _as_float(value: Any, fallback: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return fallback
        return float(value)
    except Exception:
        return fallback


def render_sidebar(pages: list[str], run_name: str) -> str:
    with st.sidebar:
        st.title("🏥 PhysioGuard")
        st.caption("Final-round clean dashboard")
        st.markdown("---")
        st.success(f"Best run loaded: {run_name}")
        st.caption("Model shown: Ensemble (best validated)")
        st.markdown("---")
        page = st.radio("Navigate", pages, label_visibility="collapsed")
        st.markdown("---")
        st.caption("Decision support only. Clinical judgment required.")
    return page


def render_overview(df: pd.DataFrame, ensemble_metrics: dict[str, float]) -> None:
    st.markdown('<div class="section-title">Executive Overview</div>', unsafe_allow_html=True)

    latest = df.sort_values(HOUR_COLUMN).groupby(EPISODE_COLUMN).tail(1).copy()
    latest["risk_band"] = latest["deterioration_risk"].apply(_risk_band)

    total = int(len(latest))
    high = int((latest["risk_band"] == "High").sum())
    mod = int((latest["risk_band"] == "Moderate").sum())
    low = int((latest["risk_band"] == "Low").sum())

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Patients", total),
        (c2, "High Risk", high),
        (c3, "Moderate Risk", mod),
        (c4, "Low Risk", low),
    ]
    for col, title, value in cards:
        with col:
            st.markdown(
                f"<div class='metric-card'><div class='metric-title'>{title}</div><div class='metric-value'>{value}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("### Best Ensemble Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("PR-AUC", f"{ensemble_metrics['pr_auc']:.6f}")
    m2.metric("ROC-AUC", f"{ensemble_metrics['roc_auc']:.6f}")
    m3.metric("Brier", f"{ensemble_metrics['brier_score']:.6f}")

    p1, p2 = st.columns([2, 1])
    with p1:
        fig = px.histogram(
            latest,
            x="deterioration_risk",
            color="risk_band",
            color_discrete_map={"High": "#e63946", "Moderate": "#ff9f1c", "Low": "#2a9d8f"},
            nbins=40,
            title="Risk Distribution",
            labels={"deterioration_risk": "Ensemble Risk Score"},
        )
        fig.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with p2:
        pie = go.Figure(
            data=[
                go.Pie(
                    labels=["High", "Moderate", "Low"],
                    values=[high, mod, low],
                    marker=dict(colors=["#e63946", "#ff9f1c", "#2a9d8f"]),
                    hole=0.45,
                    textinfo="percent+label",
                )
            ]
        )
        pie.update_layout(template="plotly_dark", height=360, showlegend=False)
        st.plotly_chart(pie, use_container_width=True)

    st.markdown("### Top High-Risk Episodes")
    cols = [EPISODE_COLUMN, "deterioration_risk", "age", "heart_rate", "respiratory_rate", "spo2_pct", "news_score"]
    show_cols = [c for c in cols if c in latest.columns]
    top = latest.sort_values("deterioration_risk", ascending=False).head(15)[show_cols].copy()
    top = top.rename(columns={EPISODE_COLUMN: "episode_id", "deterioration_risk": "risk"})
    st.dataframe(top, use_container_width=True, hide_index=True)


def render_patient_deep_dive(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Patient Deep Dive</div>', unsafe_allow_html=True)

    latest = df.sort_values(HOUR_COLUMN).groupby(EPISODE_COLUMN).tail(1)
    episode_order = latest.sort_values("deterioration_risk", ascending=False)[EPISODE_COLUMN].tolist()
    selected = st.selectbox("Select episode", episode_order)

    ep = df[df[EPISODE_COLUMN] == selected].sort_values(HOUR_COLUMN).copy()
    current_risk = float(ep["deterioration_risk"].iloc[-1])

    s1, s2, s3 = st.columns(3)
    s1.metric("Current Risk", f"{current_risk:.1%}")
    if "news_score" in ep.columns:
        s2.metric("Current NEWS", f"{_as_float(ep['news_score'].iloc[-1]):.0f}")
    if "age" in ep.columns:
        s3.metric("Age", f"{_as_float(ep['age'].iloc[0]):.0f}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ep[HOUR_COLUMN],
            y=ep["deterioration_risk"],
            mode="lines+markers",
            name="Ensemble Risk",
            line=dict(color="#1f8fff", width=3),
        )
    )
    if "news_score" in ep.columns:
        news = pd.to_numeric(ep["news_score"], errors="coerce").fillna(0.0)
        max_news = float(news.max()) if float(news.max()) > 0 else 1.0
        fig.add_trace(
            go.Scatter(
                x=ep[HOUR_COLUMN],
                y=news / max_news,
                mode="lines",
                name="NEWS (normalized)",
                line=dict(color="#ff9f1c", width=2, dash="dot"),
            )
        )

    det = ep[ep[TARGET_COLUMN] == 1]
    if not det.empty:
        fig.add_trace(
            go.Scatter(
                x=det[HOUR_COLUMN],
                y=det["deterioration_risk"],
                mode="markers",
                name="Deterioration label",
                marker=dict(color="#e63946", size=10, symbol="x"),
            )
        )

    fig.update_layout(template="plotly_dark", height=360, title="Risk Timeline")
    st.plotly_chart(fig, use_container_width=True)

    vital_cols = ["heart_rate", "respiratory_rate", "spo2_pct", "systolic_bp", "temperature_c", "lactate"]
    vital_cols = [c for c in vital_cols if c in ep.columns]
    if vital_cols:
        long_df = ep[[HOUR_COLUMN] + vital_cols].melt(id_vars=[HOUR_COLUMN], var_name="vital", value_name="value")
        vf = px.line(
            long_df,
            x=HOUR_COLUMN,
            y="value",
            color="vital",
            title="Vitals Over Time",
        )
        vf.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(vf, use_container_width=True)


def render_explainability(top_features_path: Path, shap_img_path: Path) -> None:
    st.markdown('<div class="section-title">Explainability (Best Run)</div>', unsafe_allow_html=True)

    if shap_img_path.exists():
        st.image(str(shap_img_path), caption="SHAP Summary")

    if top_features_path.exists():
        tf = pd.read_csv(top_features_path)
        abs_col = "mean_abs_shap" if "mean_abs_shap" in tf.columns else "shap_value_abs_mean"
        if "feature" in tf.columns and abs_col in tf.columns:
            top = tf[["feature", abs_col]].head(25).copy()
            fig = px.bar(top, x=abs_col, y="feature", orientation="h", title="Top Feature Contributions")
            fig.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"), height=640)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top, use_container_width=True, hide_index=True)


def render_final_benchmark(benchmark_dir: Path | None, benchmark_summary: dict[str, Any]) -> None:
    st.markdown('<div class="section-title">Final Benchmark (Ensemble vs TimeSFM Proxy)</div>', unsafe_allow_html=True)

    full = benchmark_summary.get("full_sample_metrics", {}) if benchmark_summary else {}
    ens = full.get("latest_ensemble", {})
    tsf = full.get("timesfm_proxy", {})
    delta = full.get("delta", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("PR-AUC Delta", f"{_as_float(delta.get('pr_auc_ensemble_minus_timesfm')):.6f}")
    c2.metric("ROC-AUC Delta", f"{_as_float(delta.get('roc_auc_ensemble_minus_timesfm')):.6f}")
    c3.metric("Brier Delta", f"{_as_float(delta.get('brier_timesfm_minus_ensemble')):.6f}")

    facts_rows = [
        {
            "model": "latest_ensemble",
            "pr_auc": _as_float(ens.get("pr_auc")),
            "roc_auc": _as_float(ens.get("roc_auc")),
            "brier_score": _as_float(ens.get("brier_score")),
            "n": int(_as_float(ens.get("n"))),
        },
        {
            "model": "timesfm_proxy",
            "pr_auc": _as_float(tsf.get("pr_auc")),
            "roc_auc": _as_float(tsf.get("roc_auc")),
            "brier_score": _as_float(tsf.get("brier_score")),
            "n": int(_as_float(tsf.get("n"))),
        },
    ]
    st.dataframe(pd.DataFrame(facts_rows), use_container_width=True, hide_index=True)

    if benchmark_dir and (benchmark_dir / "head_to_head_by_fraction.csv").exists():
        frac = pd.read_csv(benchmark_dir / "head_to_head_by_fraction.csv")
        keep = [
            "fraction",
            "delta_pr_ensemble_minus_timesfm",
            "delta_roc_ensemble_minus_timesfm",
            "delta_brier_timesfm_minus_ensemble",
        ]
        keep = [c for c in keep if c in frac.columns]
        if keep:
            f = frac[keep].copy()
            st.dataframe(f, use_container_width=True, hide_index=True)
            melt = f.melt(id_vars=["fraction"], var_name="metric", value_name="delta")
            fig = px.line(melt, x="fraction", y="delta", color="metric", markers=True, title="Delta by Fraction")
            fig.update_layout(template="plotly_dark", height=420)
            st.plotly_chart(fig, use_container_width=True)


def render_submission_checklist(run_name: str, run_dir: Path, benchmark_dir: Path | None) -> None:
    st.markdown('<div class="section-title">Final Round Submission Checklist</div>', unsafe_allow_html=True)

    checks = [
        {"item": "Best run folder", "path": str(run_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"), "ready": run_dir.exists()},
        {"item": "Run metrics", "path": str((run_dir / "metrics.json").relative_to(PROJECT_ROOT)).replace("\\", "/"), "ready": (run_dir / "metrics.json").exists()},
        {"item": "Run predictions", "path": str((run_dir / "predictions.csv").relative_to(PROJECT_ROOT)).replace("\\", "/"), "ready": (run_dir / "predictions.csv").exists()},
        {"item": "Final benchmark package", "path": str(benchmark_dir.relative_to(PROJECT_ROOT)).replace("\\", "/") if benchmark_dir else "", "ready": bool(benchmark_dir and benchmark_dir.exists())},
        {"item": "Evidence JSON", "path": str(EVIDENCE_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"), "ready": EVIDENCE_PATH.exists()},
    ]
    st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)

    st.markdown("### Portal Deliverables")
    st.markdown(
        """
1. Updated public Kaggle/Colab notebook
2. Final presentation deck (max 10 slides)
3. Demo video link (public)
4. Verify all links are publicly accessible before submit
        """
    )

    st.markdown("### Current Active Run")
    st.code(run_name, language="text")


def main() -> None:
    try:
        bundle = load_best_run_bundle()
    except Exception as exc:
        st.error("Dashboard could not load final artifacts.")
        st.code(str(exc), language="text")
        st.stop()

    run_name = bundle["run_name"]
    run_dir = bundle["run_dir"]
    df = bundle["df"]
    ensemble_metrics = bundle["ensemble_metrics"]
    benchmark_summary = bundle["benchmark_summary"]
    benchmark_dir = bundle["benchmark_dir"]
    top_features_path = bundle["top_features_path"]
    shap_img_path = bundle["shap_img_path"]

    pages = [
        "🏠 Executive Overview",
        "👤 Patient Deep Dive",
        "📈 Final Benchmark",
        "✅ Submission Checklist",
    ]
    if top_features_path.exists() or shap_img_path.exists():
        pages.insert(2, "🔍 Explainability")

    page = render_sidebar(pages=pages, run_name=run_name)

    st.markdown(
        f"""
<div style="margin: 6px 0 14px 0; padding: 10px 14px; border-radius: 10px;
            border: 1px solid #1e3a5f; background: linear-gradient(90deg, #0f1b2d, #10243f);">
    <span style="color:#9cc8ff;font-weight:600;">Loaded run:</span>
    <span style="color:#e6f1ff;"> {run_name} </span>
    <span style="color:#9cc8ff;font-weight:600; margin-left: 10px;">Model:</span>
    <span style="color:#e6f1ff;"> Ensemble only </span>
</div>
        """,
        unsafe_allow_html=True,
    )

    if page == "🏠 Executive Overview":
        render_overview(df=df, ensemble_metrics=ensemble_metrics)
    elif page == "👤 Patient Deep Dive":
        render_patient_deep_dive(df=df)
    elif page == "🔍 Explainability":
        render_explainability(top_features_path=top_features_path, shap_img_path=shap_img_path)
    elif page == "📈 Final Benchmark":
        render_final_benchmark(benchmark_dir=benchmark_dir, benchmark_summary=benchmark_summary)
    elif page == "✅ Submission Checklist":
        render_submission_checklist(run_name=run_name, run_dir=run_dir, benchmark_dir=benchmark_dir)


if __name__ == "__main__":
    main()

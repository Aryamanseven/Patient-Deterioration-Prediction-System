"""
Final-Round Standalone Dashboard

This app reads only from files inside final_round_clean_submission.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = BASE_DIR / "evidence"

EVIDENCE_JSON = EVIDENCE_DIR / "evidence_latest_run.json"
BENCHMARK_JSON = EVIDENCE_DIR / "benchmark_summary.json"
BENCHMARK_SUBSAMPLE_CSV = EVIDENCE_DIR / "benchmark_subsample_summary.csv"
BENCHMARK_FULL_CSV = EVIDENCE_DIR / "benchmark_full_sample_metrics.csv"

st.set_page_config(
    page_title="ANC-052 Final Dashboard",
    page_icon="H",
    layout="wide",
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def load_bundle() -> dict[str, Any]:
    evidence = read_json(EVIDENCE_JSON)
    benchmark = read_json(BENCHMARK_JSON)

    metrics = evidence.get("metrics", {})
    full = benchmark.get("full_sample_metrics", {})
    delta = full.get("delta", {})
    align = benchmark.get("alignment", {})
    sampling = benchmark.get("evaluation_sampling", {})

    subsample_df = pd.DataFrame()
    if BENCHMARK_SUBSAMPLE_CSV.exists():
        try:
            subsample_df = pd.read_csv(BENCHMARK_SUBSAMPLE_CSV)
        except Exception:
            subsample_df = pd.DataFrame()

    full_df = pd.DataFrame()
    if BENCHMARK_FULL_CSV.exists():
        try:
            full_df = pd.read_csv(BENCHMARK_FULL_CSV)
        except Exception:
            full_df = pd.DataFrame()

    return {
        "run_name": evidence.get("run_name", "N/A"),
        "metrics": metrics,
        "benchmark": benchmark,
        "full": full,
        "delta": delta,
        "align": align,
        "sampling": sampling,
        "subsample_df": subsample_df,
        "full_df": full_df,
        "config_highlights": evidence.get("config_highlights", {}),
    }


def render_overview(bundle: dict[str, Any]) -> None:
    st.title("Final Round Dashboard")
    st.caption("Standalone view using only packaged evidence files")

    run_name = bundle["run_name"]
    metrics = bundle["metrics"]
    delta = bundle["delta"]
    align = bundle["align"]

    st.info(f"Loaded run: {run_name}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Best PR-AUC", f"{safe_float(metrics.get('ensemble_pr_auc')):.6f}")
    c2.metric("Best ROC-AUC", f"{safe_float(metrics.get('ensemble_roc_auc')):.6f}")
    c3.metric("PR-AUC Delta vs TimeSFM Proxy", f"{safe_float(delta.get('pr_auc_ensemble_minus_timesfm')):.6f}")

    d1, d2, d3 = st.columns(3)
    d1.metric("ROC-AUC Delta", f"{safe_float(delta.get('roc_auc_ensemble_minus_timesfm')):.6f}")
    d2.metric("Brier Improvement", f"{safe_float(delta.get('brier_timesfm_minus_ensemble')):.6f}")
    d3.metric("Alignment Labels Match", str(bool(align.get("labels_match", False))))

    summary_rows = [
        {"Metric": "Alignment Rows", "Value": int(safe_float(align.get("rows")))},
        {"Metric": "Positive Rate", "Value": f"{safe_float(align.get('positive_rate')):.6f}"},
        {"Metric": "Eval Rows Used", "Value": int(safe_float(bundle['sampling'].get('rows_after_sampling')))},
    ]
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)


def render_benchmark(bundle: dict[str, Any]) -> None:
    st.title("Final Benchmark Evidence")

    full = bundle["full"]
    ens = full.get("latest_ensemble", {})
    tsf = full.get("timesfm_proxy", {})

    table = pd.DataFrame(
        [
            {
                "model": "latest_ensemble",
                "pr_auc": safe_float(ens.get("pr_auc")),
                "roc_auc": safe_float(ens.get("roc_auc")),
                "brier_score": safe_float(ens.get("brier_score")),
                "n": int(safe_float(ens.get("n"))),
            },
            {
                "model": "timesfm_proxy",
                "pr_auc": safe_float(tsf.get("pr_auc")),
                "roc_auc": safe_float(tsf.get("roc_auc")),
                "brier_score": safe_float(tsf.get("brier_score")),
                "n": int(safe_float(tsf.get("n"))),
            },
        ]
    )
    st.dataframe(table, hide_index=True, use_container_width=True)

    subsample_df = bundle["subsample_df"]
    if not subsample_df.empty:
        possible_cols = [
            "fraction",
            "mean_delta_pr_auc_ensemble_minus_timesfm",
            "mean_delta_roc_auc_ensemble_minus_timesfm",
            "mean_delta_brier_timesfm_minus_ensemble",
        ]
        have = [c for c in possible_cols if c in subsample_df.columns]
        if len(have) >= 2 and "fraction" in have:
            melt = subsample_df[have].melt(id_vars=["fraction"], var_name="metric", value_name="delta")
            fig = px.line(melt, x="fraction", y="delta", color="metric", markers=True, title="Subsample Delta Trend")
            st.plotly_chart(fig, use_container_width=True)

    full_df = bundle["full_df"]
    if not full_df.empty:
        st.subheader("Raw Full-Sample Metrics Table")
        st.dataframe(full_df, hide_index=True, use_container_width=True)


def render_submission_checklist() -> None:
    st.title("Final Submission Checklist")

    required = [
        BASE_DIR / "notebooks" / "Final_Round_Reproducible_Notebook.ipynb",
        BASE_DIR / "presentation" / "AesCodeNexus_Final_Round_Deck.pptx",
        BASE_DIR / "scripts" / "FINAL_DEMO_VIDEO_SCRIPT.md",
        BASE_DIR / "scripts" / "PORTAL_SUBMISSION_CHECKLIST.md",
        BASE_DIR / "evidence" / "evidence_latest_run.json",
        BASE_DIR / "evidence" / "benchmark_summary.json",
    ]

    rows = []
    for p in required:
        rows.append(
            {
                "file": str(p.relative_to(BASE_DIR)).replace("\\", "/"),
                "exists": p.exists(),
            }
        )

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    ready = all(r["exists"] for r in rows)
    if ready:
        st.success("Package status: ready")
    else:
        st.error("Package status: missing files")

    st.markdown("### Portal Rules")
    st.markdown(
        """
1. Notebook link public
2. PPT link public (max 10 slides)
3. Demo video link public
4. Verify all links in incognito before submit
        """
    )


def main() -> None:
    st.sidebar.title("ANC-052")
    st.sidebar.caption("One-folder final package")
    page = st.sidebar.radio("Page", ["Overview", "Benchmark", "Submission Checklist"])

    bundle = load_bundle()

    if page == "Overview":
        render_overview(bundle)
    elif page == "Benchmark":
        render_benchmark(bundle)
    else:
        render_submission_checklist()


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physio_warning import (  # noqa: E402
    EPISODE_COLUMN,
    HOUR_COLUMN,
    TARGET_COLUMN,
    add_episode_ids,
    assign_risk_band,
    engineer_features,
    get_model_feature_columns,
    load_metadata,
)

DEFAULT_DEMO_ARTIFACT_DIR = "artifacts/model_search_revalidated_20260326"

st.set_page_config(
    page_title="PhysioGuard EWS",
    page_icon=":hospital:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .hero {
        padding: 1.3rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #0f4c5c, #1b9aaa 55%, #f4a261);
        color: #ffffff;
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
    }
    .hero p {
        margin: 0.4rem 0 0 0;
        max-width: 58rem;
        line-height: 1.5;
    }
    .pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #102a43;
        background: #ffe8b6;
    }
    </style>
    <div class="hero">
        <div class="pill">Early warning prototype</div>
        <h1>PhysioGuard: Patient Deterioration Monitor</h1>
        <p>
            Trend-aware risk scoring for physiological deterioration in the next 12 hours,
            built from hourly vital signs and laboratory indicators.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model_bundle(artifact_dir: str) -> tuple[CatBoostClassifier, dict]:
    artifact_path = Path(artifact_dir)

    metadata_path = artifact_path / "metadata.json"
    legacy_model_path = artifact_path / "deterioration_model.cbm"

    if metadata_path.exists() and legacy_model_path.exists():
        metadata = load_metadata(metadata_path)
        metadata.setdefault("model_name", "Baseline CatBoost")
        metadata.setdefault("demo_label", "Baseline CatBoost")
        model = CatBoostClassifier()
        model.load_model(str(legacy_model_path))
        return model, metadata

    summary_path = next(iter(sorted(artifact_path.glob("*_final_artifact_summary.json"))), None)
    focused_model_path = next(iter(sorted(artifact_path.glob("*_full_train_model.cbm"))), None)
    feature_importance_path = next(iter(sorted(artifact_path.glob("*_feature_importance.csv"))), None)

    if summary_path is None or focused_model_path is None:
        raise FileNotFoundError(
            "Expected either a legacy `artifacts` bundle or a focused winner folder with "
            "`*_final_artifact_summary.json` and `*_full_train_model.cbm`.",
        )

    summary = load_metadata(summary_path)
    reference_metrics = summary["reference_holdout_metrics"]
    feature_columns = load_demo_feature_columns()
    top_features: list[dict] = []
    if feature_importance_path is not None and feature_importance_path.exists():
        feature_frame = pd.read_csv(feature_importance_path).head(15).copy()
        feature_frame["importance"] = feature_frame["importance"].astype(float)
        top_features = feature_frame.to_dict(orient="records")

    metadata = {
        "model_name": summary["model_name"],
        "demo_label": "Final Focused CatBoost Winner",
        "feature_columns": feature_columns,
        "thresholds": {
            "watch": float(reference_metrics["watch_threshold"]),
            "alert": float(reference_metrics["alert_threshold"]),
        },
        "metrics": {
            "pr_auc": float(reference_metrics["holdout_pr_auc"]),
            "roc_auc": float(reference_metrics["holdout_roc_auc"]),
            "brier_score": float(reference_metrics["holdout_brier"]),
            "watch_precision": float(reference_metrics["watch_precision"]),
            "watch_recall": float(reference_metrics["watch_recall"]),
            "alert_precision": float(reference_metrics["alert_precision"]),
            "alert_recall": float(reference_metrics["alert_recall"]),
            "alert_f1": float(reference_metrics["alert_f1"]),
        },
        "top_features": top_features,
    }

    model = CatBoostClassifier()
    model.load_model(str(focused_model_path))
    return model, metadata


@st.cache_data(show_spinner=False)
def load_local_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_demo_feature_columns() -> list[str]:
    template_df = pd.read_csv(ROOT / "dataset" / "train.csv", nrows=1)
    featured = engineer_features(add_episode_ids(template_df))
    return get_model_feature_columns(featured)


@st.cache_data(show_spinner=False)
def score_dataframe(
    raw_df: pd.DataFrame,
    feature_columns: list[str],
    watch_threshold: float,
    alert_threshold: float,
    artifact_dir: str,
) -> pd.DataFrame:
    model, _ = load_model_bundle(artifact_dir)
    featured = engineer_features(add_episode_ids(raw_df))
    scored = featured.copy()
    scored["deterioration_risk"] = model.predict_proba(scored[feature_columns])[:, 1]
    scored["risk_band"] = assign_risk_band(
        scored["deterioration_risk"],
        watch_threshold=watch_threshold,
        alert_threshold=alert_threshold,
    )
    scored["predicted_alert"] = (scored["deterioration_risk"] >= alert_threshold).astype("int8")
    return scored


artifact_dir = st.sidebar.text_input("Artifact directory", value=DEFAULT_DEMO_ARTIFACT_DIR)
try:
    _, metadata = load_model_bundle(artifact_dir)
except FileNotFoundError as exc:
    st.error(
        f"{exc} Point the app at either `artifacts` or "
        "`artifacts/model_search_revalidated_20260326`.",
    )
    st.stop()

feature_columns = metadata["feature_columns"]
watch_threshold = float(metadata["thresholds"]["watch"])
alert_threshold = float(metadata["thresholds"]["alert"])
metrics = metadata.get("metrics", {})

st.sidebar.success(
    f"Demo model: {metadata.get('demo_label', metadata.get('model_name', 'CatBoost'))}",
)
if metrics:
    st.sidebar.caption(
        f"PR-AUC {metrics.get('pr_auc', 0.0):.4f} | "
        f"ROC-AUC {metrics.get('roc_auc', 0.0):.4f}",
    )

source = st.sidebar.radio(
    "Data source",
    options=["Validation dataset", "Training dataset", "Upload CSV"],
)

uploaded_df: pd.DataFrame | None = None
if source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV that matches the training schema to run live predictions.")
        st.stop()
    uploaded_df = pd.read_csv(uploaded_file)

if source == "Validation dataset":
    raw_df = load_local_csv("dataset/val_no_labels.csv")
elif source == "Training dataset":
    raw_df = load_local_csv("dataset/train.csv")
else:
    raw_df = uploaded_df

scored_df = score_dataframe(
    raw_df=raw_df,
    feature_columns=feature_columns,
    watch_threshold=watch_threshold,
    alert_threshold=alert_threshold,
    artifact_dir=artifact_dir,
)

episodes = scored_df[EPISODE_COLUMN].drop_duplicates().tolist()
ranked_episode_risk = (
    scored_df.groupby(EPISODE_COLUMN, sort=False)["deterioration_risk"].max().sort_values(ascending=False)
)

default_episode = int(ranked_episode_risk.index[0])
selected_episode = st.sidebar.selectbox(
    "Episode",
    options=episodes,
    index=episodes.index(default_episode) if default_episode in episodes else 0,
)

vital_options = [
    "heart_rate",
    "respiratory_rate",
    "spo2_pct",
    "temperature_c",
    "systolic_bp",
    "diastolic_bp",
    "oxygen_flow",
    "sepsis_risk_score",
]
selected_vitals = st.multiselect(
    "Vitals to visualize",
    options=vital_options,
    default=["heart_rate", "respiratory_rate", "spo2_pct", "systolic_bp", "temperature_c"],
)

episode_df = scored_df.loc[scored_df[EPISODE_COLUMN] == selected_episode].copy()
latest = episode_df.iloc[-1]
peak_index = episode_df["deterioration_risk"].idxmax()
peak_row = episode_df.loc[peak_index]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest risk", f"{latest['deterioration_risk']:.1%}")
col2.metric("Risk band", str(latest["risk_band"]))
col3.metric("Peak episode risk", f"{peak_row['deterioration_risk']:.1%}")
col4.metric("Hours monitored", int(episode_df[HOUR_COLUMN].max()) + 1)

st.caption(
    f"Demo model: {metadata.get('demo_label', metadata.get('model_name', 'CatBoost'))} | "
    f"Watch threshold {watch_threshold:.3f} | Alert threshold {alert_threshold:.3f}",
)

snapshot_left, snapshot_right = st.columns([1.2, 1.8])
with snapshot_left:
    st.subheader("Patient snapshot")
    snapshot = pd.DataFrame(
        {
            "field": [
                "Age",
                "Gender",
                "Admission type",
                "Comorbidity index",
                "Latest oxygen device",
                "Latest nurse alert",
            ],
            "value": [
                int(latest["age"]),
                latest["gender"],
                latest["admission_type"],
                int(latest["comorbidity_index"]),
                latest["oxygen_device"],
                int(latest["nurse_alert"]),
            ],
        }
    )
    st.dataframe(snapshot, hide_index=True, use_container_width=True)
    st.caption(
        f"Moderate-risk watch threshold: {watch_threshold:.3f} | "
        f"High-risk alert threshold: {alert_threshold:.3f}"
    )

    if TARGET_COLUMN in episode_df.columns:
        st.write(
            "Observed deterioration labels in this episode:",
            int(episode_df[TARGET_COLUMN].sum()),
        )

with snapshot_right:
    st.subheader("Risk trend")
    risk_chart = episode_df.set_index(HOUR_COLUMN)[["deterioration_risk"]]
    st.line_chart(risk_chart, use_container_width=True)

st.subheader("Vital-sign trends")
if selected_vitals:
    st.line_chart(
        episode_df.set_index(HOUR_COLUMN)[selected_vitals],
        use_container_width=True,
    )
else:
    st.info("Select at least one vital sign to plot.")

table_left, table_right = st.columns([1.5, 1.1])
with table_left:
    st.subheader("Latest timeline rows")
    display_columns = [
        HOUR_COLUMN,
        "heart_rate",
        "respiratory_rate",
        "spo2_pct",
        "temperature_c",
        "systolic_bp",
        "diastolic_bp",
        "oxygen_device",
        "oxygen_flow",
        "deterioration_risk",
        "risk_band",
    ]
    if TARGET_COLUMN in episode_df.columns:
        display_columns.append(TARGET_COLUMN)
    st.dataframe(
        episode_df[display_columns].tail(12),
        hide_index=True,
        use_container_width=True,
    )

with table_right:
    st.subheader("Model drivers")
    driver_frame = pd.DataFrame(metadata.get("top_features", []))
    if not driver_frame.empty:
        st.dataframe(driver_frame, hide_index=True, use_container_width=True)
    else:
        st.info("Feature importance will appear after training artifacts are generated.")

st.subheader("Dataset overview")
overview_left, overview_right = st.columns(2)
with overview_left:
    st.write(
        {
            "rows": int(len(scored_df)),
            "episodes": int(scored_df[EPISODE_COLUMN].nunique()),
            "high-risk rows": int((scored_df["risk_band"] == "High").sum()),
            "moderate-risk rows": int((scored_df["risk_band"] == "Moderate").sum()),
        }
    )
with overview_right:
    top_episodes = ranked_episode_risk.head(10).reset_index()
    top_episodes.columns = ["episode_id", "peak_risk"]
    st.dataframe(top_episodes, hide_index=True, use_container_width=True)

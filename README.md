# PS-2 Patient Deterioration Prediction System (Team ANC-052)

Competition repository for early detection of patient physiological deterioration with a reproducible, evidence-first ML pipeline.

This project is designed to optimize three outcomes at the same time:
1. Predictive quality on imbalanced clinical risk prediction.
2. Reproducible execution from configuration files.
3. Audit-ready evidence for reviewers, judges, and collaborators.

---

## 0) Quick Start For New Users

If you are opening this project for the first time and want a complete run:

```powershell
py -3.10 -m pip install -r requirements.txt
py -3.10 pipelines/run_full_pipeline.py --config configs/quick_test.yaml
py -3.10 pipelines/audit_artifacts.py --artifacts-dir artifacts
py -3.10 -m streamlit run dashboard/app.py
```

Then open: `http://localhost:8501`

---

## 1) Project Overview

### Clinical problem
In ward and ICU-style monitoring, deterioration is often visible as a trajectory, not a single bad measurement. A patient may be "stable" at one timestamp, but a multi-hour trend in respiratory, hemodynamic, and inflammatory signals can indicate rising risk.

### Technical objective
Predict deterioration risk within the forecast window (competition framing) from time-indexed vital/lab features plus static context, and surface outputs in a clinician-facing dashboard.

### Why this matters
Late detection increases emergency escalations and avoidable harm. Earlier, ranked triage signals help teams allocate attention to the highest-risk patients first.

### What is different in this repository
1. It is a full pipeline system, not a notebook-only experiment.
2. It combines temporal deep learning and tabular gradient boosting, then fuses them with an ensemble.
3. It exports run evidence and artifact hashes for trust and reproducibility.
4. It includes optional advanced modules (SSL, FL, DG, XAI, DP hooks) under config control.

---

## 2) System Architecture (End To End)

This repository implements a deployable workflow:

```text
Raw Data (dataset/*.csv)
	-> Data loading and split (core/data_loader.py)
	-> Feature engineering (core/features.py)
	-> Temporal and tabular model training (models/*)
	-> Ensemble fusion (models/ensemble.py)
	-> Optional modules (modules/ssl, federated_learning, domain_generalization, xai, differential_privacy)
	-> Verified run artifacts (artifacts/run_*/...)
	-> Streamlit UI (dashboard/app.py)
```

### Not a notebook project
Notebooks exist for submission and walkthrough, but the canonical execution path is script-driven through:
1. `pipelines/run_full_pipeline.py`
2. YAML profiles in `configs/`
3. Artifact checks in `pipelines/audit_artifacts.py`

### Deployable system posture
This repo contains:
1. Trained model serialization.
2. Config-driven reruns.
3. Artifact auditing.
4. Production export hooks (ONNX/CatBoost export utilities).

This is deployability-oriented engineering, not regulatory clearance.

---

## 3) Core Feature Coverage (Explicit Verification)

| Capability | Status | Evidence | Notes |
|---|---|---|---|
| Full system pipeline | Yes | `pipelines/run_full_pipeline.py` | End-to-end orchestration is implemented. |
| Real-time capability | Partial-Yes | `dashboard/app.py` | Supports rapid triage refresh from available data, but true live streaming ingestion is not yet implemented. |
| Deployability | Yes (engineering) | `modules/deployment/exporter.py`, run scripts | Artifacts and export hooks exist; clinical production governance is outside scope. |
| Strong clinical model | Yes | `models/catboost_model.py`, `models/tcn_transformer.py` | Uses robust tabular + temporal modeling and calibrated evaluation metrics. |
| Temporal deep learning | Yes | `models/tcn_transformer.py` | Sequence-aware model captures trend dynamics over time windows. |
| Multimodal capability | Yes (structured multimodal) | feature pipeline + sequence/static split | Combines temporal vitals/labs + static/categorical context + derived clinical scores. |
| Explainable AI | Yes | `modules/xai/explainer.py` | SHAP outputs are generated; Captum support exists when deep path output is available. |
| Ensembling | Yes | `models/ensemble.py` | Weighted blend of CatBoost and deep model probabilities. |

### Missing pieces and how to add them
1. True streaming real-time ingestion is not present. Add Kafka/FHIR listener and online inference service.
2. Unstructured modalities (clinical notes, imaging) are not integrated. Add NLP and imaging encoders with aligned patient-time keys.

---

## 4) Technology Explanations (Gold Spoon Level)

| Term | What it means in simple words | Why it is used here | How it works in this project |
|---|---|---|---|
| Pipeline | A repeatable chain of steps from raw data to final output. | Prevents manual, error-prone experiments. | `run_full_pipeline.py` coordinates data, training, modules, saving, and checks. |
| Temporal model | A model that sees order and trend over time, not only one row. | Deterioration is trajectory-based. | `TCNTransformerModel` consumes episode windows from sequence prep. |
| CatBoost | A gradient boosting model strong on tabular data with mixed feature types. | Provides reliable baseline and often high precision-recall behavior on engineered features. | Trained on feature matrix and validated against holdout split. |
| Ensemble | Combining multiple model outputs into one final score. | Reduces single-model brittleness and can improve robustness. | Weighted blend of CatBoost and deep probabilities in `models/ensemble.py`. |
| SSL (Self-Supervised Learning) | Pretraining without labels by learning internal structure first. | Helps temporal encoder learn physiological patterns before supervised objective. | Optional masked-prediction pretraining in `modules/ssl/pretrain.py`. |
| FL (Federated Learning simulation) | Training in multiple local slices and aggregating model updates. | Tests robustness to distributed/non-IID style data settings. | Optional FedAvg simulation in `modules/federated_learning/simulation.py`. |
| DG (Domain Generalization) | Testing whether model generalizes across domains. | Clinical settings differ by unit/site/data regime. | Leave-One-Domain-Out evaluation in `modules/domain_generalization/lodo.py`. |
| SHAP | Explains which input features influenced a prediction most. | Clinicians need transparent signals, not a black box score only. | Top-feature CSV and SHAP plot outputs under run artifacts. |
| Captum | Deep model attribution toolkit for time/feature influence analysis. | Adds deep-model interpretability. | Temporal heatmap generation in XAI module when deep outputs are available. |
| PR-AUC | Precision-Recall area under curve. | Better for imbalanced clinical events than plain accuracy. | Core metric tracked for CatBoost, deep model, and ensemble. |
| ROC-AUC | Probability the model ranks positives above negatives. | Useful global separability metric. | Logged per model in run `metrics.json`. |
| Brier score | Measures probability calibration quality. | Important when risk percentage is used for action planning. | Reported in comparison outputs and result tables. |
| Multimodal (here) | Multiple structured information channels together. | Single-channel views can miss risk context. | Uses temporal sequences + static/categorical + engineered clinical scores. |

---

## 5) Model Details

### Models used
1. Temporal deep model: `TCNTransformerModel`.
2. Tabular supervised model: `CatBoostWrapper`.
3. Final predictor: weighted ensemble.

### Why this combination
1. Temporal branch captures progression patterns.
2. CatBoost branch captures strong non-linear interactions in engineered tabular space.
3. Ensemble leverages both strengths and mitigates single-branch failure modes.

### Training approach
1. Build episode-aware train and validation split.
2. Engineer features and prepare sequences.
3. Optional SSL pretraining/reuse.
4. Train deep model and CatBoost.
5. Fit ensemble on validation-aligned predictions.
6. Optionally run FL, DG, and XAI modules.
7. Save and verify artifacts.

### Assumptions
1. Input schema and feature engineering contracts remain consistent with training.
2. Evaluation split is representative of deployment conditions.
3. Risk score is decision support, not autonomous diagnosis.

---

## 6) Performance And Evaluation

### Metrics tracked
1. PR-AUC.
2. ROC-AUC.
3. Optional additional metrics (for comparisons and reports).

### Why these matter clinically
1. PR-AUC emphasizes positive event detection quality under class imbalance.
2. ROC-AUC measures ranking quality across thresholds.
3. Calibration-oriented metrics support safer action thresholds.

### Latest evidence snapshot
From `artifacts/evidence_latest_run.json`:
1. Deep learning PR-AUC: 0.6145
2. CatBoost PR-AUC: 0.7346
3. Ensemble PR-AUC: 0.7389
4. Deep learning ROC-AUC: 0.9253
5. CatBoost ROC-AUC: 0.9641
6. Ensemble ROC-AUC: 0.9642

### Strengths
1. Ensemble improves over individual deep branch and slightly over CatBoost branch on PR-AUC in latest evidence.
2. End-to-end runs emit auditable outputs and hashes.

### Limitations
1. Metric quality can vary by run profile and data regime.
2. Current dashboard risk generation path should be aligned to deterministic saved predictions for strict evidence fidelity.

---

## 7) Clinical Relevance And Interpretation

### What output means to a doctor
The model outputs a risk probability (0 to 1) representing relative likelihood of deterioration in the forecast horizon.

### Practical interpretation
1. Higher score means higher priority for review, not certainty of harm.
2. Score should be interpreted with vitals, NEWS/MEWS/qSOFA context, and bedside judgment.
3. Explainability outputs indicate top contributing features for transparency.

### Example triage framing
Dashboard logic includes action-oriented tiers and threshold cues (for example Immediate/Urgent/Watch/Routine paths) to support rapid queueing.

### Clinical safety boundary
1. This is a decision-support tool.
2. It should not replace clinician judgment.
3. Escalation actions must follow institutional protocols.

---

## 8) Setup And Run Instructions (Beginner-Proof)

### Prerequisites
1. Windows PowerShell or equivalent shell.
2. Python 3.10.
3. Dataset files available in `dataset/`.

### Step-by-step

1. Open terminal at repository root.

```powershell
cd "d:\nEXU2.0 TRY\Patient-Deterioration-Prediction-System"
```

2. Install dependencies.

```powershell
py -3.10 -m pip install -r requirements.txt
```

3. Run smoke test pipeline.

```powershell
py -3.10 pipelines/run_full_pipeline.py --config configs/quick_test.yaml
```

4. Run full profile.

```powershell
py -3.10 pipelines/run_full_pipeline.py --config configs/default.yaml
```

5. Audit run outputs.

```powershell
py -3.10 pipelines/audit_artifacts.py --artifacts-dir artifacts
```

6. Export latest run evidence summary.

```powershell
py -3.10 pipelines/export_run_evidence.py --artifacts-dir artifacts --out artifacts/evidence_latest_run.json
```

7. Launch dashboard.

```powershell
py -3.10 -m streamlit run dashboard/app.py
```

### Expected outputs after successful full run
In latest `artifacts/run_*/` folder:
1. `model/dl_model_final.pt`
2. `model/model.cbm`
3. `model/scaler.pkl`
4. `model/ensemble.pkl`
5. `model/feature_columns.json`
6. `metrics.json`
7. `predictions.csv`
8. `ssl_pretrained_tcntransformer.pt` (when SSL enabled or reused)

Optional module files (when modules enabled):
1. `fl_rounds_history.json`
2. `lodo_results.csv`
3. `top_features.csv`
4. `shap_summary.png`
5. `captum_temporal_heatmap.png`

---

## 9) UI And User Workflow

The Streamlit app in `dashboard/app.py` is built for evaluation and triage walkthrough.

### Typical workflow
1. Open Ward Overview for global risk distribution.
2. Open Clinical Action Board for prioritized queue and suggested urgency tier.
3. Open Patient Deep-Dive to review timeline trends and latest risk.
4. Open Explainability views for SHAP/Captum evidence when available.
5. Review FL and DG pages when module outputs exist.

### What users see
1. Risk segmentation and patient counts.
2. Ranked patient list with urgency badges.
3. Patient trajectory charts and key bedside signals.
4. Explainability artifacts from latest complete run.

---

## 10) Limitations And Future Work

### Current limitations
1. Real-time ingestion is not continuous streaming yet (batch/snapshot oriented).
2. Dashboard prediction path should be tightened to deterministic run predictions for strict evidence fidelity.
3. One optional deep model selection path requires interface alignment before broad switching across model types.
4. Clinical deployment governance (prospective validation, protocol integration, regulation) is outside competition scope.
5. Unstructured modalities (notes, imaging) are not integrated yet.

### Immediate improvements
1. Make dashboard consume saved run predictions directly.
2. Unify artifact contract definitions across runtime, docs, and packages.
3. Remove machine-specific script assumptions for better portability.

### Next-phase improvements
1. Add streaming ingestion and online inference service.
2. Add calibration dashboards and drift monitoring.
3. Add prospective workflow study with clinician feedback loops.

---

## Additional Competition Documents

1. 3-minute judge script: `docs/JUDGE_3MIN_SCRIPT.md`
2. Clinical sanity check: `docs/CLINICAL_SANITY_CHECK.md`
3. System completeness check: `docs/SYSTEM_COMPLETENESS_CHECK.md`
4. Submission assets: `submission/`

---

## Important Disclaimer

This system is a clinical decision-support prototype for competition and research contexts. It is not a standalone medical device and does not replace physician judgment.

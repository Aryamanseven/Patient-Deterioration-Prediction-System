# Team ANC-052 Final Submission
## PS-2 Patient Deterioration Prediction System

**Team Lead**: Pranav Sharma
**Team ID**: ANC 52

---

## Slide 1 - Problem Statement & Mission

**Problem**: Clinical deterioration typically emerges as subtle multi-signal drift (e.g., heart rate and respiratory drift) hours before collapse. Delayed detection fundamentally increases ICU transfer risks and ballooning intervention costs. 

**Solution Approach**: 
We developed an end-to-end Patient Deterioration Prediction System designed specifically to triage risk in the next 12 hours based on hourly time-series vitals and demographic data. 

**Why It Stands Out**: 
Our solution does not just retroactively score data or blindly apply deep learning. It uses a clinically-grounded ensemble approach (combining tabular gradient boosting via CatBoost with temporal Deep Learning via TCN-Transformer), built specifically to be robust against class imbalance. It ensures explainability, is reproducible, and produces thresholded alerts for actionable clinical triage.

---

## Slide 2 - Dealing with Class Imbalance & Real Evaluation

**Our Reality**: Our positive class rate (deteriorating patients) is extremely low (approx 5.25%). In this realm, classical accuracy metrics are actively misleading. 

**Why It's Really Valid**: 
- We evaluate on PR-AUC (Precision-Recall Area Under Curve), which heavily penalizes false positives and genuinely measures rare-event detection quality. 
- We employ Brier Score Loss to confirm our model outputs calibrated probabilities, crucial for setting reliable triage thresholds.
- We deliberately built our evaluation structure to mimic clinical conditions where false confidence can cost lives.

---

## Slide 3 - Our Pipeline: From Raw Data to Live Alerts

**Start to Finish Workflow**:
1. **Input Pipeline**: Consumes hourly structured clinical vitals, static demographics, and labs.
2. **Preprocessing & Feature Engineering**: Leakage-safe, augmented with explicit clinical risk indices.
3. **Dual Model Architecture**: 
   - A tabular discriminative branch (CatBoost) to catch immediate feature-interaction triggers.
   - A temporal sequence model branch (TCN-Transformer) designed to capture drift dependencies over hours.
4. **Fusion**: We merge branch outputs via a robust weighted ensemble.
5. **Output**: Calibrated risk probabilities and threshold-based alerts mapped to actionable categories (critical, high, medium, low).

---

## Slide 4 - Essential Modules Behind The Intelligence

Every component in our system was engineered to solve a distinct clinical AI failure mode:

- **Self-Supervised Learning (SSL) module**: Resolves the lack of dense annotations by reusing pretrained temporal encoders on unlabeled vitals.
- **Supervised CatBoost module**: Learns complex nonlinear tabular interactions quickly and robustly.
- **Deep Temporal Module (TCN-Transformer)**: Contextualizes historical drift over the 12-hour window.
- **Ensemble Fusion Module**: Prevents branch-level variance and smooths predictions.
- **XAI (Explainable AI) Module**: Uses SHAP and attention weights to generate feature importance, building physician trust in our black-box risk scores.
- **Federated Learning & Domain Generalization Modules**: Demonstrates our system's scalability and readiness for cross-hospital heterogeneous data sources.

---

## Slide 5 - Training Reproducibility & Model Artifacts

**Engineering Rigor over Hype:**
Our system is 100% reproducible. The full pipeline reliably produces consistent weights and artifacts.

**Generated Artifacts:**
- `model/model.cbm` (CatBoost)
- `model/dl_model_final.pt` (Deep Learning Weights)
- `model/ensemble.pkl` 
- `ssl_pretrained_tcntransformer.pt` 
- Metrics and predictions tracked via checksums in `evidence_latest_run.json`.

This pipeline is transparent, debuggable, and scientifically reproducible.

---

## Slide 6 - Final Evaluated Results

We optimize and judge solely based on the latest valid ensemble, avoiding training noise:

- **Ensemble PR-AUC**: ~0.7389
- **Ensemble ROC-AUC**: ~0.9641
- **Brier Score**: ~0.0237

Our system decisively outperforms standard sequence baselines in raw validation. We explicitly suppress false positives and preserve clinical actionability.

---

## Slide 7 - External Benchmark Rationale

**Why Compare Against the Google TimeSFM Proxy?**
We actively tested our solution head-to-head against external proxy benchmarks modeled on Google’s foundation TimeSFM architecture. 

**Why It Matters**: 
- Benchmarks used exactly aligned labels and the same data splits.
- This represents a strict stress-test to prove that our specialized clinical architecture outperforms generalized sequence modeling baselines for this specific triage objective.

---

## Slide 8 - Benchmark Outcomes & Evidence

**Head-to-Head Proof**:
- **Latest Ensemble PR-AUC**: 0.7144
- **TimeSFM Proxy PR-AUC**: 0.0793
- **Absolute PR-AUC Delta**: +0.6351 (Ensemble outperforms proxy)
- **ROC-AUC Delta**: +0.2980
- **Brier Score Calibration**: +0.2405 better

Our architecture's specialization drastically supersedes vanilla sequence models in rare-event class structures.

---

## Slide 9 - Inference Workflow & Actionability

**Live Execution Pipeline:**
1. Loads our validated ensemble objects and serialized feature schema (`feature_columns.json`, `scaler.pkl`).
2. Generates precise determinist inference. 
3. Identifies threshold breaches (default 0.50 risk score).
4. Categorizes the patient into a Risk Band. 
5. Emits `reproducibility_report.json` to prove the exact same models that were validated are running in production.

This bridges the gap between pure research notebooks and safe operational outputs.

---

## Slide 10 - Final Impact Statement & Ask

**The Value Proposition:**
- Earlier Risk Prioritization Support for ICU resources.
- Highly Calibrated Triage Confidence.
- Transparent, Reproducible Workflow.

**Final Positioning**:
This project represents a complete, robust, and explainable 12-hour Patient Deterioration Early Warning System. It is strictly benchmarked, evidence-backed, and optimized for clinical reality over hypothetical accuracy metrics. 

We proudly present our final clean submission.

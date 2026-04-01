# 3-Minute Judge Script (Competition Version)

Use this script as a clear, no-fluff speaking narrative for judges.

## 0:00 to 0:30 - Problem Hook
"Clinical deterioration is often missed because teams are forced to monitor many patients at once, and early warning is spread across trends, not single numbers. A heart rate of 110 once may be fine, but a trajectory of worsening respiration, lactate, and oxygen saturation over hours is not. Our system was built to surface that risk earlier and rank who needs attention first."

## 0:30 to 1:10 - What We Built
"We built an end-to-end system, not a notebook demo. The pipeline starts from raw CSV data, engineers clinical and temporal features, trains two complementary models, combines them in an ensemble, verifies artifacts, and serves outputs in a triage dashboard.

The two model paths are:
1. A temporal TCN-Transformer for sequence dynamics.
2. A CatBoost model for strong tabular supervision.

Then we blend both outputs with a weighted ensemble."

## 1:10 to 1:55 - Why It Is Strong
"This architecture is strong because each branch solves a different failure mode.

The temporal branch captures progression over time.
The CatBoost branch is highly effective on engineered tabular features.
The ensemble stabilizes prediction quality and reduces single-model brittleness.

In our latest evidence snapshot, the ensemble achieves the strongest PR-AUC among the three branches, which is especially important for imbalanced clinical events."

## 1:55 to 2:30 - Trust, Explainability, and Reproducibility
"A score without trust is not clinically usable. So every run is config-driven, artifact-verified, and exportable as machine-readable evidence.

We provide explainability outputs:
1. SHAP feature attribution for model transparency.
2. Captum temporal attribution when deep-model outputs are available.

We also include optional robustness modules for federated simulation and domain generalization."

## 2:30 to 3:00 - Clinical Impact and Closing
"For a clinician, this system is a prioritization engine: who to review first, why risk is elevated, and how trends evolved over time. It is decision support, not autonomous diagnosis.

Our value is practical and measurable: reproducible execution, transparent evidence, and clinically meaningful risk ranking under real constraints. That is the difference between a model demo and a system you can operationalize."

---

## 20-Second Backup Close
"We did not optimize for a leaderboard screenshot. We optimized for trustworthy execution: temporal modeling, robust tabular baseline, ensemble fusion, explainability, and artifact-level proof. That is why this project is competitive and deployment-oriented."

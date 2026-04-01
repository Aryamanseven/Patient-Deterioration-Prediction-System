# Clinical Sanity Check

## Direct Answer

If a doctor sees this system today, they will partly trust it and partly question it.

Trust comes from:
1. Clear triage framing and urgency tiers in the UI.
2. Explainability outputs (SHAP and optional Captum artifacts).
3. Reproducible run artifacts with metric summaries.

Potential confusion comes from:
1. Risk generation in the current dashboard path should be fully tied to deterministic saved run predictions.
2. Action thresholds can be interpreted as clinical directives unless clearly labeled as support cues.
3. No prospective clinical validation is packaged as bedside outcome evidence yet.

---

## Potential Clinical Issues

### 1) Confusing outputs
Issue:
Risk score and urgency tier can appear authoritative if context is not shown.

Risk:
Over-trust by non-specialist users.

### 2) Interpretability gaps in some runs
Issue:
If Captum output is missing for a given run, users may see partial explanation depth.

Risk:
Perceived inconsistency in explainability.

### 3) Operational realism assumptions
Issue:
Current workflow is run-artifact based, not continuous streaming bedside ingestion.

Risk:
Users may assume true live monitoring when it is snapshot/batch oriented.

### 4) Threshold portability
Issue:
Single threshold strategy may not transfer equally across different hospital populations.

Risk:
Alert burden or under-alerting depending on local prevalence and workflow.

---

## Immediate Fixes (Do Now)

1. Route dashboard risk display directly from deterministic saved prediction artifacts for selected runs.
2. Add explicit UI disclaimer near score cards: "Decision support only, not diagnosis."
3. Show timestamp and run ID on every triage page so users know model context.
4. Add a visible "explanation availability" badge when Captum is absent.
5. Add threshold guidance text: thresholds are operational defaults and must be locally validated.

---

## Improvements With More Time

1. Prospective silent trial in real workflow with clinician feedback.
2. Calibration and drift monitoring dashboards by subgroup and shift.
3. Site-specific threshold tuning protocol with safety committee review.
4. Structured intervention logging to evaluate whether alerts improve outcomes.
5. Formal human factors evaluation of UI clarity and alarm fatigue.

---

## Clinical Positioning Statement

This system is suitable today as an engineering-grade clinical decision-support prototype for triage prioritization and model evaluation. It is not yet ready for independent clinical deployment without local validation, governance, and monitoring.

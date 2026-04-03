# Final Demo Video Script (Master Version)

Project title to speak exactly:
PS-2 Patient Deterioration Prediction System (Team ANC-052)

This script is designed to be complete, evidence-first, and final-round safe.

## Recording setup checklist

1. Open final deck: presentation/AesCodeNexus_Final_Round_Deck.pptx
2. Start app: py -3.10 -m streamlit run final_round_clean_submission/app/app.py
3. Open notebook: notebooks/Final_Round_Reproducible_Notebook.ipynb
4. Keep evidence JSONs open:
   - evidence/evidence_latest_run.json
   - evidence/benchmark_summary.json
5. Keep this script visible on second screen.

## Time-coded script (4:00)

### 0:00 to 0:20 | Opening punch

Voice:
"We are Team ANC-052. This is our PS-2 Patient Deterioration Prediction System. We predict 12-hour clinical deterioration risk with a reproducible, evidence-backed pipeline built for reliable triage support."

Screen:
- Deck title slide.
- Team name and problem statement.

### 0:20 to 0:50 | Clinical problem and metric logic

Voice:
"This task is imbalanced at about 5.25% positives, so raw accuracy is misleading. We optimize and report PR-AUC, ROC-AUC, and Brier score, because early warning quality and calibration matter in clinical action workflows."

Screen:
- App Best Model page metric cards.

### 0:50 to 1:25 | Architecture in one flow

Voice:
"Our final run combines structured feature engineering, SSL reuse for temporal representation, CatBoost for strong tabular discrimination, deep sequence modeling, and a weighted ensemble as final predictor. The run config also includes federated rounds, domain generalization, and XAI hooks for robustness and trust."

Screen:
- App execution profile table.
- Brief deck architecture slide.

### 1:25 to 2:00 | Best model results (only what is final)

Voice:
"Our best final model is the latest ensemble. On the validated run, ensemble PR-AUC is 0.738931 and ROC-AUC is 0.964163. We intentionally center the dashboard on this best model only, to avoid noise and keep final claims precise."

Screen:
- App Best Model page.
- Highlight ensemble metrics.

### 2:00 to 2:35 | Benchmark proof against TimeSFM proxy

Voice:
"For strict head-to-head benchmark evidence, we compare latest ensemble versus TimeSFM proxy on aligned labels. Ensemble PR-AUC is 0.714477 versus 0.079360, with delta +0.635117. ROC-AUC delta is +0.298051, and Brier improves by +0.240594 in our favor."

Screen:
- Open evidence/benchmark_summary.json.
- Show app benchmark deltas.

### 2:35 to 3:10 | Inference behavior and thresholding

Voice:
"Inference is deterministic from saved predictions. At threshold 0.50, we produce alert decisions and inspect true positives, false positives, and risk bands. This turns raw probabilities into operational triage steps."

Screen:
- App Inference Trace page.
- Move threshold once and show TP/FP/TN/FN update.

### 3:10 to 3:40 | Reproducibility proof

Voice:
"Our notebook rebuilds metrics from stored predictions and validates config, evidence integrity, and benchmark consistency. It writes a reproducibility report with explicit pass flags, so anyone can rerun and verify."

Screen:
- Notebook final report cell.
- Show reproducibility_report.json.

### 3:40 to 4:00 | Closing

Voice:
"ANC-052 delivers performance, calibration, and reproducibility in one final package. Reliable by design, explainable by evidence, and submission-ready for final-round evaluation."

Screen:
- Final deck close slide.

## Safety guardrails while speaking

1. Never claim hospital deployment approval.
2. Never claim guaranteed mortality reduction.
3. Speak only values present in evidence files.
4. If interrupted, return to evidence-first statements.

## Single-folder rule

All final assets must come from final_round_clean_submission only.

## 3:30 compressed fallback

If you must shorten live:

1. Keep 0:00 to 0:20 opener unchanged.
2. Merge architecture and results in one 50-second block.
3. Keep benchmark proof and notebook reproducibility segments unchanged.
4. End with 20-second close.

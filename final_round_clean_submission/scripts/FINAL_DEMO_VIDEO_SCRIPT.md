# Final Demo Video Script (3:30)

This is the final submission-grade script for recording a confident, evidence-anchored demo under final-round judging pressure.

## On-screen setup before recording

1. Streamlit app running from final package path.
2. Notebook already opened at final report cell (optional cut-in).
3. benchmark_summary.json ready for one proof shot.
4. Final PPT deck open for intro and close.
5. Keep this script visible on a second screen for timing discipline.

## Script

### 0:00 - 0:20 | Hook + Problem

Voice:
"Clinical deterioration is rarely sudden; it appears as a subtle pattern across noisy hourly vitals and labs. Team ANC-052 built a reliable 12-hour early-warning pipeline that turns those patterns into actionable triage signals with reproducible evidence."

Screen:
- Slide 1 title from final deck.
- Transition to dashboard home page.

### 0:20 - 0:55 | What this system does

Voice:
"This dashboard is powered by our best validated run, auto-loaded from evidence artifacts. Its purpose is focused and practical: help clinicians prioritize risk earlier with measurable confidence, not replace clinical judgment."

Screen:
- Ward overview section.
- Risk distribution and key metric cards.

### 0:55 - 1:35 | Clinical flow demonstration

Voice:
"The workflow is simple and repeatable. First, identify high-risk patients from ward-level distribution. Second, open a patient timeline. Third, review trend context and confidence together before escalation."

Screen:
- Navigate to patient deep-dive page.
- Show risk trajectory and corresponding vital trends.

### 1:35 - 2:05 | Explainability without clutter

Voice:
"Every alert needs justification. The explainability view surfaces the strongest feature drivers so decisions stay transparent and auditable."

Screen:
- Open explainability section in app.
- Pause briefly on top features.

### 2:05 - 2:45 | Benchmark proof (core credibility moment)

Voice:
"Now the core evidence. Our final strict benchmark compares the latest validated ensemble against the TimeSFM proxy on aligned data. From benchmark summary: ensemble PR-AUC is 0.7145 versus 0.0794 for the proxy, with a PR-AUC delta of plus 0.6351. Alignment checks pass with labels_match true."

Screen:
- Open benchmark section in app.
- Optional zoom on evidence/benchmark_summary.json file.

### 2:45 - 3:10 | Reproducibility proof

Voice:
"This is not a one-off run. Our final reproducibility notebook reconstructs the exact validation split, recomputes metrics from predictions, and exports a machine-readable report with pass_all status."

Screen:
- Show notebook final cell output with PASS true.
- Briefly show reproducibility_report.json.

### 3:10 - 3:30 | Close

Voice:
"ANC-052 delivers what clinical AI needs most: measurable performance, strict reproducibility, and a clean path from model output to actionable triage. Reliable, explainable, and final-round ready."

Screen:
- Final slide from deck.
- End on project/team name.

## Voice and delivery notes

1. Keep speed calm and authoritative.
2. Do not claim deployment approval or clinical replacement.
3. Use exact numbers only from benchmark_summary.json and reproducibility_report.json.
4. Keep all claims tied to files in final_round_clean_submission/evidence.
5. If you miss a line, return to evidence-first messaging, not improvised claims.

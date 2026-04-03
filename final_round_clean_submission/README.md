# Final Round Clean Submission

This is the single active folder for final-round submission and demo assets.
Use this folder only.

Project title:
PS-2 Patient Deterioration Prediction System (Team ANC-052)

## Package layout

1. app/
   - app.py (final-mode dashboard; best-run evidence only)
   - README.md (dashboard runtime notes)

2. notebooks/
   - Final_Round_Reproducible_Notebook.ipynb (from-scratch reproducibility notebook)
   - reproducibility_report.json (generated when notebook is executed)

3. evidence/
   - evidence_latest_run.json (best run pointer and metrics snapshot)
   - final_benchmark_latest.json (final benchmark pointer)
   - benchmark_summary.json (strict final benchmark summary)
   - benchmark_full_sample_metrics.csv
   - benchmark_subsample_summary.csv
   - benchmark_winner_by_fraction.csv
   - run_config_stage2_fl10.yaml (final module configuration)

4. presentation/
   - AesCodeNexus_Final_Round_Deck.pptx (active final deck)
   - TEN_SLIDE_SUBMISSION_CONTENT.md (detailed 10-slide content)

5. scripts/
   - FINAL_DEMO_VIDEO_SCRIPT.md (final presentation and demo narration)
   - DEMO_DAY_RUN_ORDER.md (time-bound execution order)
   - PORTAL_SUBMISSION_CHECKLIST.md (submission lock checklist)
   - JUDGE_QA_DEFENSE_CHEATSHEET.md (judge Q&A defense)

## Reviewer quick path

1. Read SUBMISSION.md for final portal-ready mapping.
2. Open presentation/TEN_SLIDE_SUBMISSION_CONTENT.md and finalize exact 10-slide narrative.
3. Open notebooks/Final_Round_Reproducible_Notebook.ipynb and run all cells.
4. Confirm notebooks/reproducibility_report.json has overall_pass = true.
5. Open evidence/benchmark_summary.json and verify benchmark deltas.
6. Open evidence/run_config_stage2_fl10.yaml for full module coverage.
7. Run app/app.py with Streamlit for the live demo.
8. Use scripts/FINAL_DEMO_VIDEO_SCRIPT.md for the final recording.

## Runtime commands

From repository root:

```powershell
py -3.10 -m streamlit run final_round_clean_submission/app/app.py
```

Notebook execution is expected with Python 3.10.

## Submission readiness checklist

1. Notebook link is public (Kaggle or Colab).
2. Deck has max 10 slides (active deck has 10 slides).
3. Demo video follows the scripted timing and includes benchmark evidence.
4. Claims in speech match evidence files exactly.

## Teammate handoff (roommate-friendly)

1. This folder is the single source of truth to continue submission work from GitHub.
2. If your teammate opens the repository website, direct them to final_round_clean_submission/README.md first.
3. Then follow final_round_clean_submission/scripts/PORTAL_SUBMISSION_CHECKLIST.md step by step.
4. Do not use archived or older submission folders for final portal links.

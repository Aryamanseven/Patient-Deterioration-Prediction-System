# Pipelines

This folder contains orchestration scripts for end-to-end training and artifact verification.

## Files

1. run_full_pipeline.py
	Main orchestrator for data loading, model training, optional modules, and hardened saving.

2. audit_artifacts.py
	Non-destructive checker for required artifact completeness.

3. run_submission_pipeline.ps1
	Python 3.10 helper script to install requirements, run full training, and audit outputs.

4. overnight_supervisor.ps1
	Optional internal launcher/monitor script for unattended execution.

5. run_final_verified.ps1
	Single-command full replay runner with enforced audit and machine-readable summary logs.

6. build_best_overall_package.py
	Creates a clean judge-facing package with latest complete run, logs, evidence, notebooks, docs, and manifest.

7. run_benchmark_suite.py
	Runs complete benchmark sweep across run-level predictions and flat artifact CSVs, then writes structured output with per-folder README files.

8. run_timesfm_vs_latest_ensemble_subsample.py
	Runs strict two-model benchmarking between latest ensemble predictions and freshly generated TimeSFM proxy scores on the same validation split, including stratified subsample analysis.

9. finalize_benchmark_artifacts.py
	One-shot finalizer that runs the strict benchmark, builds a single cleaned final benchmark package, and removes old benchmark clutter from artifacts/.

## Execution order in run_full_pipeline.py

1. Data loading and feature engineering
2. SSL pretrain or SSL reuse
3. DL training (with optional DP wrapper)
4. FL simulation (if enabled)
5. CatBoost training
6. Ensemble fitting
7. DG LODO evaluation (if enabled)
8. XAI generation (if enabled)
9. Hardened artifact save verification

If required artifacts are missing, the run exits with failure instead of reporting false success.

## Recommended operational flow

1. Run quick smoke test:
	py -3.10 pipelines/run_full_pipeline.py --config configs/quick_test.yaml

2. Run full submission pipeline:
	py -3.10 pipelines/run_full_pipeline.py --config configs/default.yaml

3. Audit:
	py -3.10 pipelines/audit_artifacts.py --artifacts-dir artifacts

4. One-command helper (Windows PowerShell):
	./pipelines/run_submission_pipeline.ps1 -ConfigPath configs/default.yaml

5. Final verified full replay (Windows PowerShell):
	./pipelines/run_final_verified.ps1 -ConfigPath configs/final_full_replay.yaml

6. Build best-overall package:
	py -3.10 pipelines/build_best_overall_package.py

7. Run benchmark suite:
	py -3.10 pipelines/run_benchmark_suite.py --n-bootstrap 400 --seed 42

8. Run strict TimeSFM vs latest ensemble subsample benchmark:
	py -3.10 pipelines/run_timesfm_vs_latest_ensemble_subsample.py --max-eval-rows 10000

9. Run final benchmark packager + cleanup (recommended final step):
	py -3.10 pipelines/finalize_benchmark_artifacts.py --max-eval-rows 10000 --final-dir-name benchmark_final

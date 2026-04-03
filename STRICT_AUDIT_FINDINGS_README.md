# Strict Non-Destructive Dependency Audit (Master Report)

Date: 2026-04-01
Scope: Full consolidated findings across previously audited folders and this final attached chunk.
Method: Read-only analysis, dependency tracing across repository, no runtime refactor/editing during audit.

---

## Folder: .pytest_cache

### 1. PURPOSE
- Stores pytest cache metadata for local test acceleration (`--lf`, `--ff`).
- Not required for training, inference, packaging, or dashboard runtime.

### 2. STRUCTURE QUALITY
- Standard pytest-generated structure.
- Not scalable/system-facing by design; purely local cache.

### 3. FILE ANALYSIS
- `README.md`: standard cache explanation; needed only for local context.
- `CACHEDIR.TAG`, `v/*`: cache internals, non-runtime.
- `.gitignore` inside cache: standard marker, non-runtime.

### 4. DEPENDENCY CHECK
- No code imports this folder.
- Repository root `.gitignore` excludes `.pytest_cache/`.
- Removing it does not break pipeline stages.

### 5. README / DOCUMENTATION
- Has README; accurate and sufficient for purpose.

### 6. ISSUES
- CRITICAL:
1. None.
- IMPORTANT:
1. None.
- MINOR:
1. None.

### 7. VERDICT
- KEEP (ignored cache; non-impactful).

---

## Folder: archive

### 1. PURPOSE
- Historical quarantine: failed runs, legacy outputs, runtime log dumps.
- Useful for forensics only, not active evidence path.

### 2. STRUCTURE QUALITY
- Naming is clear (`runs_failed`, `runtime_logs`, `legacy_outputs`).
- Scalable enough for archival, but currently mixed encodings and stale context make it noisy.

### 3. FILE ANALYSIS
- `README.md`: correct policy intent.
- `runs_failed/run_*/config.yaml`: useful for postmortem diffing.
- `runs_failed/run_*/logs/pipeline.log`: many failed runs stop near step start; confirms incompleteness.
- `runtime_logs/error.txt`: Python313/Numpy experimental warnings; environment contamination evidence.
- `runtime_logs/pipeline_error.txt`, `runtime_logs/error.log`: UTF-16/byte-like dumps; hard to parse directly.
- `legacy_outputs/**/*`: old experiments; not current pipeline source of truth.

### 4. DEPENDENCY CHECK
- No active runtime imports or reads from archive by pipeline/dashboard.
- Mentioned only in policy/docs and ignore rules.
- Removing archive does not break active pipeline stages.

### 5. README / DOCUMENTATION
- Has README and clear policy.
- Accurate about non-current use.

### 6. ISSUES
- CRITICAL:
1. None for active runtime.
- IMPORTANT:
1. Archived logs include environment drift indicators (Python313 warnings) that can confuse reproducibility narrative.
- MINOR:
1. UTF-16/binary-like archived logs reduce debuggability.

### 7. VERDICT
- KEEP (archive-only). Do not use as submission evidence.

---

## Folder: artifacts

### 1. PURPOSE
- Canonical runtime output/evidence store for completed runs.
- Required by dashboard, audit/export scripts, and package builders.

### 2. STRUCTURE QUALITY
- Run-based structure (`run_*`) is correct and scalable.
- Root-level legacy files (`metadata.json`, old model files) coexist with run-based artifacts and create ambiguity.

### 3. FILE ANALYSIS
- `README.md`: mostly accurate but contract drifts with other components.
- `evidence_latest_run.json`: high-value machine-readable proof summary.
- `metadata.json`: stale legacy pointer model metadata (`artifacts/deterioration_model.cbm`), not consumed by active code.
- `run_20260401_025010_...`: complete core run assets present.
- `run_20260401_095900_...`: strongest complete stage2 run; core + FL + DG + SHAP outputs present.
- `run_20260401_095900_.../module_outputs/`: empty directory (created but unused).

### 4. DEPENDENCY CHECK
- Written by `core.config.get_output_dir` and `pipelines/run_full_pipeline.py`.
- Read by dashboard (`Path("artifacts")`) and scripts (`audit_artifacts.py`, `export_run_evidence.py`, package builder).
- Removing/changing structure breaks pipeline evidence flow and dashboard run selection.

### 5. README / DOCUMENTATION
- README exists and is mostly useful.
- Accuracy gap: states advanced outputs include Captum; latest complete stage2 run has no Captum file despite XAI enabled.

### 6. ISSUES
- CRITICAL:
1. Artifact contract mismatch across files (7-core vs 8-core definition with SSL).
- IMPORTANT:
1. Legacy root metadata is stale and not pipeline-aligned.
2. `module_outputs/` is provisioned but unused by current modules.
- MINOR:
1. Mixed historical files at root increase evaluator confusion.

### 7. VERDICT
- KEEP, REFACTOR artifact contract and root-level legacy cleanup policy.

---

## Folder: catboost_info

### 1. PURPOSE
- CatBoost generated training diagnostics (loss curves, timing, logs).
- Optional debugging telemetry.

### 2. STRUCTURE QUALITY
- Standard CatBoost output layout.
- Appropriate for diagnostics; not intended as durable artifact API.

### 3. FILE ANALYSIS
- `README.md`: accurate (safe to delete, not required for final inference artifacts).
- `catboost_training.json`, `learn_error.tsv`, `test_error.tsv`, `time_left.tsv`: valid telemetry files.
- `learn/`, `test/`, `tmp/`: generated internal folders.

### 4. DEPENDENCY CHECK
- Not imported/used by pipeline runtime logic.
- Ignored by root `.gitignore`.
- Removing it does not break pipeline outputs.

### 5. README / DOCUMENTATION
- Has README, accurate and sufficient.

### 6. ISSUES
- CRITICAL:
1. None.
- IMPORTANT:
1. None.
- MINOR:
1. Large telemetry files may inflate local disk and search noise.

### 7. VERDICT
- KEEP OPTIONAL (diagnostic-only).

---

## Folder: configs

### 1. PURPOSE
- Central config profiles controlling runs, modules, and hyperparameters.
- Required for reproducible runs.

### 2. STRUCTURE QUALITY
- Good profile naming (`quick_test`, `default`, `recovery`, `recovery_fl_stage2_10rounds`, `final_full_replay`).
- Scalable profile approach.

### 3. FILE ANALYSIS
- `default.yaml`: full stack profile; valid.
- `quick_test.yaml`: smoke profile; valid.
- `recovery.yaml`: staged reuse profile; valid.
- `recovery_fl_stage2_10rounds.yaml`: stage2 FL profile; valid.
- `final_full_replay.yaml`: verifier profile; valid.
- `README.md`: useful but does not call out output.base_dir runtime mismatch explicitly.

### 4. DEPENDENCY CHECK
- Loaded by `load_config` in pipeline; required for startup.
- Scripts and tests pass config paths to pipeline.
- Removing/changing keys breaks run orchestration.

### 5. README / DOCUMENTATION
- Has README; mostly accurate and helpful.
- Accuracy gap: configs expose `output.base_dir`, but runtime ignores it in output resolver.

### 6. ISSUES
- CRITICAL:
1. Config claims output path configurability; implementation hardcodes artifacts path.
- IMPORTANT:
1. Module enabled flags are not fully honored in orchestrator flow for deep/supervised/ensemble invocation behavior.
- MINOR:
1. Documentation does not clearly separate mandatory vs optional module combinations.

### 7. VERDICT
- KEEP, REFACTOR semantics alignment with runtime.

---

## Folder: core

### 1. PURPOSE
- Foundational utilities: config, feature engineering, loading, metrics, reproducibility, logging.
- Required by all runtime paths.

### 2. STRUCTURE QUALITY
- Cohesive file split and clear naming.
- Good modular boundaries.

### 3. FILE ANALYSIS
- `config.py`: central validator + output dir resolver; critical.
- `data_loader.py`: data load/split/sequence dataset; critical.
- `features.py`: full feature construction contract; critical.
- `clinical_scores.py`: clinical score utilities; used by features and dashboard context.
- `metrics.py`: centralized metric calculations; good.
- `logger.py`: centralized logging; good.
- `reproducibility.py`: seed setup; useful.
- `README.md`: mostly accurate.

### 4. DEPENDENCY CHECK
- Imported by models/modules/pipeline/dashboard/tests.
- Removing any major file breaks training path.

### 5. README / DOCUMENTATION
- Has README and invariants.
- Minor drift from implementation nuances.

### 6. ISSUES
- CRITICAL:
1. `output.base_dir` not applied by `get_output_dir`.
- IMPORTANT:
1. `config.validate_config` hard-fails missing dataset while data loader has synthetic fallback logic.
- MINOR:
1. Comment/doc string overstates no hardcoded values.

### 7. VERDICT
- KEEP, REFACTOR consistency edges.

---

## Folder: dashboard

### 1. PURPOSE
- Streamlit evaluator UI for run selection, triage views, and evidence page.
- Important for judge-facing demonstration.

### 2. STRUCTURE QUALITY
- Single-file app; manageable now but scaling pressure exists.
- Naming mostly clear.

### 3. FILE ANALYSIS
- `app.py`: functional but has evidence-faithfulness issues.
- `README.md`: useful, partially stale vs implementation details.

### 4. DEPENDENCY CHECK
- Reads artifacts run folders and docs feedback snapshot.
- Imports core feature utilities.
- Removing breaks demo path but not training runtime.

### 5. README / DOCUMENTATION
- Has README; moderate quality.
- Accuracy gap: app docstring says reads outputs folder, runtime reads artifacts folder.

### 6. ISSUES
- CRITICAL:
1. Risk scoring in dashboard uses synthetic random DL blend/fallback instead of deterministic run predictions.
- IMPORTANT:
1. Run completeness criteria checks only metrics + predictions; does not enforce full artifact contract.
- MINOR:
1. Unused imports (`add_episode_ids`, `compute_scores_vectorized`).

### 7. VERDICT
- KEEP, REFACTOR for deterministic evidence integrity.

---

## Folder: dataset

### 1. PURPOSE
- Source data (`train.csv`) and unlabeled validation (`val_no_labels.csv`).
- Essential to training and submission workflows.

### 2. STRUCTURE QUALITY
- Minimal and clear.
- Naming consistent.

### 3. FILE ANALYSIS
- `train.csv`: required by all configs.
- `val_no_labels.csv`: submission/inference path support.
- `README.md`: clear policy and constraints.

### 4. DEPENDENCY CHECK
- Pipeline config paths target `dataset/train.csv`.
- Dashboard samples from `dataset/train.csv` for visualization.
- Removing data files breaks pipeline start.

### 5. README / DOCUMENTATION
- Has README; accurate and sufficient.

### 6. ISSUES
- CRITICAL:
1. None.
- IMPORTANT:
1. No dataset version/hash control in runtime checks.
- MINOR:
1. None.

### 7. VERDICT
- KEEP.

---

## Folder: docs

### 1. PURPOSE
- Technical and evaluator-facing narrative docs.
- Required for judge comprehension, not runtime execution.

### 2. STRUCTURE QUALITY
- Rich and comprehensive.
- Some overlap and drift across narrative docs.

### 3. FILE ANALYSIS
- Core docs (`ARCHITECTURE.md`, `MODULE_CONNECTIONS.md`, `TECHNICAL_REPORT.md`, `LIMITATIONS.md`, `README.md`) are high-value.
- Strategy docs (`ANC052_*`, `COMPETITION_*`, `BEST_OVERALL_NEXT_STEPS.md`) are useful for submission framing.
- Several claims are stronger than current implementation guarantees.

### 4. DEPENDENCY CHECK
- Dashboard reads `COMPETITION_FEEDBACK_SNAPSHOT.md`.
- Package builders copy docs into submission bundles.
- Removing docs does not break training but harms evaluator pipeline.

### 5. README / DOCUMENTATION
- Strong top-level docs README exists.
- Accuracy gaps in module I/O and some limitations claims.

### 6. ISSUES
- CRITICAL:
1. None runtime-critical.
- IMPORTANT:
1. `MODULE_CONNECTIONS.md` claims file I/O heavily isolated to deployment only; false (SSL/FL/DG/XAI write files).
2. `LIMITATIONS.md` hard-coded categorical embedding claim does not match current TCN numeric sequence pipeline.
- MINOR:
1. Repeated narrative content across package copies.

### 7. VERDICT
- KEEP, REFACTOR claim accuracy.

---

## Folder: models

### 1. PURPOSE
- Implements trainable model wrappers and registry for supervised/deep/ensemble.
- Required in core pipeline stages.

### 2. STRUCTURE QUALITY
- Clean naming and modular split.
- Interface consistency is partial, not complete.

### 3. FILE ANALYSIS
- `model_registry.py`: required, but exposes model variants with incompatible fit signatures relative to orchestrator assumptions.
- `catboost_model.py`: required; good wrapper; has unused import `os`.
- `tcn_transformer.py`: required and primary deep path; robust handling present; has duplicate `math` import style smell.
- `lstm_attention.py`: optional baseline; currently not orchestration-compatible with dataset-based fit call pattern.
- `ensemble.py`: required by pipeline; simple and effective for 2-model blend.
- `__init__.py`: acceptable package export surface.
- `README.md`: generally useful; shared interface claim overstates real compatibility.

### 4. DEPENDENCY CHECK
- Pipeline imports model registry + ensemble directly.
- Modules import specific model classes (`TCNTransformerNetwork`, `CatBoostWrapper`).
- Changing signatures breaks pipeline and module integrations.

### 5. README / DOCUMENTATION
- Has README and contract section.
- Accuracy gap on strict shared interface compatibility.

### 6. ISSUES
- CRITICAL:
1. Registry exposes `lstm_attention` path but pipeline fit invocation assumes TCN-style dataset wrapper API; selecting LSTM can break.
- IMPORTANT:
1. Unused imports and minor dead code patterns (`os`, top-level `math` in LSTM).
- MINOR:
1. Interface contract docs too generic for real fit signature divergence.

### 7. VERDICT
- KEEP, REFACTOR interface consistency.

---

## Folder: modules

### 1. PURPOSE
- Optional advanced capabilities: SSL, FL, DG, XAI, DP, deployment.
- Used by pipeline for extended run modes.

### 2. STRUCTURE QUALITY
- Clear submodule partitioning and naming.
- Good scalability pattern.

### 3. FILE ANALYSIS
- `README.md`: useful high-level map.
- `deployment/exporter.py`: useful export utility; writes ONNX and CatBoost artifact copies.
- `deployment/__init__.py`: clean.
- `differential_privacy/optimizer.py`: functional DP hook; robust fallback when Opacus absent.
- `differential_privacy/__init__.py`: clean.
- `domain_generalization/lodo.py`: useful DG evaluation and export.
- `domain_generalization/augmentation.py`: used by TCN training when DG enabled; good.
- `domain_generalization/__init__.py`: clean.
- `federated_learning/simulation.py`: useful FL simulation, includes best-round restoration.
- `federated_learning/__init__.py`: clean.
- `ssl/pretrain.py`: required for SSL pretraining/reuse output; has unused `numpy` import.
- `ssl/__init__.py`: clean.
- `xai/explainer.py`: SHAP and Captum generation; has unused `warnings` import.
- `xai/__init__.py`: exports only `run_xai_analysis`, not `run_captum_analysis` (API asymmetry).
- Submodule READMEs: mostly accurate; xai README under-specifies Captum output artifact.

### 4. DEPENDENCY CHECK
- Pipeline imports all major module entry points.
- TCN model dynamically imports DP and DG augmentation internals.
- Removing module subfolders breaks optional pipeline stages and some default profiles.

### 5. README / DOCUMENTATION
- Module-level and submodule READMEs exist.
- Generally good; several details are incomplete (e.g., Captum output expectations).

### 6. ISSUES
- CRITICAL:
1. None in core training path when modules enabled as currently used.
- IMPORTANT:
1. `pipelines/run_full_pipeline.py` imports `apply_differential_privacy` from modules but never uses that import directly.
2. `xai/__init__.py` incomplete export surface vs actual pipeline usage.
- MINOR:
1. Unused imports in `ssl/pretrain.py` and `xai/explainer.py`.

### 7. VERDICT
- KEEP, REFACTOR minor API/documentation hygiene.

---

## Folder: outputs

### 1. PURPOSE
- Intended as lightweight smoke-test output location placeholder.
- Not used by active runtime output writer.

### 2. STRUCTURE QUALITY
- Minimal (README only).
- Current state indicates conceptual drift/dead placeholder.

### 3. FILE ANALYSIS
- `README.md`: documents `outputs/quick_test`, but folder currently has only README.

### 4. DEPENDENCY CHECK
- No active Python runtime writes to `outputs/`.
- Code writes runs to `artifacts/run_*` regardless of config `output.base_dir`.
- Removing `outputs` folder does not break active pipeline.

### 5. README / DOCUMENTATION
- Has README, but operationally stale vs implementation.

### 6. ISSUES
- CRITICAL:
1. None runtime-critical.
- IMPORTANT:
1. Folder intent conflicts with actual output path behavior, creating reproducibility confusion.
- MINOR:
1. Placeholder folder with no active artifacts.

### 7. VERDICT
- REFACTOR (or REMOVE after path semantics are unified).

---

## Folder: pipelines

### 1. PURPOSE
- End-to-end orchestration, auditing, evidence export, and automation scripts.
- Critical for full execution and packaging.

### 2. STRUCTURE QUALITY
- Good separation of concerns by script.
- Some script portability constraints and stale parameter expectations.

### 3. FILE ANALYSIS
- `run_full_pipeline.py`: core orchestrator; robust save checks; key central dependency node.
- `audit_artifacts.py`: non-destructive completeness checker; useful.
- `export_run_evidence.py`: machine-readable evidence export; useful.
- `build_best_overall_package.py`: deterministic package builder; useful.
- `run_submission_pipeline.ps1`: simple reproducible launcher; acceptable.
- `run_final_verified.ps1`: useful flow but hardcoded Python path and no `StopOtherPipelineRuns` switch support.
- `overnight_supervisor.ps1`: stage orchestration useful but hardcoded Python/root paths and analyzer warnings.
- `README.md`: good process overview.
- `__init__.py`: placeholder, acceptable.

### 4. DEPENDENCY CHECK
- Invoked by tests, docs commands, notebooks, and packaging flow.
- Imports core/models/modules extensively.
- Modifying/removing scripts breaks reproducibility automation and evaluator workflow.

### 5. README / DOCUMENTATION
- Has README and execution order.
- Mostly accurate.

### 6. ISSUES
- CRITICAL:
1. Pipeline logic not fully config-flag-driven for deep/supervised/ensemble stage invocation.
- IMPORTANT:
1. Hardcoded machine-specific paths in `run_final_verified.ps1` and `overnight_supervisor.ps1`.
2. User-invoked parameter mismatch (`-StopOtherPipelineRuns`) fails because script does not define it.
3. Unused import `apply_differential_privacy` in orchestrator.
- MINOR:
1. PowerShell lint warnings (unapproved verb) in supervisor helper.

### 7. VERDICT
- KEEP, REFACTOR automation portability + config semantics.

---

## Folder: submission

### 1. PURPOSE
- Competition-facing deliverables (notebooks, guides, pitch, CSV outputs).
- Important for evaluator submission path.

### 2. STRUCTURE QUALITY
- Clear naming and consistent artifacts.
- Mixes executable notebooks and static outputs as expected for deliverables.

### 3. FILE ANALYSIS
- `Patient_Deterioration_Week1_Official_Submission_Notebook.ipynb`: primary submission notebook; required.
- `Best_Overall_Submission_Walkthrough.ipynb`: concise judge walkthrough; required for current package narrative.
- `Reproducible_EndToEnd_Runbook.ipynb`: reproducibility runner notebook; useful.
- `NOTEBOOK_SUBMISSION_GUIDE.md`: useful but references `submission/ps2_submission_predictions.csv` not currently present.
- `PITCH_SCRIPT.md`: strong constraints-aware narrative; acceptable.
- `week1_official_submission_results.csv`: static result summary; useful.
- `official_winner_reproduced_metrics.csv`: useful comparison artifact.
- `focused_subsample_lr0048_iter1450_official_submission_predictions.csv`: preserved model-search evidence output.
- `README.md`: clear folder guidance.

### 4. DEPENDENCY CHECK
- Referenced by root README/docs and package builder.
- Not required by training runtime.
- Removing weakens submission workflow and package completeness.

### 5. README / DOCUMENTATION
- Has README and guide.
- Guide generally clear but has output filename drift.

### 6. ISSUES
- CRITICAL:
1. None runtime-critical.
- IMPORTANT:
1. Submission guide output filename expectation mismatch (`ps2_submission_predictions.csv` missing in repo state).
- MINOR:
1. Notebook outputs are mostly empty in committed JSON (acceptable for source control but weaker as pre-rendered evidence).

### 7. VERDICT
- KEEP, REFACTOR guide/output naming consistency.

---

## Folder: submission_best_overall_package

### 1. PURPOSE
- Final curated judge-facing package from latest complete run.
- Intended distribution bundle, not runtime source.

### 2. STRUCTURE QUALITY
- Strong structure: artifacts, logs, docs, configs, pipelines, submission, manifest.
- Coherent and self-contained.

### 3. FILE ANALYSIS
- `README.md`: clear quick path with key metrics.
- `CONTENT_MANIFEST.md`: complete provenance list of copied files.
- `requirements.txt`: dependency mirror.
- `artifacts/evidence_latest_run.json`: high-value evidence summary.
- `artifacts/run_.../**/*`: full run payload with core assets; no Captum file present.
- `logs/*`: highlights + metrics summary + full pipeline log included.
- `configs/*`, `docs/*`, `pipelines/*`, `submission/*`: mirrored curated copies from root state.

### 4. DEPENDENCY CHECK
- Built by `pipelines/build_best_overall_package.py`.
- Not consumed by active training runtime.
- Removing does not break pipeline but removes evaluator-ready bundle.

### 5. README / DOCUMENTATION
- Strong package-level README and logs README.
- Accurate for included content.

### 6. ISSUES
- CRITICAL:
1. None in package integrity.
- IMPORTANT:
1. Contains duplicated source scripts/docs, creating parallel-drift risk against root repository over time.
- MINOR:
1. Captum artifact expectation may be implied by broader docs but missing in included run.

### 7. VERDICT
- KEEP (distribution artifact), REFACTOR update workflow to avoid drift.

---

## Folder: submission_cleaned_package

### 1. PURPOSE
- Older curated package variant for evaluator sharing.
- Secondary bundle.

### 2. STRUCTURE QUALITY
- Coherent top-level structure but less complete than best_overall package.

### 3. FILE ANALYSIS
- `README.md`: concise and usable.
- `CONTENT_MANIFEST.md`: curated list but less granular than best_overall manifest style.
- `artifacts/*`: includes run and evidence snapshot.
- `docs/*`: includes key docs but lacks some newer snapshot/next-step files found in best_overall.
- `configs/*`: core profiles included.
- `pipelines/*`: excludes `build_best_overall_package.py` by design.
- `submission/*`: lacks `Best_Overall_Submission_Walkthrough.ipynb` that exists in current submission folder.

### 4. DEPENDENCY CHECK
- Not active runtime dependency.
- Used only as static packaged output.
- Removing does not affect pipeline execution.

### 5. README / DOCUMENTATION
- Has README and manifest.
- Adequate but not as complete as best_overall package.

### 6. ISSUES
- CRITICAL:
1. None runtime-critical.
- IMPORTANT:
1. Duplicate package track increases evaluator confusion about canonical bundle.
2. Diverges from newest package content (missing newer docs/notebook).
- MINOR:
1. Parallel maintenance overhead.

### 7. VERDICT
- REFACTOR (deprecate or clearly mark non-canonical package).

---

## Folder: tests

### 1. PURPOSE
- Lightweight smoke verification for import wiring and dry-run invocation.
- Basic repository health checks.

### 2. STRUCTURE QUALITY
- Minimal and clear.
- Coverage depth is low.

### 3. FILE ANALYSIS
- `test_imports.py`: import smoke checks only.
- `test_pipeline_dryrun.py`: subprocess quick_test run; useful but expensive and not behavior-granular.
- `README.md`: correctly states smoke scope.
- `__init__.py`: placeholder only.

### 4. DEPENDENCY CHECK
- Depends on pipeline entrypoint and module import paths.
- Not used by runtime; used by CI/manual validation.
- Removing tests does not break runtime but removes verification guardrail.

### 5. README / DOCUMENTATION
- Has README; accurate about limited scope.

### 6. ISSUES
- CRITICAL:
1. No tests for dashboard determinism or artifact-contract consistency.
- IMPORTANT:
1. No unit-level tests for model registry compatibility across model types.
2. No assertions for script parameter contracts in PowerShell automation.
- MINOR:
1. `assert True` pattern in module import test is non-informative.

### 7. VERDICT
- KEEP, REFACTOR test depth.

---

# GLOBAL CHECK

## 1. Architecture Consistency
- Core layering is mostly coherent: core -> models/modules -> pipeline -> artifacts -> dashboard/package.
- Main orchestrator is central and functional.
- Documentation sometimes overstates strict modular guarantees.

## 2. Hidden Coupling Issues
- Output path coupling: runtime hardcoded to artifacts despite config output section.
- Automation coupling: machine-specific absolute paths in PowerShell scripts.
- Dashboard coupling: computes fresh synthetic risks from sampled train data instead of run predictions.

## 3. Pipeline/Data Flow Correctness
- End-to-end run from latest stage2 is complete for major outputs and logs.
- Config-driven claim is partially violated by unconditional stage assumptions and interface mismatch risk (`lstm_attention`).
- Artifact reporting semantics drift (7/7 messaging vs SSL-required variants).

## 4. Redundant Logic Across Folders
- Significant duplication in `submission_best_overall_package` and `submission_cleaned_package` mirrors source scripts/docs.
- Duplicate code copies increase drift and search noise.

## 5. Missing Components
- Missing deterministic dashboard prediction path from saved run predictions.
- Missing stronger contract tests for:
1. config flag honoring,
2. model registry compatibility,
3. artifact contract uniformity,
4. script CLI parameter support,
5. docs-claim synchronization.

---

# What Can Be Done (Dependency-Safe Action Plan)

## Priority A (Immediate, high impact)
1. Unify artifact contract definition in one source and consume it in pipeline, dashboard, docs, and package scripts.
2. Make dashboard risk view deterministic by loading run prediction artifacts instead of random synthetic blending.
3. Fix output path semantics: either honor `output.base_dir` or remove the config key and all outputs-folder claims.

## Priority B (Stability and portability)
1. Align model registry and orchestrator fit interfaces; block unsupported model_type at config validation if necessary.
2. Remove hardcoded absolute paths from PowerShell scripts; derive Python executable and root dynamically.
3. Resolve script parameter contract mismatch (documented accepted parameters only).

## Priority C (Hygiene and maintainability)
1. Remove unused imports and dead import surfaces.
2. Mark one package as canonical (`submission_best_overall_package`) and deprecate or regenerate others from one script.
3. Correct documentation drift (`MODULE_CONNECTIONS.md`, `LIMITATIONS.md`, outputs claims).

## Priority D (Verification depth)
1. Add tests for dashboard determinism and artifact contract checks.
2. Add tests for model_type compatibility and failure modes.
3. Add tests for automation scripts parameter validation and portability.

---

# Final Consolidated Verdict
- System core is runnable and evidence-capable.
- Most severe risks are contract drift and non-deterministic demo behavior, not total pipeline failure.
- Canonical path should be: config-driven run -> artifacts evidence export -> best_overall package build.

# Archive Folder

## Purpose

This folder keeps historical and non-canonical outputs out of active execution paths.

## Subfolders

1. runs_failed
   Incomplete or failed run directories moved out of active `artifacts/`.

2. runtime_logs
   Legacy error dumps and historical supervisor logs.

3. legacy_outputs
   Older experiment outputs retained for forensic reference only.

## Dependency notes

1. Active pipeline execution does not read from `archive/`.
2. Dashboard and packaging flows should use active `artifacts/run_*` outputs.

## Handoff guidance

1. Use archive only for debugging and historical comparison.
2. Do not cite archive files as current submission evidence.
3. Prefer moving uncertain files here before deletion to preserve traceability.

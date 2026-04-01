# Runtime Logs Archive

## Purpose

Holds historical error dumps and older supervisor/pipeline logs that are not part of active execution.

## Dependency notes

1. Active pipeline uses current run logs under `artifacts/run_*/logs`.
2. This folder is supplemental and archival.

## Handoff guidance

1. Use only when tracing historical failures.
2. Prefer active run logs for current reproducibility claims.

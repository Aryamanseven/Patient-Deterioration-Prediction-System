# Outputs Folder

This folder stores lightweight experimental outputs (for example quick smoke-test exports).

## Current usage

1. outputs/quick_test contains small artifacts from dry-run tests.

## Dependency notes

1. Canonical pipeline evidence is stored under `artifacts/run_*`.
2. This folder is supplementary and may be empty in many workflows.

## Cleanup policy

1. Safe to clear before final packaging.
2. Keep only if you need reproducibility of smoke-test behavior.
3. Production-grade artifacts should be taken from artifacts/run_* folders.

## Sharing guidance

1. Do not treat this folder as authoritative evidence for judging.
2. Prefer sharing `artifacts/` plus exported evidence JSON for handoff.

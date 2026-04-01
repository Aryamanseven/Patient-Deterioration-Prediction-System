# Failed Runs Archive

## Purpose

Contains failed or incomplete historical runs moved out of active artifacts to reduce operational noise.

## Dependency notes

1. No active script should depend on this folder.
2. Useful for debugging crashes, interrupted runs, and incomplete artifact patterns.

## Handoff guidance

1. Keep for postmortem analysis only.
2. Do not package this folder for judge-facing submission unless explicitly requested.

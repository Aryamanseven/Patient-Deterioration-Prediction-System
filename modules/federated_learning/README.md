# Federated Learning Module

Simulates federated training over client splits using FedAvg.

## What it does

- Splits training data into client subsets.
- Trains local deep models for local_epochs.
- Aggregates model states with weighted averaging.
- Tracks validation metrics per communication round.
- Restores best-performing global state if later rounds degrade.

## Current scope

- Applies to the deep model path.
- Enabled when modules.federated_learning.enabled=true.
- Enabled in default.yaml by default.

## Primary artifact

- fl_rounds_history.json

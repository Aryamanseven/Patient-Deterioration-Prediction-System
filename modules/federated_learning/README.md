# Federated Learning (FL) Simulation Module

Proves our system's ability to maintain high performance while strictly adhering to data privacy laws (HIPAA/GDPR) through decentralized learning.

## Method
- Simulates multiple isolated hospital "silos" using Dirichlet data partitioning (to enforce real-world non-IID conditions).
- Uses FedAvg to securely aggregate CatBoost tree structures (or deep learning weights) across hospitals without ever sharing raw patient records.

## Artifacts Produced
- `fl_rounds_history.json`: Validation performance progression across communication rounds.

## Config
Controlled via `modules.federated_learning` in `default.yaml`.

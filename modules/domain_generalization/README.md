# Domain Generalization (LODO) Module

Ensures the model doesn't overfit to a specific hospital's recording practices and can generalize to unseen hospitals (Out-Of-Distribution performance).

## Method
- **Leave-One-Domain-Out (LODO) Validation:** If data contains patients from $N$ different hospitals or units, we train the model $N$ times. Each time we train on $N-1$ units and test strictly on the left-out unit.
- This creates a massive penalty for algorithms that memorize silo-specific artifacts instead of true physiological deterioration signatures.

## Artifacts Produced
- `lodo_results.csv`: Table containing the PR-AUC and ROC-AUC for every left-out domain, plus the mean.

## Config
Controlled via `modules.domain_generalization` in `default.yaml`. We need to specify the `domain_column`.

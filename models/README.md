# Models

This folder contains all model implementations and ensemble fusion logic.

## Model components

1. catboost_model.py
	Supervised tabular learner for engineered feature matrix.

2. tcn_transformer.py
	Main deep sequence model used in current competition profiles.

3. lstm_attention.py
	Alternative sequence baseline kept for comparative experiments.

4. ensemble.py
	Learns weighted blend across CatBoost and deep model probabilities.

5. model_registry.py
	Factory for selecting model implementations from config.

## Shared interface contract

1. fit(...)
2. predict_proba(...)
3. save(...)
4. load(...)

## Competition behavior

1. Core training path runs DL + CatBoost and saves hardened artifacts.
2. Federated rounds run when federated_learning.enabled=true in config.
3. Final risk score is the blended ensemble probability.

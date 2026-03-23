# Detailed Model Metrics

Thresholds below are computed from each model's own holdout precision-recall curve.

## catboost
- Family: `baseline`
- ROC-AUC: `0.9649`
- PR-AUC: `0.7115`
- Brier score: `0.0482`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4525` with precision `0.4194` and recall `0.8502`
- Alert threshold: `0.8423` with precision `0.7344`, recall `0.6796`, and F1 `0.7060`
- Config: `{"loss_function": "Logloss", "eval_metric": "AUC", "iterations": 457, "depth": 7, "learning_rate": 0.05, "auto_class_weights": "Balanced", "random_strength": 0.5, "l2_leaf_reg": 5.0, "random_seed": 42}`

## catboost_best_deep_average
- Family: `ensemble`
- ROC-AUC: `0.9627`
- PR-AUC: `0.7083`
- Brier score: `0.0556`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4883` with precision `0.3960` and recall `0.8502`
- Alert threshold: `0.8855` with precision `0.7487`, recall `0.6627`, and F1 `0.7031`

## catboost_top2_deep_average
- Family: `ensemble`
- ROC-AUC: `0.9619`
- PR-AUC: `0.7056`
- Brier score: `0.0539`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4599` with precision `0.3860` and recall `0.8502`
- Alert threshold: `0.8433` with precision `0.7081`, recall `0.6930`, and F1 `0.7005`

## deep_average_top3
- Family: `ensemble`
- ROC-AUC: `0.9563`
- PR-AUC: `0.6700`
- Brier score: `0.0620`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4563` with precision `0.3526` and recall `0.8502`
- Alert threshold: `0.9004` with precision `0.6554`, recall `0.6627`, and F1 `0.6591`

## deep_average_top2
- Family: `ensemble`
- ROC-AUC: `0.9549`
- PR-AUC: `0.6687`
- Brier score: `0.0621`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4493` with precision `0.3440` and recall `0.8502`
- Alert threshold: `0.9109` with precision `0.6672`, recall `0.6544`, and F1 `0.6608`

## transformer_encoder
- Family: `deep`
- ROC-AUC: `0.9513`
- PR-AUC: `0.6547`
- Brier score: `0.0742`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.5078` with precision `0.3274` and recall `0.8502`
- Alert threshold: `0.9390` with precision `0.6695`, recall `0.6340`, and F1 `0.6513`
- Best epoch: `6`
- Validation PR-AUC: `0.6869`
- Config: `{"max_seq_len": 24, "d_model": 96, "num_heads": 4, "num_layers": 2, "dropout": 0.15, "batch_size": 256, "learning_rate": 0.001, "weight_decay": 0.0001, "epochs": 8, "patience": 2, "scheduler_name": null}`

## transformer_encoder_wide
- Family: `deep`
- ROC-AUC: `0.9502`
- PR-AUC: `0.6517`
- Brier score: `0.0594`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.3904` with precision `0.3275` and recall `0.8502`
- Alert threshold: `0.9092` with precision `0.6368`, recall `0.6580`, and F1 `0.6472`
- Best epoch: `7`
- Validation PR-AUC: `0.6832`
- Config: `{"max_seq_len": 36, "d_model": 128, "num_heads": 8, "num_layers": 3, "dropout": 0.1, "batch_size": 192, "learning_rate": 0.0005, "weight_decay": 5e-05, "epochs": 12, "patience": 3, "scheduler_name": "cosine"}`

## transformer_encoder_long
- Family: `deep`
- ROC-AUC: `0.9523`
- PR-AUC: `0.6459`
- Brier score: `0.0690`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4868` with precision `0.3416` and recall `0.8502`
- Alert threshold: `0.9204` with precision `0.6279`, recall `0.6548`, and F1 `0.6411`
- Best epoch: `2`
- Validation PR-AUC: `0.6792`
- Config: `{"max_seq_len": 48, "d_model": 160, "num_heads": 8, "num_layers": 4, "dropout": 0.1, "batch_size": 128, "learning_rate": 0.00035, "weight_decay": 5e-05, "epochs": 14, "patience": 4, "scheduler_name": "cosine"}`

## gru_attention
- Family: `deep`
- ROC-AUC: `0.9540`
- PR-AUC: `0.6148`
- Brier score: `0.0681`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4888` with precision `0.3488` and recall `0.8502`
- Alert threshold: `0.8906` with precision `0.6374`, recall `0.6203`, and F1 `0.6288`
- Best epoch: `1`
- Validation PR-AUC: `0.6681`
- Config: `{"hidden_size": 96, "num_layers": 2, "dropout": 0.2, "max_seq_len": 24, "batch_size": 256, "learning_rate": 0.001, "weight_decay": 0.0001, "epochs": 8, "patience": 2, "scheduler_name": null}`

## deep_average_all
- Family: `ensemble`
- ROC-AUC: `0.9532`
- PR-AUC: `0.6059`
- Brier score: `0.0667`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4805` with precision `0.3572` and recall `0.8502`
- Alert threshold: `0.8109` with precision `0.6121`, recall `0.6799`, and F1 `0.6442`

## tcn
- Family: `deep`
- ROC-AUC: `0.8543`
- PR-AUC: `0.3635`
- Brier score: `0.1473`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4170` with precision `0.1241` and recall `0.8511`
- Alert threshold: `0.8838` with precision `0.5099`, recall `0.3366`, and F1 `0.4055`
- Best epoch: `4`
- Validation PR-AUC: `0.4420`
- Config: `{"channels": [64, 64, 64], "kernel_size": 3, "dropout": 0.15, "max_seq_len": 24, "batch_size": 256, "learning_rate": 0.001, "weight_decay": 0.0001, "epochs": 8, "patience": 2, "scheduler_name": null}`

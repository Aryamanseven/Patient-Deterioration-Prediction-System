# Best Model Search Summary

Thresholds below are computed from each model's own evaluation scores.

## catboost_gpu_subsample_train80
- Family: `catboost_retrain80`
- ROC-AUC: `0.9621`
- PR-AUC: `0.7323`
- Brier score: `0.0279`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.1484` with precision `0.3905` and recall `0.8502`
- Alert threshold: `0.6533` with precision `0.7350`, recall `0.6860`, and F1 `0.7096`
- Validation PR-AUC: `nan`
- Config: `{"iterations": 1300, "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 6.0, "random_strength": 0.7, "bootstrap_type": "Bernoulli", "subsample": 0.85, "border_count": 254}`

## catboost_gpu_subsample_plus_top2_transformers
- Family: `weighted_ensemble`
- ROC-AUC: `0.9640`
- PR-AUC: `0.7071`
- Brier score: `0.0520`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4712` with precision `0.4195` and recall `0.8502`
- Alert threshold: `0.8376` with precision `0.7236`, recall `0.6752`, and F1 `0.6985`
- Validation PR-AUC: `0.7351`
- Config: `{"catboost_gpu_subsample": 0.7, "transformer_encoder": 0.12, "transformer_encoder_wide": 0.18}`

## catboost_gpu_subsample_plus_transformer_encoder_wide
- Family: `weighted_ensemble`
- ROC-AUC: `0.9639`
- PR-AUC: `0.7070`
- Brier score: `0.0509`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4518` with precision `0.4078` and recall `0.8505`
- Alert threshold: `0.8368` with precision `0.7280`, recall `0.6723`, and F1 `0.6990`
- Validation PR-AUC: `0.7343`
- Config: `{"catboost_gpu_subsample": 0.77, "transformer_encoder_wide": 0.23}`

## catboost_gpu_subsample_logistic_stack
- Family: `stacking`
- ROC-AUC: `0.9641`
- PR-AUC: `0.7066`
- Brier score: `0.0615`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.5684` with precision `0.4217` and recall `0.8502`
- Alert threshold: `0.9506` with precision `0.7234`, recall `0.6761`, and F1 `0.6990`
- Validation PR-AUC: `0.7346`
- Config: `{"weights": {"catboost_gpu_subsample": 5.615347, "transformer_encoder": 1.137685, "transformer_encoder_wide": 0.703867}, "intercept": -3.292546, "models": ["catboost_gpu_subsample", "transformer_encoder", "transformer_encoder_wide"]}`

## catboost_gpu_subsample_plus_transformer_encoder
- Family: `weighted_ensemble`
- ROC-AUC: `0.9637`
- PR-AUC: `0.7053`
- Brier score: `0.0541`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4889` with precision `0.4205` and recall `0.8502`
- Alert threshold: `0.8372` with precision `0.7166`, recall `0.6796`, and F1 `0.6976`
- Validation PR-AUC: `0.7334`
- Config: `{"catboost_gpu_subsample": 0.76, "transformer_encoder": 0.24}`

## catboost_gpu_subsample_plus_transformer_encoder_long
- Family: `weighted_ensemble`
- ROC-AUC: `0.9641`
- PR-AUC: `0.7031`
- Brier score: `0.0536`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4745` with precision `0.4107` and recall `0.8508`
- Alert threshold: `0.8619` with precision `0.7390`, recall `0.6599`, and F1 `0.6972`
- Validation PR-AUC: `0.7337`
- Config: `{"catboost_gpu_subsample": 0.75, "transformer_encoder_long": 0.25}`

## catboost_gpu_subsample
- Family: `catboost_search`
- ROC-AUC: `0.9635`
- PR-AUC: `0.7025`
- Brier score: `0.0527`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4656` with precision `0.4079` and recall `0.8502`
- Alert threshold: `0.8451` with precision `0.7345`, recall `0.6659`, and F1 `0.6985`
- Validation PR-AUC: `0.7303`
- Config: `{"iterations": 1300, "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 6.0, "random_strength": 0.7, "bootstrap_type": "Bernoulli", "subsample": 0.85, "border_count": 254}`

## catboost_gpu_low_lr
- Family: `catboost_search`
- ROC-AUC: `0.9638`
- PR-AUC: `0.7008`
- Brier score: `0.0566`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4846` with precision `0.4037` and recall `0.8502`
- Alert threshold: `0.8501` with precision `0.7326`, recall `0.6646`, and F1 `0.6970`
- Validation PR-AUC: `0.7289`
- Config: `{"iterations": 2000, "depth": 8, "learning_rate": 0.025, "l2_leaf_reg": 6.0, "random_strength": 0.4, "bootstrap_type": "Bayesian", "bagging_temperature": 0.8, "border_count": 254}`

## catboost_gpu_deep
- Family: `catboost_search`
- ROC-AUC: `0.9635`
- PR-AUC: `0.7000`
- Brier score: `0.0556`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4861` with precision `0.4071` and recall `0.8505`
- Alert threshold: `0.8557` with precision `0.7395`, recall `0.6595`, and F1 `0.6972`
- Validation PR-AUC: `0.7298`
- Config: `{"iterations": 1500, "depth": 8, "learning_rate": 0.035, "l2_leaf_reg": 7.0, "random_strength": 1.0, "bootstrap_type": "Bayesian", "bagging_temperature": 1.0, "border_count": 254}`

## catboost_gpu_base
- Family: `catboost_search`
- ROC-AUC: `0.9630`
- PR-AUC: `0.6996`
- Brier score: `0.0590`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4990` with precision `0.3987` and recall `0.8508`
- Alert threshold: `0.8696` with precision `0.7374`, recall `0.6580`, and F1 `0.6954`
- Validation PR-AUC: `0.7277`
- Config: `{"iterations": 1200, "depth": 7, "learning_rate": 0.05, "l2_leaf_reg": 5.0, "random_strength": 0.5, "bootstrap_type": "Bayesian", "bagging_temperature": 0.5, "border_count": 254}`

## catboost_gpu_regularized
- Family: `catboost_search`
- ROC-AUC: `0.9632`
- PR-AUC: `0.6900`
- Brier score: `0.0614`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.5141` with precision `0.4029` and recall `0.8502`
- Alert threshold: `0.8689` with precision `0.7255`, recall `0.6548`, and F1 `0.6883`
- Validation PR-AUC: `0.7209`
- Config: `{"iterations": 1600, "depth": 7, "learning_rate": 0.04, "l2_leaf_reg": 9.0, "random_strength": 1.5, "bootstrap_type": "Bayesian", "bagging_temperature": 1.5, "border_count": 254}`

## catboost_gpu_fast
- Family: `catboost_search`
- ROC-AUC: `0.9608`
- PR-AUC: `0.6802`
- Brier score: `0.0692`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.5267` with precision `0.3815` and recall `0.8502`
- Alert threshold: `0.8515` with precision `0.6930`, recall `0.6605`, and F1 `0.6764`
- Validation PR-AUC: `0.7147`
- Config: `{"iterations": 900, "depth": 6, "learning_rate": 0.07, "l2_leaf_reg": 4.0, "random_strength": 0.25, "bootstrap_type": "Bayesian", "bagging_temperature": 0.3, "border_count": 128}`

## transformer_encoder
- Family: `deep_existing`
- ROC-AUC: `0.9513`
- PR-AUC: `0.6547`
- Brier score: `0.0742`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.5078` with precision `0.3274` and recall `0.8502`
- Alert threshold: `0.9390` with precision `0.6695`, recall `0.6340`, and F1 `0.6513`
- Validation PR-AUC: `0.6869`
- Config: `{"max_seq_len": 24, "d_model": 96, "num_heads": 4, "num_layers": 2, "dropout": 0.15, "batch_size": 256, "learning_rate": 0.001, "weight_decay": 0.0001, "epochs": 8, "patience": 2, "scheduler_name": null}`

## transformer_encoder_wide
- Family: `deep_existing`
- ROC-AUC: `0.9502`
- PR-AUC: `0.6517`
- Brier score: `0.0594`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.3904` with precision `0.3275` and recall `0.8502`
- Alert threshold: `0.9092` with precision `0.6368`, recall `0.6580`, and F1 `0.6472`
- Validation PR-AUC: `0.6832`
- Config: `{"max_seq_len": 36, "d_model": 128, "num_heads": 8, "num_layers": 3, "dropout": 0.1, "batch_size": 192, "learning_rate": 0.0005, "weight_decay": 5e-05, "epochs": 12, "patience": 3, "scheduler_name": "cosine"}`

## transformer_encoder_long
- Family: `deep_existing`
- ROC-AUC: `0.9523`
- PR-AUC: `0.6459`
- Brier score: `0.0690`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4868` with precision `0.3416` and recall `0.8502`
- Alert threshold: `0.9204` with precision `0.6279`, recall `0.6548`, and F1 `0.6411`
- Validation PR-AUC: `0.6792`
- Config: `{"max_seq_len": 48, "d_model": 160, "num_heads": 8, "num_layers": 4, "dropout": 0.1, "batch_size": 128, "learning_rate": 0.00035, "weight_decay": 5e-05, "epochs": 14, "patience": 4, "scheduler_name": "cosine"}`

## gru_attention
- Family: `deep_existing`
- ROC-AUC: `0.9540`
- PR-AUC: `0.6148`
- Brier score: `0.0681`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4888` with precision `0.3488` and recall `0.8502`
- Alert threshold: `0.8906` with precision `0.6374`, recall `0.6203`, and F1 `0.6288`
- Validation PR-AUC: `0.6681`
- Config: `{"hidden_size": 96, "num_layers": 2, "dropout": 0.2, "max_seq_len": 24, "batch_size": 256, "learning_rate": 0.001, "weight_decay": 0.0001, "epochs": 8, "patience": 2, "scheduler_name": null}`

## tcn
- Family: `deep_existing`
- ROC-AUC: `0.8543`
- PR-AUC: `0.3635`
- Brier score: `0.1473`
- Holdout positive rate: `0.0525`
- Watch threshold: `0.4170` with precision `0.1241` and recall `0.8511`
- Alert threshold: `0.8838` with precision `0.5099`, recall `0.3366`, and F1 `0.4055`
- Validation PR-AUC: `0.4420`
- Config: `{"channels": [64, 64, 64], "kernel_size": 3, "dropout": 0.15, "max_seq_len": 24, "batch_size": 256, "learning_rate": 0.001, "weight_decay": 0.0001, "epochs": 8, "patience": 2, "scheduler_name": null}`

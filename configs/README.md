# Configuration Files

All pipeline behavior is controlled via YAML configuration files. **No hardcoded parameters exist anywhere in the codebase.**

## Files

| File | Purpose |
|---|---|
| `default.yaml` | Full pipeline configuration — runs everything (SSL → DL → Supervised → Ensemble → XAI → FL → DP → DG → Export) |
| `quick_test.yaml` | Fast dry-run on 5% of data. Used to verify the pipeline works end-to-end before committing to full training |
| `production.yaml` | Optimized for maximum performance — larger models, more epochs, all modules enabled |

## Config Schema

```yaml
general:
  seed: 42                    # Global random seed for reproducibility
  python_version: "3.10"      # Enforced Python version
  device: "auto"              # "cpu", "cuda", or "auto"
  run_name: "experiment_001"  # Unique run identifier

data:
  path: "dataset/train.csv"   # Path to dataset
  test_size: 0.2              # Validation split ratio
  max_rows: null              # null = use all data, integer = subsample

features:
  use_advanced: true          # Enable 257-feature advanced engineering
  use_clinical_scores: true   # Add NEWS/MEWS/qSOFA as features

modules:
  ssl:
    enabled: true             # Enable Self-Supervised Learning pre-training
    pretext_task: "masked_prediction"  # "masked_prediction" or "contrastive"
    mask_ratio: 0.15
    pretrain_epochs: 10
  
  supervised:
    enabled: true
    model_type: "catboost"    # "catboost"
    params: {...}             # CatBoost hyperparameters
  
  deep_learning:
    enabled: true
    model_type: "tcn_transformer"  # "lstm_attention" or "tcn_transformer"
    use_ssl_weights: true     # Initialize from SSL pre-trained encoder
    epochs: 25
    batch_size: 512
    learning_rate: 0.001
  
  ensemble:
    enabled: true
    method: "weighted_blend"  # "weighted_blend" or "stacking"
  
  xai:
    enabled: true
    shap_samples: 1000
    gradcam: true
    attention_maps: true
    clinical_comparison: true
  
  federated_learning:
    enabled: true
    num_hospitals: 4
    fl_rounds: 5
    aggregation: "fedavg"
  
  differential_privacy:
    enabled: true
    target_epsilon: 8.0
    max_grad_norm: 1.0
  
  domain_generalization:
    enabled: true
    method: "lodo"            # Leave-One-Domain-Out
  
  cross_validation:
    enabled: true
    n_splits: 5
  
  fairness:
    enabled: true
    protected_attributes: ["gender", "age_group"]
  
  deployment:
    onnx_export: true
    openvino_convert: true    # Attempted, graceful fallback if unavailable

output:
  base_dir: "outputs"        # All outputs go here
```

## Usage

```bash
# Full pipeline
python pipelines/run_full_pipeline.py --config configs/default.yaml

# Quick test (verify everything works)
python pipelines/run_full_pipeline.py --config configs/quick_test.yaml

# Custom config
python pipelines/run_full_pipeline.py --config configs/my_experiment.yaml
```

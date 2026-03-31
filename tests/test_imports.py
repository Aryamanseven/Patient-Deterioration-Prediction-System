import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_core_imports():
    from core import config, logger, data_loader, features, clinical_scores, metrics, reproducibility
    assert hasattr(config, "load_config")
    assert hasattr(features, "engineer_all_features")

def test_model_imports():
    from models import catboost_model, lstm_attention, model_registry, tcn_transformer, ensemble
    assert hasattr(model_registry, "ModelRegistry")

def test_module_imports():
    from modules.ssl import pretrain
    from modules.xai import explainer
    from modules.federated_learning import simulation
    from modules.differential_privacy import optimizer
    from modules.domain_generalization import lodo
    from modules.deployment import exporter
    
    assert True  # Just verifying they import without syntax/circular deps

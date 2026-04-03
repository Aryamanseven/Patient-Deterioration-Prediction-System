"""
Configuration loader and validator.

Every parameter in the system comes from a YAML config file.
No hardcoded values exist anywhere else in the codebase.
"""
from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Any

import yaml


def get_seed(config: dict[str, Any], default: int = 42) -> int:
    """Resolve seed from either legacy or current config keys."""
    general = config.get("general", {})
    seed_value = general.get("seed", general.get("random_seed", default))
    try:
        return int(seed_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid seed value: {seed_value}") from exc


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate that essential config fields are present and correct."""
    # Check Python version
    required_version = config.get("general", {}).get("python_version", "3.10")
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    if current != required_version:
        print(f"[WARN] Config requires Python {required_version}, running {current}")

    # Required top-level keys
    required_keys = ["general", "data", "features", "modules", "output"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: '{key}'")

    # Data path must exist
    data_path = Path(config["data"]["path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # Validate module configs
    modules = config.get("modules", {})
    valid_modules = {
        "ssl", "supervised", "deep_learning", "ensemble", "xai",
        "calibration", "federated_learning", "differential_privacy",
        "domain_generalization", "cross_validation", "fairness", "deployment",
    }
    for module_name in modules:
        if module_name not in valid_modules:
            raise ValueError(f"Unknown module: '{module_name}'. Valid: {valid_modules}")

    # If ensemble is enabled, at least one model must be enabled
    if modules.get("ensemble", {}).get("enabled", False):
        has_model = (
            modules.get("supervised", {}).get("enabled", False)
            or modules.get("deep_learning", {}).get("enabled", False)
        )
        if not has_model:
            raise ValueError("Ensemble requires at least one model (supervised or deep_learning)")


def get_module_config(config: dict[str, Any], module_name: str) -> dict[str, Any]:
    """Get configuration for a specific module."""
    return config.get("modules", {}).get(module_name, {})


def is_module_enabled(config: dict[str, Any], module_name: str) -> bool:
    """Check if a module is enabled in the config."""
    return get_module_config(config, module_name).get("enabled", False)


def get_device(config: dict[str, Any]) -> str:
    """Resolve the device setting, prioritizing CUDA > DirectML (AMD) > CPU."""
    import torch
    device_setting = config.get("general", {}).get("device", "auto")
    if device_setting == "auto":
        if torch.cuda.is_available():
            return "cuda"
        try:
            import torch_directml
            if torch_directml.is_available():
                return torch_directml.device()
        except ImportError:
            pass
        return "cpu"
    return device_setting


def get_output_dir(config: dict[str, Any]) -> Path:
    """Get the output directory for the current run, create strict structure."""
    import datetime
    
    # Force artifacts/run_<id>/ structure
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    raw_run_name = str(config.get("general", {}).get("run_name", "")).strip()
    safe_run_name = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_run_name).strip("_")
    run_suffix = f"_{safe_run_name}" if safe_run_name else ""

    run_dir = Path("artifacts") / f"run_{timestamp}{run_suffix}"
    
    # Create required subdirectories
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "module_outputs").mkdir(parents=True, exist_ok=True)
    
    # Save a copy of the active config
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)
        
    return run_dir

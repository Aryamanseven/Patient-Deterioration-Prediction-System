"""
Export machine-readable run evidence for evaluator trust.

Generates a JSON report with:
- run identity and timestamps
- config path and key flags
- metrics summary
- SHA256 hashes for critical artifacts
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_latest_complete_run(artifacts_dir: Path) -> Path:
    run_dirs = sorted(
        [d for d in artifacts_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        if (run_dir / "metrics.json").exists() and (run_dir / "predictions.csv").exists():
            return run_dir
    raise FileNotFoundError("No completed run (with metrics.json and predictions.csv) found.")


def parse_metrics(metrics_path: Path) -> dict[str, Any]:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "ensemble" in data:
        return {
            "schema": "nested",
            "deep_learning_pr_auc": data.get("deep_learning", {}).get("pr_auc"),
            "catboost_pr_auc": data.get("catboost", {}).get("pr_auc"),
            "ensemble_pr_auc": data.get("ensemble", {}).get("pr_auc"),
            "deep_learning_roc_auc": data.get("deep_learning", {}).get("roc_auc"),
            "catboost_roc_auc": data.get("catboost", {}).get("roc_auc"),
            "ensemble_roc_auc": data.get("ensemble", {}).get("roc_auc"),
        }
    return {
        "schema": "flat",
        "ensemble_pr_auc": data.get("pr_auc"),
        "ensemble_roc_auc": data.get("roc_auc"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export latest completed run evidence JSON.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    parser.add_argument("--out", default="artifacts/evidence_latest_run.json", help="Output JSON path")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    out_path = Path(args.out)

    run_dir = pick_latest_complete_run(artifacts_dir)
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.csv"
    model_dir = run_dir / "model"

    config_data: dict[str, Any] = {}
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        try:
            import yaml  # type: ignore

            config_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            config_data = {}

    evidence = {
        "run_name": run_dir.name,
        "run_path": str(run_dir),
        "generated_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "timestamps": {
            "run_last_modified": run_dir.stat().st_mtime,
            "metrics_last_modified": metrics_path.stat().st_mtime,
            "predictions_last_modified": predictions_path.stat().st_mtime,
        },
        "metrics": parse_metrics(metrics_path),
        "config_highlights": {
            "python_version": config_data.get("general", {}).get("python_version"),
            "run_name": config_data.get("general", {}).get("run_name"),
            "federated_learning_enabled": config_data.get("modules", {}).get("federated_learning", {}).get("enabled"),
            "domain_generalization_enabled": config_data.get("modules", {}).get("domain_generalization", {}).get("enabled"),
            "xai_enabled": config_data.get("modules", {}).get("xai", {}).get("enabled"),
            "ssl_reuse_existing": config_data.get("modules", {}).get("ssl", {}).get("reuse_existing"),
            "catboost_iterations": config_data.get("modules", {}).get("supervised", {}).get("params", {}).get("iterations"),
            "catboost_learning_rate": config_data.get("modules", {}).get("supervised", {}).get("params", {}).get("learning_rate"),
        },
        "artifact_hashes_sha256": {
            "metrics.json": sha256_file(metrics_path),
            "predictions.csv": sha256_file(predictions_path),
            "model/model.cbm": sha256_file(model_dir / "model.cbm"),
            "model/dl_model_final.pt": sha256_file(model_dir / "dl_model_final.pt"),
            "model/ensemble.pkl": sha256_file(model_dir / "ensemble.pkl"),
            "model/feature_columns.json": sha256_file(model_dir / "feature_columns.json"),
            "ssl_pretrained_tcntransformer.pt": sha256_file(run_dir / "ssl_pretrained_tcntransformer.pt"),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()

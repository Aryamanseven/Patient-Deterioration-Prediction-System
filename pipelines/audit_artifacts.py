"""
Artifact audit utility (non-destructive).

Scans artifacts/run_* directories and reports whether each run has
all required output files for reloadable inference/training handoff.

This script NEVER deletes files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


BASE_REQUIRED = [
    "model/dl_model_final.pt",
    "model/model.cbm",
    "model/scaler.pkl",
    "model/ensemble.pkl",
    "model/feature_columns.json",
    "metrics.json",
    "predictions.csv",
]
SSL_REQUIRED = "ssl_pretrained_tcntransformer.pt"


def _ssl_expected(config_path: Path) -> bool:
    if not config_path.exists() or yaml is None:
        return False

    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        ssl_cfg = cfg.get("modules", {}).get("ssl", {})
        return bool(
            ssl_cfg.get("enabled", False)
            or ssl_cfg.get("reuse_existing", False)
            or ssl_cfg.get("pretrained_weights_path")
        )
    except Exception:
        return False


def inspect_run(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.yaml"
    required = list(BASE_REQUIRED)
    if _ssl_expected(config_path):
        required.append(SSL_REQUIRED)

    missing = []
    present = []
    for rel in required:
        target = run_dir / rel
        if target.exists() and target.is_file() and target.stat().st_size > 0:
            present.append(rel)
        else:
            missing.append(rel)

    model_dir = run_dir / "model"
    model_files = []
    if model_dir.exists():
        model_files = sorted([p.name for p in model_dir.glob("*") if p.is_file()])

    status = "complete" if not missing else "incomplete"

    # Conservative cleanup hint only: no model files means run is usually unusable.
    cleanup_candidate = status == "incomplete" and len(model_files) == 0

    return {
        "run": run_dir.name,
        "status": status,
        "required_count": len(required),
        "present_count": len(present),
        "missing_count": len(missing),
        "missing": missing,
        "model_files": model_files,
        "cleanup_candidate": cleanup_candidate,
    }


def print_table(rows: list[dict[str, Any]], include_complete: bool) -> None:
    filtered = rows if include_complete else [r for r in rows if r["status"] != "complete"]
    if not filtered:
        print("No runs to show with current filter.")
        return

    header = f"{'run':<56} {'status':<11} {'present/required':<17} {'model_files':<11} {'cleanup?':<9}"
    print(header)
    print("-" * len(header))

    for row in filtered:
        ratio = f"{row['present_count']}/{row['required_count']}"
        model_count = str(len(row["model_files"]))
        cleanup = "yes" if row["cleanup_candidate"] else "no"
        print(f"{row['run']:<56} {row['status']:<11} {ratio:<17} {model_count:<11} {cleanup:<9}")

    print("\nMissing files by run:")
    for row in filtered:
        if row["missing"]:
            print(f"- {row['run']}: {', '.join(row['missing'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit run artifacts without deleting anything")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Path to artifacts directory")
    parser.add_argument(
        "--include-complete",
        action="store_true",
        help="Include complete runs in table output",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional output path to save full JSON report",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    run_dirs = sorted(
        [p for p in artifacts_dir.iterdir() if p.is_dir() and p.name.startswith("run_")],
        key=lambda p: p.name,
    )

    rows = [inspect_run(run_dir) for run_dir in run_dirs]
    print_table(rows, include_complete=args.include_complete)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"\nJSON report written to: {out_path}")


if __name__ == "__main__":
    main()

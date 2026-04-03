from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def _run_final_benchmark(*, max_eval_rows: int, fractions: str, repeats: int, seed: int) -> Path:
    script_path = PROJECT_ROOT / "pipelines" / "run_timesfm_vs_latest_ensemble_subsample.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--fractions",
        str(fractions),
        "--repeats",
        str(repeats),
        "--seed",
        str(seed),
    ]

    if max_eval_rows <= 0:
        cmd.extend(["--max-eval-rows", "0"])
    else:
        cmd.extend(["--max-eval-rows", str(max_eval_rows)])

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "Final benchmark script failed.\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

    pointer_path = ARTIFACTS_DIR / "timesfm_vs_latest_ensemble_latest.json"
    if not pointer_path.exists():
        raise FileNotFoundError(f"Missing benchmark pointer after run: {pointer_path}")

    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    rel = payload.get("path")
    if not rel:
        raise ValueError("timesfm_vs_latest_ensemble_latest.json missing 'path'")

    source_dir = PROJECT_ROOT / str(rel)
    if not source_dir.exists():
        raise FileNotFoundError(f"Generated benchmark folder not found: {source_dir}")

    return source_dir


def _build_final_package(*, source_dir: Path, final_dir: Path) -> dict[str, Any]:
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    keep_files = [
        "summary.json",
        "full_sample_metrics.csv",
        "head_to_head_by_fraction.csv",
        "winner_by_fraction.csv",
        "subsample_metrics_summary.csv",
        "subsample_metrics_detailed.csv",
    ]

    copied: list[str] = []
    for rel in keep_files:
        src = source_dir / rel
        if src.exists():
            shutil.copy2(src, final_dir / rel)
            copied.append(rel)

    summary_path = final_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected summary not found in final package: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    full = summary.get("full_sample_metrics", {})
    ens = full.get("latest_ensemble", {})
    tsf = full.get("timesfm_proxy", {})
    delta = full.get("delta", {})

    facts_rows = [
        {
            "model": "latest_ensemble",
            "pr_auc": ens.get("pr_auc"),
            "roc_auc": ens.get("roc_auc"),
            "brier_score": ens.get("brier_score"),
            "n": ens.get("n"),
        },
        {
            "model": "timesfm_proxy",
            "pr_auc": tsf.get("pr_auc"),
            "roc_auc": tsf.get("roc_auc"),
            "brier_score": tsf.get("brier_score"),
            "n": tsf.get("n"),
        },
        {
            "model": "delta",
            "pr_auc": delta.get("pr_auc_ensemble_minus_timesfm"),
            "roc_auc": delta.get("roc_auc_ensemble_minus_timesfm"),
            "brier_score": delta.get("brier_timesfm_minus_ensemble"),
            "n": ens.get("n"),
        },
    ]
    pd.DataFrame(facts_rows).to_csv(final_dir / "benchmark_facts.csv", index=False)

    facts_md = [
        "# Final Benchmark Facts",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "Strict two-model benchmark:",
        "1. latest_ensemble",
        "2. timesfm_proxy",
        "",
        "## Full-Sample Metrics",
        "",
        f"1. Latest Ensemble PR-AUC: {float(ens.get('pr_auc', 0.0)):.6f}",
        f"2. TimeSFM Proxy PR-AUC: {float(tsf.get('pr_auc', 0.0)):.6f}",
        f"3. PR-AUC delta (Ensemble - TimeSFM): {float(delta.get('pr_auc_ensemble_minus_timesfm', 0.0)):.6f}",
        f"4. Latest Ensemble ROC-AUC: {float(ens.get('roc_auc', 0.0)):.6f}",
        f"5. TimeSFM Proxy ROC-AUC: {float(tsf.get('roc_auc', 0.0)):.6f}",
        f"6. ROC-AUC delta (Ensemble - TimeSFM): {float(delta.get('roc_auc_ensemble_minus_timesfm', 0.0)):.6f}",
        f"7. Latest Ensemble Brier: {float(ens.get('brier_score', 0.0)):.6f}",
        f"8. TimeSFM Proxy Brier: {float(tsf.get('brier_score', 0.0)):.6f}",
        f"9. Brier delta (TimeSFM - Ensemble): {float(delta.get('brier_timesfm_minus_ensemble', 0.0)):.6f}",
        "",
        "## Included Files",
        "",
        "1. summary.json",
        "2. benchmark_facts.csv",
        "3. full_sample_metrics.csv",
        "4. head_to_head_by_fraction.csv",
        "5. winner_by_fraction.csv",
        "6. subsample_metrics_summary.csv",
        "7. subsample_metrics_detailed.csv",
    ]
    (final_dir / "README.md").write_text("\n".join(facts_md) + "\n", encoding="utf-8")

    return {
        "copied_files": copied,
        "source_dir": str(source_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "final_dir": str(final_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }


def _cleanup_old_benchmark_noise(*, protected_final_dir: Path) -> dict[str, list[str]]:
    removed_dirs: list[str] = []
    removed_files: list[str] = []

    # Remove noisy benchmark directories from previous iterations.
    dir_patterns = [
        "timesfm_vs_latest_ensemble_subsample_*",
        "benchmark_suite_*",
    ]

    for pattern in dir_patterns:
        for path in ARTIFACTS_DIR.glob(pattern):
            if not path.is_dir():
                continue
            if path.resolve() == protected_final_dir.resolve():
                continue
            shutil.rmtree(path)
            removed_dirs.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    optional_dirs = [
        ARTIFACTS_DIR / "subsample_benchmark_latest",
    ]
    for path in optional_dirs:
        if path.exists() and path.is_dir() and path.resolve() != protected_final_dir.resolve():
            shutil.rmtree(path)
            removed_dirs.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    removable_files = [
        ARTIFACTS_DIR / "timesfm_vs_latest_ensemble_latest.json",
        ARTIFACTS_DIR / "benchmark_suite_latest.json",
    ]
    for path in removable_files:
        if path.exists() and path.is_file():
            path.unlink()
            removed_files.append(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    return {
        "removed_dirs": sorted(removed_dirs),
        "removed_files": sorted(removed_files),
    }


def finalize_artifacts(
    *,
    max_eval_rows: int,
    fractions: str,
    repeats: int,
    seed: int,
    final_dir_name: str,
) -> Path:
    source_dir = _run_final_benchmark(
        max_eval_rows=max_eval_rows,
        fractions=fractions,
        repeats=repeats,
        seed=seed,
    )

    final_dir = ARTIFACTS_DIR / final_dir_name
    package_info = _build_final_package(source_dir=source_dir, final_dir=final_dir)
    cleanup_info = _cleanup_old_benchmark_noise(protected_final_dir=final_dir)

    pointer_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(final_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_benchmark_dir": package_info["source_dir"],
    }
    (ARTIFACTS_DIR / "final_benchmark_latest.json").write_text(
        json.dumps(pointer_payload, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "final_package": package_info,
        "cleanup": cleanup_info,
        "run_config": {
            "max_eval_rows": max_eval_rows,
            "fractions": fractions,
            "repeats": repeats,
            "seed": seed,
        },
    }
    (final_dir / "cleanup_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return final_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one final benchmark pass, build one final package, and remove old benchmark clutter."
    )
    parser.add_argument("--max-eval-rows", type=int, default=10000)
    parser.add_argument("--fractions", default="0.05,0.1,0.2,0.3,0.5,0.8")
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final-dir-name", default="benchmark_final")
    args = parser.parse_args()

    final_dir = finalize_artifacts(
        max_eval_rows=int(args.max_eval_rows),
        fractions=str(args.fractions),
        repeats=int(args.repeats),
        seed=int(args.seed),
        final_dir_name=str(args.final_dir_name),
    )

    print(str(final_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

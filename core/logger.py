"""
Structured logging — consistent output across all modules.

Writes to both console and a log file in the run output directory.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_configured = False


def get_logger(
    name: str = "physioguard",
    log_dir: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Get or create a logger with console + file output."""
    logger = logging.getLogger(name)

    if getattr(logger, "is_configured", False):
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (if log_dir given)
    if log_dir is not None:
        log_dir = Path(log_dir)
        logs_subdir = log_dir / "logs"
        logs_subdir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logs_subdir / "pipeline.log", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.is_configured = True
    return logger

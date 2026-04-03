"""
Structured logging — consistent output across all modules.

Writes to both console and a log file in the run output directory.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_configured = False
_global_log_dir: Path | None = None


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _has_console_handler(logger: logging.Logger) -> bool:
    return any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def _has_file_handler_for(logger: logging.Logger, log_file: Path) -> bool:
    target = str(log_file.resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if str(Path(handler.baseFilename).resolve()) == target:
                    return True
            except Exception:
                continue
    return False


def _ensure_console_handler(logger: logging.Logger, level: int) -> None:
    if _has_console_handler(logger):
        return
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(_build_formatter())
    logger.addHandler(console)


def _ensure_file_handler(logger: logging.Logger, log_dir: Path, level: int) -> None:
    logs_subdir = Path(log_dir) / "logs"
    logs_subdir.mkdir(parents=True, exist_ok=True)
    log_file = logs_subdir / "pipeline.log"
    if _has_file_handler_for(logger, log_file):
        return
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(_build_formatter())
    logger.addHandler(file_handler)


def set_global_log_dir(log_dir: Path | str | None, level: int = logging.INFO) -> None:
    """Attach file logging to all configured loggers for a run-wide evidence trail."""
    global _global_log_dir
    _global_log_dir = None if log_dir is None else Path(log_dir)
    if _global_log_dir is None:
        return

    manager = logging.Logger.manager.loggerDict
    for name in list(manager.keys()):
        logger_obj = logging.getLogger(name)
        if not isinstance(logger_obj, logging.Logger):
            continue
        logger_obj.setLevel(level)
        logger_obj.propagate = False
        _ensure_console_handler(logger_obj, level)
        _ensure_file_handler(logger_obj, _global_log_dir, level)


def get_logger(
    name: str = "physioguard",
    log_dir: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Get or create a logger with console + file output."""
    logger = logging.getLogger(name)

    if getattr(logger, "is_configured", False):
        target_log_dir = log_dir if log_dir is not None else _global_log_dir
        if target_log_dir is not None:
            _ensure_file_handler(logger, Path(target_log_dir), level)
        return logger

    logger.setLevel(level)
    logger.propagate = False

    _ensure_console_handler(logger, level)

    target_log_dir = log_dir if log_dir is not None else _global_log_dir
    if target_log_dir is not None:
        _ensure_file_handler(logger, Path(target_log_dir), level)

    logger.is_configured = True
    return logger

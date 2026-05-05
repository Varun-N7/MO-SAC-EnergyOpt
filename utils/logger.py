"""
Shared logging utilities for training and evaluation scripts.
"""

import logging
from pathlib import Path
from typing import Dict, Any

import config


def get_logger(name: str) -> logging.Logger:
    """Create or reuse a standardized console logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def ensure_project_dirs() -> None:
    """Create all required runtime directories if they do not exist."""
    for path in (
        config.MODELS_DIR,
        config.RESULTS_DIR,
        config.RESULTS_LOGS_DIR,
        config.RESULTS_PLOTS_DIR,
        config.TENSORBOARD_DIR,
        config.REAL_DATA_DIR,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)


def log_metrics(logger: logging.Logger, title: str, metrics: Dict[str, Any]) -> None:
    """Log a compact key-value metrics block."""
    logger.info("%s", title)
    for key, value in metrics.items():
        logger.info("  %s: %s", key, value)

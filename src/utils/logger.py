"""Logging utilities for fraud detection pipeline."""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "fraud_detection",
    log_dir: Optional[Path] = None,
    log_file: str = "pipeline.log",
    level: int = logging.INFO,
    additional_log_dirs: Optional[list] = None,
) -> logging.Logger:
    """Configure and return a dual file-and-console logger.

    Args:
        name: Logger name.
        log_dir: Directory where primary log file is saved.
        log_file: Name of the log file.
        level: Logging level.
        additional_log_dirs: Optional list of additional directories to save the log file to.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to allow clean re-initialization
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Primary File Handler
    target_dirs = []
    if log_dir is not None:
        target_dirs.append(Path(log_dir))
    if additional_log_dirs:
        for d in additional_log_dirs:
            target_dirs.append(Path(d))

    for directory in target_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / log_file
        file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "fraud_detection") -> logging.Logger:
    """Retrieve existing logger or create default.

    Args:
        name: Logger name.

    Returns:
        logging.Logger: Logger instance.
    """
    return logging.getLogger(name)

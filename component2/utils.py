"""
utils.py — Logging, plotting, and results utilities for MATH562 Experiment 1.
"""

import os
import logging
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────

def setup_logger(log_dir: str, name: str = "experiment", log_stem: str = None) -> logging.Logger:
    """
    Create a logger that writes to both a timestamped file (DEBUG+)
    and stdout (INFO+).

    Parameters
    ----------
    log_dir  : directory where the log file is created
    name     : logger name (also used as file prefix)
    log_stem : if provided, reuse this stem and open the log in append mode
               (used when resuming a run from a checkpoint)

    Returns
    -------
    (logging.Logger, log_stem)
    """
    os.makedirs(log_dir, exist_ok=True)
    if log_stem is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_stem = f"{name}_{timestamp}"
        file_mode = "w"
    else:
        file_mode = "a"  # append to existing log on resume
    log_file = os.path.join(log_dir, f"{log_stem}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # avoid duplicate handlers on re-runs in notebooks

    # File handler — everything
    fh = logging.FileHandler(log_file, mode=file_mode, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  [%(levelname)-7s]  %(message)s", datefmt="%H:%M:%S"
    ))

    # Console handler — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Logging to {log_file}")
    log_stem = f"{name}_{timestamp}"
    return logger, log_stem

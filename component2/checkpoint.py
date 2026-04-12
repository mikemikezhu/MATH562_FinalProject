"""
checkpoint.py — Save/load experiment checkpoints for MATH562 Component 2.

Usage
-----
After each completed configuration in a main.py loop:

    from component2.checkpoint import save_checkpoint, load_checkpoint

    save_checkpoint(checkpoint_path, log_stem, results_serializable,
                    completed_keys_set, vars(args), "experiment1")

To resume:

    python component2/exp_1/main.py --resume results/checkpoints/experiment1_<stem>_checkpoint.json
"""

import os
import json


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

def _to_serialisable(obj):
    """Recursively convert numpy types / arrays into JSON-serialisable Python types."""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    log_stem: str,
    results_serializable: dict,
    completed_keys,
    args_dict: dict,
    experiment_name: str,
) -> None:
    """
    Atomically write a checkpoint JSON file.

    Uses a .tmp intermediary + os.replace() so a crash mid-write never
    leaves a corrupted checkpoint on disk.

    Parameters
    ----------
    path               : full path for the checkpoint file
    log_stem           : the run's log stem (e.g. "experiment1_20240402_120000")
    results_serializable : dict[str_key → dict] with already-serialisable values
    completed_keys     : iterable of string keys that have finished
    args_dict          : vars(args) from argparse
    experiment_name    : "experiment1", "experiment2", or "experiment3"
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "experiment": experiment_name,
        "log_stem": log_stem,
        "args": _to_serialisable(args_dict),
        "completed_keys": list(completed_keys),
        "results": results_serializable,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_checkpoint(path: str):
    """
    Load a checkpoint file.

    Returns
    -------
    (log_stem, results_dict, completed_keys_set)
        log_stem          : str — the original run's log stem
        results_dict      : dict[str_key → dict] — serialised results so far
        completed_keys_set: set[str] — keys of completed configurations
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for key in ("log_stem", "completed_keys", "results"):
        if key not in data:
            raise ValueError(f"Checkpoint missing required key '{key}': {path}")
    return data["log_stem"], data["results"], set(data["completed_keys"])

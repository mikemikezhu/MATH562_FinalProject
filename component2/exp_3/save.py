import os
import json
import logging
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_serialisable(obj):
    """
    Recursively convert numpy types / arrays into JSON-serialisable Python types.
    """
    if isinstance(obj, dict):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# Results I/O
# ──────────────────────────────────────────────────────────────────────────────

def save_results(results: dict, save_dir: str, log_stem: str = "results") -> str:
    """
    Save metadata + kernel metrics to JSON.

    Expected structure of results:
        {
            (regime, activation, m): {
                "metadata": {...},
                "metrics": {
                    checkpoint: {
                        "fro_norm": ...,
                        "abs_change": ...,
                        "rel_change": ...,
                        "mean_value": ...
                    },
                    ...
                }
            },
            ...
        }

    Returns path to the JSON file.
    """
    os.makedirs(save_dir, exist_ok=True)

    serialisable = {}
    for (regime, activation, m, beta_scaling), val in results.items():
        scale_str = f"{beta_scaling:g}"
        str_key = f"{regime}__{activation}__{m}__scale_{scale_str}"

        metadata = val.get("metadata", {})
        metrics = val.get("metrics", {})

        serialisable[str_key] = {
            "regime": regime,
            "activation": activation,
            "m": m,
            "beta_scaling": beta_scaling,
            "metadata": _to_serialisable(metadata),
            "metrics": _to_serialisable(metrics),
        }

    path = os.path.join(save_dir, f"{log_stem}.json")
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)

    return path


def save_kernel_matrices(results: dict, save_dir: str, log_stem: str = "results") -> str:
    """
    Save kernel matrices at each checkpoint to a .npz file.

    Expected structure of results:
        {
            (regime, activation, m): {
                "kernels": {
                    checkpoint: K_checkpoint,   # numpy arrays
                    ...
                },
                ...
            },
            ...
        }

    In the .npz file, each array is stored under a key like:
        "NTK__tanh__400__ckpt_0"
        "NTK__tanh__400__ckpt_25"
        ...
    """
    os.makedirs(save_dir, exist_ok=True)

    arrays_to_save = {}

    for (regime, activation, m, beta_scaling), val in results.items():
        scale_str = f"{beta_scaling:g}"
        kernels = val.get("kernels", {})
        for checkpoint, K in kernels.items():
            array_key = f"{regime}__{activation}__{m}__scale_{scale_str}__ckpt_{checkpoint}"
            arrays_to_save[array_key] = np.asarray(K)

    path = os.path.join(save_dir, f"{log_stem}_kernels.npz")
    np.savez(path, **arrays_to_save)

    return path


def print_summary_table(results: dict, logger: logging.Logger):
    """
    Print a formatted summary table using the final checkpoint metrics.

    Columns:
        Regime | Activation | m | Final Checkpoint | Fro Norm | Abs Change | Rel Change | Mean Value
    """
    col_w = 14
    sep = "─" * (12 + 12 + 8 + 14 + col_w * 4 + 12)

    header = (
        f"{'Regime':<12} "
        f"{'Activation':<12} "
        f"{'m':<8} "
        f"{'Checkpoint':<14} "
        f"{'Fro Norm':>{col_w}} "
        f"{'Abs Change':>{col_w}} "
        f"{'Rel Change':>{col_w}} "
        f"{'Mean Value':>{col_w}}"
    )

    logger.info(sep)
    logger.info("  FINAL KERNEL METRICS SUMMARY")
    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    for key in sorted(results.keys()):
        regime, activation, m, beta_scaling = key
        metrics = results[key].get("metrics", {})

        if not metrics:
            continue

        checkpoints = sorted(metrics.keys())
        final_ckpt = checkpoints[-1]
        final_metrics = metrics[final_ckpt]

        logger.info(
            f"{regime:<12} "
            f"{activation:<12} "
            f"{m:<8} "
            f"{final_ckpt:<14} "
            f"{final_metrics['fro_norm']:>{col_w}.6f} "
            f"{final_metrics['abs_change']:>{col_w}.6f} "
            f"{final_metrics['rel_change']:>{col_w}.6f} "
            f"{final_metrics['mean_value']:>{col_w}.6f}"
        )

    logger.info(sep)
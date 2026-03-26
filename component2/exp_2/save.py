import os
import json
import logging


# ──────────────────────────────────────────────────────────────────────────────
# Results I/O
# ──────────────────────────────────────────────────────────────────────────────

def save_results(results: dict, save_dir: str, log_stem: str = "results") -> str:
    """
    Serialise the results dict to JSON.

    Keys are (regime, alpha, prefactor,, m) tuples → converted to "regime__act__m" strings.
    The file is named <log_stem>.json so it matches the log and plots for that run.
    Returns the path of the written file.
    """
    os.makedirs(save_dir, exist_ok=True)
    serialisable = {}
    for (regime, alpha, prefactor, m), val in results.items():
        str_key = f"{regime}__{alpha}__{prefactor}__{m}"
        serialisable[str_key] = {
            "regime": regime,
            "alpha": alpha,
            "prefactor": prefactor,
            "m": m,
            "train_losses": val["train_losses"],
            "test_losses": val["test_losses"],
            "final_train_loss": val["train_losses"][-1],
            "final_test_loss": val["test_losses"][-1],
        }
    path = os.path.join(save_dir, f"{log_stem}.json")
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    return path


def print_summary_table(results: dict, logger: logging.Logger):
    """Print a formatted table of final train/test losses for every run."""
    col_w = 14
    sep = "─" * (12 + 10 + 12 + 8 + col_w * 2 + 8)
    header = (
        f"{'Regime':<12} "
        f"{'alpha':<10} "
        f"{'prefactor':<12} "
        f"{'m':<8} "
        f"{'Train Loss':>{col_w}} "
        f"{'Test Loss':>{col_w}}"
    )
    logger.info(sep)
    logger.info("  FINAL LOSSES SUMMARY")
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for key in sorted(results.keys()):
        regime, alpha, prefactor, m = key
        trl = results[key]["train_losses"][-1]
        tel = results[key]["test_losses"][-1]
        logger.info(
            f"{regime:<12} "
            f"{alpha:<10.2f} "
            f"{prefactor:<12.3g} "
            f"{m:<8} "
            f"{trl:>{col_w}.6f} "
            f"{tel:>{col_w}.6f}"
        )
    logger.info(sep)

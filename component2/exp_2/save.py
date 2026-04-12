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


def save_variance_results(variance_results: dict, save_dir: str, log_stem: str = "results") -> str:
    """
    Save mean-variance results to JSON.
    """
    os.makedirs(save_dir, exist_ok=True)

    serialisable = {}
    for (regime, alpha, prefactor), val in variance_results.items():
        str_key = f"{regime}__{alpha}__{prefactor}"
        serialisable[str_key] = {
            "regime": regime,
            "alpha": alpha,
            "prefactor": prefactor,
            "mean_train_variance": val["mean_train_variance"],
            "mean_test_variance": val["mean_test_variance"],
            "mean_log_train_variance": val["mean_log_train_variance"],
            "mean_log_test_variance": val["mean_log_test_variance"],
        }

    path = os.path.join(save_dir, f"{log_stem}_variance.json")
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)

    return path


def save_learn_score_results(learn_score_results: dict, save_dir: str, log_stem: str = "results") -> str:
    """
    Save learn score results to JSON.
    """
    os.makedirs(save_dir, exist_ok=True)

    serialisable = {}
    for (regime, alpha, prefactor), val in learn_score_results.items():
        str_key = f"{regime}__{alpha}__{prefactor}"
        serialisable[str_key] = {
            "regime": regime,
            "alpha": alpha,
            "prefactor": prefactor,
            "mean_train_learn_score": val["mean_train_learn_score"],
            "mean_test_learn_score": val["mean_test_learn_score"],
            "mean_log_train_learn_score": val["mean_log_train_learn_score"],
            "mean_log_test_learn_score": val["mean_log_test_learn_score"],
            "worst_train_learn_score": val["worst_train_learn_score"],
            "worst_test_learn_score": val["worst_test_learn_score"],
            "worst_log_train_learn_score": val["worst_log_train_learn_score"],
            "worst_log_test_learn_score": val["worst_log_test_learn_score"]
        }

    path = os.path.join(save_dir, f"{log_stem}_learn_score.json")
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)

    return path


def print_variance_table(variance_results: dict, logger: logging.Logger):
    """Print a formatted table of mean variance for each scaling."""
    col_w = 18
    header = (
        f"{'Regime':<12} "
        f"{'alpha':<10} "
        f"{'prefactor':<12} "
        f"{'Mean Train Var':>{col_w}} "
        f"{'Mean Test Var':>{col_w}} "
        f"{'Mean Log Train Var':>{col_w}} "
        f"{'Mean Log Test Var':>{col_w}}"
    )
    sep = "─" * len(header)

    logger.info(sep)
    logger.info("  MEAN VARIANCE SUMMARY")
    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    for key in sorted(variance_results.keys()):
        regime, alpha, prefactor = key
        mean_train_var = variance_results[key]["mean_train_variance"]
        mean_test_var = variance_results[key]["mean_test_variance"]
        mean_log_train_var = variance_results[key]["mean_log_train_variance"]
        mean_log_test_var = variance_results[key]["mean_log_test_variance"]

        logger.info(
            f"{regime:<12} "
            f"{alpha:<10.2f} "
            f"{prefactor:<12.3g} "
            f"{mean_train_var:>{col_w}.6e} "
            f"{mean_test_var:>{col_w}.6e} "
            f"{mean_log_train_var:>{col_w}.6e} "
            f"{mean_log_test_var:>{col_w}.6e}"
        )

    logger.info(sep)


def print_learn_score_table(learning_score_results: dict, logger: logging.Logger):
    """Print a formatted table of learn score for each scaling."""
    col_w = 18
    header = (
        f"{'Regime':<12} "
        f"{'alpha':<10} "
        f"{'prefactor':<12} "
        f"{'Mean Train Learn Score':>{col_w}} "
        f"{'Mean Test Learn Score':>{col_w}} "
        f"{'Mean Log Train Learn Score':>{col_w}} "
        f"{'Mean Log Test Learn Score':>{col_w}} "
        f"{'Worst Train Learn Score':>{col_w}} "
        f"{'Worst Test Learn Score':>{col_w}} "
        f"{'Worst Log Train Learn Score':>{col_w}} "
        f"{'Worst Log Test Learn Score':>{col_w}}"
    )
    sep = "─" * len(header)

    logger.info(sep)
    logger.info("  Learn Score SUMMARY")
    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    for key in sorted(learning_score_results.keys()):
        regime, alpha, prefactor = key
        mean_train_learn_score = learning_score_results[key]["mean_train_learn_score"]
        mean_test_learn_score = learning_score_results[key]["mean_test_learn_score"]
        mean_log_train_learn_score = learning_score_results[key]["mean_log_train_learn_score"]
        mean_log_test_learn_score = learning_score_results[key]["mean_log_test_learn_score"]
        worst_train_learn_score = learning_score_results[key]["worst_train_learn_score"]
        worst_test_learn_score = learning_score_results[key]["worst_test_learn_score"]
        worst_log_train_learn_score = learning_score_results[key]["worst_log_train_learn_score"]
        worst_log_test_learn_score = learning_score_results[key]["worst_log_test_learn_score"]

        logger.info(
            f"{regime:<12} "
            f"{alpha:<10.2f} "
            f"{prefactor:<12.3g} "
            f"{mean_train_learn_score:>{col_w}.6e} "
            f"{mean_test_learn_score:>{col_w}.6e} "
            f"{mean_log_train_learn_score:>{col_w}.6e} "
            f"{mean_log_test_learn_score:>{col_w}.6e} "
            f"{worst_train_learn_score:>{col_w}.6e} "
            f"{worst_test_learn_score:>{col_w}.6e} "
            f"{worst_log_train_learn_score:>{col_w}.6e} "
            f"{worst_log_test_learn_score:>{col_w}.6e}"
        )

    logger.info(sep)


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

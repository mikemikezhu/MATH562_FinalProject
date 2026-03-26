"""
utils.py — Logging, plotting, and results utilities for MATH562 Experiment 1.
"""

import os
import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Aesthetics
# ──────────────────────────────────────────────────────────────────────────────

REGIME_COLORS  = {"NTK": "#1f77b4", "MF": "#ff7f0e", "RF": "#2ca02c"}
REGIME_LABELS  = {"NTK": "NTK",     "MF": "Mean Field", "RF": "Random Features"}
ACT_MARKERS    = {"relu": "o",       "erf": "s",          "tanh": "^"}
ACT_LINESTYLES = {"relu": "-",       "erf": "--",         "tanh": "-."}

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────

def setup_logger(log_dir: str, name: str = "experiment") -> logging.Logger:
    """
    Create a logger that writes to both a timestamped file (DEBUG+)
    and stdout (INFO+).

    Parameters
    ----------
    log_dir : directory where the log file is created
    name    : logger name (also used as file prefix)

    Returns
    -------
    logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()   # avoid duplicate handlers on re-runs in notebooks

    # File handler — everything
    fh = logging.FileHandler(log_file, encoding="utf-8")
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


# ──────────────────────────────────────────────────────────────────────────────
# Results I/O
# ──────────────────────────────────────────────────────────────────────────────

def save_results(results: dict, save_dir: str, log_stem: str = "results") -> str:
    """
    Serialise the results dict to JSON.

    Keys are (regime, activation, m) tuples → converted to "regime__act__m" strings.
    The file is named <log_stem>.json so it matches the log and plots for that run.
    Returns the path of the written file.
    """
    os.makedirs(save_dir, exist_ok=True)
    serialisable = {}
    for (regime, activation, m), val in results.items():
        str_key = f"{regime}__{activation}__{m}"
        serialisable[str_key] = {
            "regime":           regime,
            "activation":       activation,
            "m":                m,
            "train_losses":     val["train_losses"],
            "test_losses":      val["test_losses"],
            "final_train_loss": val["train_losses"][-1],
            "final_test_loss":  val["test_losses"][-1],
        }
    path = os.path.join(save_dir, f"{log_stem}.json")
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    return path


def print_summary_table(results: dict, logger: logging.Logger):
    """Print a formatted table of final train/test losses for every run."""
    col_w = 14
    sep = "─" * (12 + 12 + 8 + col_w * 2 + 6)
    header = f"{'Regime':<12} {'Activation':<12} {'m':<8} " \
             f"{'Train Loss':>{col_w}} {'Test Loss':>{col_w}}"
    logger.info(sep)
    logger.info("  FINAL LOSSES SUMMARY")
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for key in sorted(results.keys()):
        regime, activation, m = key
        trl = results[key]["train_losses"][-1]
        tel = results[key]["test_losses"][-1]
        logger.info(
            f"{regime:<12} {activation:<12} {m:<8} "
            f"{trl:>{col_w}.6f} {tel:>{col_w}.6f}"
        )
    logger.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _width_colormap(widths):
    """Return a dict mapping each width to a colour from the viridis palette."""
    cmap = plt.cm.viridis
    n = max(len(widths) - 1, 1)
    return {m: cmap(i / n) for i, m in enumerate(sorted(widths))}


def _iters_axis(losses, log_interval):
    """Iteration numbers corresponding to the recorded loss values."""
    return np.arange(1, len(losses) + 1) * log_interval


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1 — Training curves (per regime × activation)
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(results: dict, save_dir: str, log_interval: int = 1):
    """
    For every (regime, activation) pair, create a figure with two panels:
    train loss and test loss vs iteration, one curve per width m.

    Saved as: curves_{regime}_{activation}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    regimes     = sorted({k[0] for k in results})
    activations = sorted({k[1] for k in results})
    widths      = sorted({k[2] for k in results})
    w_colors    = _width_colormap(widths)

    for regime in regimes:
        for activation in activations:
            keys_present = [k for k in results if k[0] == regime and k[1] == activation]
            if not keys_present:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
            title = (f"{REGIME_LABELS.get(regime, regime)} "
                     f"— {activation} activation")
            fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

            for m in sorted(widths):
                key = (regime, activation, m)
                if key not in results:
                    continue
                r    = results[key]
                iters = _iters_axis(r["train_losses"], log_interval)
                col  = w_colors[m]
                axes[0].semilogy(iters, r["train_losses"], color=col,
                                 linewidth=1.5, label=f"m={m}")
                axes[1].semilogy(iters, r["test_losses"],  color=col,
                                 linewidth=1.5, label=f"m={m}")

            for ax, title_ax in zip(axes, ["Training Loss (MSE)", "Test Loss (MSE)"]):
                ax.set_xlabel("Iteration")
                ax.set_ylabel("MSE (log scale)")
                ax.set_title(title_ax)
                ax.legend(fontsize=8, loc="upper right")

            plt.tight_layout()
            fname = os.path.join(save_dir, f"curves_{regime}_{activation}.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2 — Final test loss vs width m (scaling)
# ──────────────────────────────────────────────────────────────────────────────

def plot_test_loss_vs_width(results: dict, save_dir: str):
    """
    For each regime, plot final test loss vs m on a log–log scale.
    One line per activation function.

    Saved as: width_scaling_{regime}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    regimes     = sorted({k[0] for k in results})
    activations = sorted({k[1] for k in results})
    widths      = sorted({k[2] for k in results})

    for regime in regimes:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title(
            f"Test Loss vs Width — {REGIME_LABELS.get(regime, regime)}",
            fontsize=12, fontweight="bold"
        )
        any_plotted = False
        for activation in activations:
            ms, losses = [], []
            for m in widths:
                key = (regime, activation, m)
                if key in results:
                    ms.append(m)
                    losses.append(results[key]["test_losses"][-1])
            if ms:
                ax.loglog(
                    ms, losses,
                    marker=ACT_MARKERS.get(activation, "o"),
                    linestyle=ACT_LINESTYLES.get(activation, "-"),
                    linewidth=1.8, markersize=7,
                    label=activation,
                )
                any_plotted = True

        if any_plotted:
            ax.set_xlabel("Width m")
            ax.set_ylabel("Final Test MSE")
            ax.legend()
            plt.tight_layout()
            fname = os.path.join(save_dir, f"width_scaling_{regime}.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3 — Regime comparison at fixed m
# ──────────────────────────────────────────────────────────────────────────────

def plot_regime_comparison(results: dict, save_dir: str, fixed_m: int = None):
    """
    At a fixed width (default: largest available), compare all three regimes.
    One figure per activation, two panels: train loss / test loss vs iteration.

    Saved as: regime_comparison_{activation}_m{m}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    widths      = sorted({k[2] for k in results})
    if fixed_m is None:
        fixed_m = widths[-1]

    regimes     = sorted({k[0] for k in results})
    activations = sorted({k[1] for k in results})

    for activation in activations:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
        fig.suptitle(
            f"Regime Comparison — {activation} activation  (m={fixed_m})",
            fontsize=12, fontweight="bold", y=1.01
        )

        for regime in regimes:
            key = (regime, activation, fixed_m)
            if key not in results:
                continue
            r     = results[key]
            iters = np.arange(1, len(r["train_losses"]) + 1)
            col   = REGIME_COLORS.get(regime, "gray")
            lbl   = REGIME_LABELS.get(regime, regime)
            axes[0].semilogy(iters, r["train_losses"], color=col,
                             linewidth=2, label=lbl)
            axes[1].semilogy(iters, r["test_losses"],  color=col,
                             linewidth=2, label=lbl)

        for ax, title_ax in zip(axes, ["Training Loss (MSE)", "Test Loss (MSE)"]):
            ax.set_xlabel("Iteration")
            ax.set_ylabel("MSE (log scale)")
            ax.set_title(title_ax)
            ax.legend()

        plt.tight_layout()
        fname = os.path.join(
            save_dir, f"regime_comparison_{activation}_m{fixed_m}.png"
        )
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 4 — Heatmap of final test losses
# ──────────────────────────────────────────────────────────────────────────────

def plot_final_loss_heatmap(results: dict, save_dir: str):
    """
    Heatmap grid: activation (rows) × width m (columns), one subplot per regime.
    Cell values are the final test MSE.

    Saved as: heatmap_final_test_loss.png
    """
    os.makedirs(save_dir, exist_ok=True)

    regimes     = sorted({k[0] for k in results})
    activations = sorted({k[1] for k in results})
    widths      = sorted({k[2] for k in results})

    n_regimes = len(regimes)
    fig, axes = plt.subplots(1, n_regimes, figsize=(5.5 * n_regimes, 4.0))
    if n_regimes == 1:
        axes = [axes]

    for ax, regime in zip(axes, regimes):
        matrix = np.full((len(activations), len(widths)), np.nan)
        for i, act in enumerate(activations):
            for j, m in enumerate(widths):
                key = (regime, act, m)
                if key in results:
                    matrix[i, j] = results[key]["test_losses"][-1]

        vmax = np.nanmax(matrix)
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="Final Test MSE", shrink=0.8)

        ax.set_title(REGIME_LABELS.get(regime, regime),
                     fontweight="bold", fontsize=11)
        ax.set_xticks(range(len(widths)))
        ax.set_xticklabels([str(m) for m in widths])
        ax.set_yticks(range(len(activations)))
        ax.set_yticklabels(activations)
        ax.set_xlabel("Width m")
        ax.set_ylabel("Activation")

        # Annotate cells with numeric values
        for i in range(len(activations)):
            for j in range(len(widths)):
                if not np.isnan(matrix[i, j]):
                    brightness = matrix[i, j] / (vmax + 1e-12)
                    txt_color  = "white" if brightness > 0.6 else "black"
                    ax.text(j, i, f"{matrix[i, j]:.4f}",
                            ha="center", va="center",
                            fontsize=8, color=txt_color)

    fig.suptitle("Final Test Loss Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(save_dir, "heatmap_final_test_loss.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 5 — All-in-one overview: final test loss bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_final_loss_bars(results: dict, save_dir: str):
    """
    Grouped bar chart of final test loss, grouped by activation,
    one group of bars per regime, one figure per width m.

    Saved as: bars_final_test_loss_m{m}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    regimes     = sorted({k[0] for k in results})
    activations = sorted({k[1] for k in results})
    widths      = sorted({k[2] for k in results})

    for m in widths:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.set_title(f"Final Test Loss  (m={m})", fontsize=12, fontweight="bold")

        n_act   = len(activations)
        n_reg   = len(regimes)
        x       = np.arange(n_act)
        width_b = 0.8 / n_reg   # bar width

        for reg_idx, regime in enumerate(regimes):
            vals = []
            for activation in activations:
                key = (regime, activation, m)
                vals.append(results[key]["test_losses"][-1] if key in results else 0.0)

            offset = (reg_idx - (n_reg - 1) / 2) * width_b
            bars = ax.bar(
                x + offset, vals, width_b,
                label=REGIME_LABELS.get(regime, regime),
                color=REGIME_COLORS.get(regime, "gray"),
                edgecolor="white", linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(activations)
        ax.set_xlabel("Activation")
        ax.set_ylabel("Final Test MSE")
        ax.legend()
        plt.tight_layout()
        fname = os.path.join(save_dir, f"bars_final_test_loss_m{m}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

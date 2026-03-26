"""
utils.py — Logging, plotting, and results utilities for MATH562 Experiment 2.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Aesthetics
# ──────────────────────────────────────────────────────────────────────────────

REGIME_COLORS = {"NTK": "#1f77b4", "MF": "#ff7f0e", "RF": "#2ca02c"}
REGIME_LABELS = {"NTK": "NTK", "MF": "Mean Field", "RF": "Random Features"}
ACT_MARKERS = {"relu": "o", "erf": "s", "tanh": "^"}
ACT_LINESTYLES = {"relu": "-", "erf": "--", "tanh": "-."}

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


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


def _get_scaling_type(alpha, prefactor):
    if alpha == -0.5:
        m_part = r"\frac{1}{\sqrt{m}}"
    elif alpha == 0.5:
        m_part = r"\sqrt{m}"
    elif alpha == 1.0:
        m_part = r"m"
    elif alpha == 2.0:
        m_part = r"m^2"
    elif alpha == 0.0:
        m_part = ""
    else:
        m_part = rf"m^{{{alpha}}}"

    if prefactor == 1.0:
        return rf"$\eta = \beta \times {m_part}$"

    if alpha == -0.5:
        return rf"$\eta = \frac{{{prefactor}\beta}}{{\sqrt{{m}}}}$"

    return rf"$\eta = {prefactor}\beta {m_part}$"


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1 — Training curves (per regime × prefactor x alpha)
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(args, results: dict, save_dir: str, log_interval: int = 1):
    """
    For every (regime, activation) pair, create a figure with two panels:
    train loss and test loss vs iteration, one curve per width m.

    Saved as: curves_{regime}_{activation}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    regimes = sorted({k[0] for k in results})
    alphas = sorted({k[1] for k in results})
    prefactors = sorted({k[2] for k in results})
    widths = sorted({k[3] for k in results})
    w_colors = _width_colormap(widths)

    for regime in regimes:
        for alpha in alphas:
            for prefactor in prefactors:
                keys_present = [k for k in results if k[0] == regime and k[1] == alpha and k[2] == prefactor]
                if not keys_present:
                    continue

                fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

                scaling_type = _get_scaling_type(alpha, prefactor)

                title = (rf"{REGIME_LABELS.get(regime, regime)} "
                         rf"— {scaling_type} "
                         rf"- Beta: {args.beta}")
                fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

                for m in sorted(widths):
                    key = (regime, alpha, prefactor, m)
                    if key not in results:
                        continue
                    r = results[key]
                    iters = _iters_axis(r["train_losses"], log_interval)
                    col = w_colors[m]
                    axes[0].semilogy(iters, r["train_losses"], color=col,
                                     linewidth=1.5, label=f"m={m}")
                    axes[1].semilogy(iters, r["test_losses"], color=col,
                                     linewidth=1.5, label=f"m={m}")

                for ax, title_ax in zip(axes, ["Training Loss (MSE)", "Test Loss (MSE)"]):
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("MSE (log scale)")
                    ax.set_title(title_ax)
                    ax.legend(fontsize=8, loc="upper right")

                plt.tight_layout()
                fname = os.path.join(save_dir,
                                     f"curves_{regime}_beta_{args.beta}_alpha_{alpha}_prefactor_{prefactor}.png")
                fig.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close(fig)

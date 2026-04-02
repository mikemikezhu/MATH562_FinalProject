"""
utils.py — Logging, plotting, and results utilities for MATH562 Experiment 1.
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

    regimes = sorted({k[0] for k in results})
    activations = sorted({k[1] for k in results})
    widths = sorted({k[2] for k in results})
    w_colors = _width_colormap(widths)

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

    regimes = sorted({k[0] for k in results})
    activations = sorted({k[1] for k in results})
    widths = sorted({k[2] for k in results})

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

def plot_regime_comparison(results: dict, save_dir: str, fixed_m: int = None, log_interval: int = 1):
    """
    At a fixed width (default: largest available), compare all three regimes.
    One figure per activation, two panels: train loss / test loss vs iteration.

    Saved as: regime_comparison_{activation}_m{m}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    widths = sorted({k[2] for k in results})
    if fixed_m is None:
        fixed_m = widths[-1]

    regimes = sorted({k[0] for k in results})
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
            r = results[key]
            iters = _iters_axis(r["train_losses"], log_interval)
            col = REGIME_COLORS.get(regime, "gray")
            lbl = REGIME_LABELS.get(regime, regime)
            axes[0].semilogy(iters, r["train_losses"], color=col,
                             linewidth=2, label=lbl)
            axes[1].semilogy(iters, r["test_losses"], color=col,
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

    regimes = sorted({k[0] for k in results})
    activations = sorted({k[1] for k in results})
    widths = sorted({k[2] for k in results})

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
                    txt_color = "white" if brightness > 0.6 else "black"
                    ax.text(j, i, f"{matrix[i, j]:.4f}",
                            ha="center", va="center",
                            fontsize=8, color=txt_color)

    fig.suptitle("Final Test Loss Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(save_dir, "heatmap_final_test_loss.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Report Figure 1 — Training curves grid (regimes × train/test)
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves_grid(results: dict, save_dir: str,
                              log_interval: int = 1,
                              ref_activation: str = "relu"):
    """
    Grid of training curves, one figure per activation.
    Rows = regimes, cols = (Train Loss, Test Loss), curves coloured by width m.

    Saved as: training_curves_grid_{activation}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    regimes = sorted({k[0] for k in results})
    widths = sorted({k[2] for k in results})
    activations = sorted({k[1] for k in results})
    w_colors = _width_colormap(widths)

    n_rows = len(regimes)

    for activation in activations:
        fig, axes = plt.subplots(n_rows, 2, figsize=(13, 4.5 * n_rows), sharey=False)
        if n_rows == 1:
            axes = [axes]  # keep as list of rows

        fig.suptitle(f"Training Dynamics — {activation} activation",
                     fontsize=13, fontweight="bold", y=1.01)

        for row, regime in enumerate(regimes):
            ax_train, ax_test = axes[row]
            row_label = REGIME_LABELS.get(regime, regime)

            for m in widths:
                key = (regime, activation, m)
                if key not in results:
                    continue
                r = results[key]
                iters = _iters_axis(r["train_losses"], log_interval)
                col = w_colors[m]
                ax_train.semilogy(iters, r["train_losses"], color=col,
                                  linewidth=1.5, label=f"m={m}")
                ax_test.semilogy(iters, r["test_losses"], color=col,
                                 linewidth=1.5, label=f"m={m}")

            for ax, title_ax in zip([ax_train, ax_test],
                                     ["Training Loss (MSE)", "Test Loss (MSE)"]):
                ax.set_xlabel("Iteration")
                ax.set_ylabel("MSE (log scale)")
                ax.set_title(f"{row_label} — {title_ax}")
                ax.legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        fname = os.path.join(save_dir, f"training_curves_grid_{activation}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Report Figure 2 — Regime comparison grid (activations × train/test)
# ──────────────────────────────────────────────────────────────────────────────

def plot_regime_comparison_grid(results: dict, save_dir: str,
                                fixed_m: int = None, log_interval: int = 1):
    """
    Grid comparing all regimes, one figure per width m.
    Rows = activations, cols = (Train Loss, Test Loss), curves coloured by regime.

    Saved as: regime_comparison_grid_m{m}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    widths = sorted({k[2] for k in results})
    regimes = sorted({k[0] for k in results})
    activations = sorted({k[1] for k in results})

    # If a specific fixed_m is requested, only plot that one; otherwise all widths
    m_values = [fixed_m] if fixed_m is not None else widths

    n_rows = len(activations)

    for m in m_values:
        fig, axes = plt.subplots(n_rows, 2, figsize=(13, 4.5 * n_rows), sharey=False)
        if n_rows == 1:
            axes = [axes]

        fig.suptitle(f"Regime Comparison at m={m}",
                     fontsize=13, fontweight="bold", y=1.01)

        for row, activation in enumerate(activations):
            ax_train, ax_test = axes[row]

            for regime in regimes:
                key = (regime, activation, m)
                if key not in results:
                    continue
                r = results[key]
                iters = _iters_axis(r["train_losses"], log_interval)
                col = REGIME_COLORS.get(regime, "gray")
                lbl = REGIME_LABELS.get(regime, regime)
                ax_train.semilogy(iters, r["train_losses"], color=col,
                                  linewidth=2, label=lbl)
                ax_test.semilogy(iters, r["test_losses"], color=col,
                                 linewidth=2, label=lbl)

            for ax, title_ax in zip([ax_train, ax_test],
                                     ["Training Loss (MSE)", "Test Loss (MSE)"]):
                ax.set_xlabel("Iteration")
                ax.set_ylabel("MSE (log scale)")
                ax.set_title(f"{activation} — {title_ax}")
                ax.legend(fontsize=9)

        plt.tight_layout()
        fname = os.path.join(save_dir, f"regime_comparison_grid_m{m}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Report Figure 3 — Width scaling grid (one subplot per regime)
# ──────────────────────────────────────────────────────────────────────────────

def plot_width_scaling_grid(results: dict, save_dir: str):
    """
    Side-by-side log-log plots of final test loss vs width, one per regime.
    Lines coloured/styled by activation function.

    Saved as: width_scaling_grid.png
    """
    os.makedirs(save_dir, exist_ok=True)

    regimes = sorted({k[0] for k in results})
    activations = sorted({k[1] for k in results})
    widths = sorted({k[2] for k in results})

    n_cols = len(regimes)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5), sharey=False)
    if n_cols == 1:
        axes = [axes]

    fig.suptitle("Final Test Loss vs Width", fontsize=13, fontweight="bold")

    for ax, regime in zip(axes, regimes):
        ax.set_title(REGIME_LABELS.get(regime, regime), fontweight="bold")
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
        ax.set_xlabel("Width m")
        ax.set_ylabel("Final Test MSE")
        ax.legend()

    plt.tight_layout()
    fname = os.path.join(save_dir, "width_scaling_grid.png")
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

    regimes = sorted({k[0] for k in results})
    activations = sorted({k[1] for k in results})
    widths = sorted({k[2] for k in results})

    for m in widths:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.set_title(f"Final Test Loss  (m={m})", fontsize=12, fontweight="bold")

        n_act = len(activations)
        n_reg = len(regimes)
        x = np.arange(n_act)
        width_b = 0.8 / n_reg  # bar width

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

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # safe for scripts / remote runs
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Aesthetics
# ──────────────────────────────────────────────────────────────────────────────

REGIME_COLORS = {"NTK": "#1f77b4", "MF": "#ff7f0e", "RF": "#2ca02c"}
REGIME_LABELS = {"NTK": "NTK", "MF": "Mean Field", "RF": "Random Features"}
METRIC_LABELS = {
    "fro_norm": "Frobenius norm",
    "abs_change": "Absolute change",
    "rel_change": "Relative change",
    "mean_value": "Mean kernel value",
}

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

def _checkpoint_axis(metrics_dict: dict):
    """Sorted checkpoint list."""
    return sorted(metrics_dict.keys())


def _extract_metric_series(metrics_dict: dict, metric_name: str):
    """
    From
        metrics_dict = {
            0: {"fro_norm": ..., ...},
            25: {...},
            ...
        }
    return
        checkpoints, values
    """
    checkpoints = sorted(metrics_dict.keys())
    values = [metrics_dict[t][metric_name] for t in checkpoints]
    return checkpoints, values


def _width_colormap(widths):
    """Return a dict mapping width m to a colour."""
    cmap = plt.cm.viridis
    n = max(len(widths) - 1, 1)
    return {m: cmap(i / n) for i, m in enumerate(sorted(widths))}


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1 — All kernel metrics for one run
# ──────────────────────────────────────────────────────────────────────────────

def plot_kernel_metrics(results: dict, save_dir: str):
    """
    For each (regime, activation, m, beta_scaling) run, create a 2x2 figure with:
      - Frobenius norm vs checkpoint
      - Absolute change vs checkpoint
      - Relative change vs checkpoint
      - Mean kernel value vs checkpoint

    Saved as:
      kernel_metrics_{regime}_{activation}_m{m}_scale_{beta_scaling}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    for key in sorted(results.keys()):
        regime, activation, m, beta_scaling = key
        metrics = results[key].get("metrics", {})
        if not metrics:
            continue

        meta = results[key]["metadata"]
        default_beta = meta["default_beta"]
        effective_beta = meta["effective_beta"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            f"{REGIME_LABELS.get(regime, regime)} — {activation} activation — m={m}\n"
            f"scale={beta_scaling:g}, β0={default_beta:g}, β={effective_beta:g}",
            fontsize=12,
            fontweight="bold",
            y=1.02
        )

        plot_specs = [
            ("fro_norm", "Frobenius Norm", axes[0, 0]),
            ("abs_change", "Absolute Change from $K_0$", axes[0, 1]),
            ("rel_change", "Relative Change from $K_0$", axes[1, 0]),
            ("mean_value", "Mean Kernel Value", axes[1, 1]),
        ]

        color = REGIME_COLORS.get(regime, "gray")

        for metric_name, title, ax in plot_specs:
            checkpoints, values = _extract_metric_series(metrics, metric_name)
            ax.plot(checkpoints, values, marker="o", linewidth=1.8, color=color)
            ax.set_title(title)
            ax.set_xlabel("Checkpoint / Iteration")
            ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name))

        plt.tight_layout()
        scale_str = f"{beta_scaling:g}"
        fname = os.path.join(
            save_dir,
            f"kernel_metrics_{regime}_{activation}_m{m}_scale_{scale_str}.png"
        )
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2 — Heatmap of kernel matrix at one checkpoint
# ──────────────────────────────────────────────────────────────────────────────

def plot_kernel_heatmap(K: np.ndarray, checkpoint: int, regime: str,
                        activation: str, m: int, beta_scaling: float,
                        default_beta: float, effective_beta: float,
                        save_dir: str, prefix: str = "kernel_heatmap"):
    """
    Plot a single heatmap of the kernel matrix K.

    Saved as:
      {prefix}_{regime}_{activation}_m{m}_scale_{beta_scaling}_ckpt{checkpoint}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(K, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, shrink=0.85, label="Kernel value")

    ax.set_title(
        f"{REGIME_LABELS.get(regime, regime)} — {activation} — m={m} — checkpoint {checkpoint}\n"
        f"scale={beta_scaling:g}, β0={default_beta:g}, β={effective_beta:g}",
        fontsize=11,
        fontweight="bold"
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Sample index")

    plt.tight_layout()
    scale_str = f"{beta_scaling:g}"
    fname = os.path.join(
        save_dir,
        f"{prefix}_{regime}_{activation}_m{m}_scale_{scale_str}_ckpt{checkpoint}.png"
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3 — Initial/final heatmaps for each run
# ──────────────────────────────────────────────────────────────────────────────

def plot_initial_final_heatmaps(results: dict, save_dir: str):
    """
    For each (regime, activation, m, beta_scaling) run, plot side-by-side heatmaps of:
      - initial kernel K_0
      - final kernel K_T

    Saved as:
      kernel_heatmaps_{regime}_{activation}_m{m}_scale_{beta_scaling}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    for key in sorted(results.keys()):
        regime, activation, m, beta_scaling = key
        kernels = results[key].get("kernels", {})
        if not kernels:
            continue

        meta = results[key]["metadata"]
        default_beta = meta["default_beta"]
        effective_beta = meta["effective_beta"]

        checkpoints = sorted(kernels.keys())
        t0 = checkpoints[0]
        tT = checkpoints[-1]

        K0 = np.asarray(kernels[t0])
        KT = np.asarray(kernels[tT])

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle(
            f"{REGIME_LABELS.get(regime, regime)} — {activation} activation — m={m}\n"
            f"scale={beta_scaling:g}, β0={default_beta:g}, β={effective_beta:g}",
            fontsize=12,
            fontweight="bold",
            y=1.02
        )

        vmin = min(K0.min(), KT.min())
        vmax = max(K0.max(), KT.max())

        im0 = axes[0].imshow(K0, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Initial kernel (checkpoint {t0})")
        axes[0].set_xlabel("Sample index")
        axes[0].set_ylabel("Sample index")

        im1 = axes[1].imshow(KT, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Final kernel (checkpoint {tT})")
        axes[1].set_xlabel("Sample index")
        axes[1].set_ylabel("Sample index")

        plt.colorbar(im0, ax=axes[0], shrink=0.8)
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        plt.tight_layout()
        scale_str = f"{beta_scaling:g}"
        fname = os.path.join(
            save_dir,
            f"kernel_heatmaps_{regime}_{activation}_m{m}_scale_{scale_str}.png"
        )
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 4 — Compare one metric across widths m
# ──────────────────────────────────────────────────────────────────────────────

def plot_metric_vs_checkpoint_by_width(results: dict, save_dir: str,
                                       regime: str, activation: str,
                                       beta_scaling: float,
                                       metric_name: str = "rel_change"):
    """
    For a fixed (regime, activation, beta_scaling), plot one metric vs checkpoint,
    with one curve per width m.

    Useful for showing, for example:
      - NTK relative change gets smaller as m increases.

    Saved as:
      compare_{metric_name}_{regime}_{activation}_scale_{beta_scaling}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    keys_present = [
        k for k in results
        if k[0] == regime and k[1] == activation and k[3] == beta_scaling
    ]
    if not keys_present:
        return

    widths = sorted({k[2] for k in keys_present})
    w_colors = _width_colormap(widths)

    fig, ax = plt.subplots(figsize=(8, 5))

    sample_key = (regime, activation, widths[0], beta_scaling)
    meta = results[sample_key]["metadata"]

    ax.set_title(
        f"{metric_name} vs checkpoint — {REGIME_LABELS.get(regime, regime)} — {activation}\n"
        f"scale={beta_scaling:g}, β0={meta['default_beta']:g}, β={meta['effective_beta']:g}",
        fontsize=12,
        fontweight="bold"
    )

    for m in widths:
        key = (regime, activation, m, beta_scaling)
        if key not in results:
            continue

        metrics = results[key].get("metrics", {})
        if not metrics:
            continue

        checkpoints, values = _extract_metric_series(metrics, metric_name)
        ax.plot(
            checkpoints,
            values,
            marker="o",
            linewidth=1.8,
            color=w_colors[m],
            label=f"m={m}"
        )

    ax.set_xlabel("Checkpoint / Iteration")
    ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name))
    ax.legend(fontsize=8)
    plt.tight_layout()

    scale_str = f"{beta_scaling:g}"
    fname = os.path.join(
        save_dir,
        f"compare_{metric_name}_{regime}_{activation}_scale_{scale_str}.png"
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 5 — Compare one metric across regimes
# ──────────────────────────────────────────────────────────────────────────────

def plot_metric_vs_checkpoint_by_regime(results: dict, save_dir: str,
                                        activation: str, m: int,
                                        beta_scaling: float,
                                        metric_name: str = "rel_change"):
    """
    For a fixed (activation, m, beta_scaling), compare regimes on one metric vs checkpoint.

    Useful for showing:
      RF stays constant, NTK changes a little, MF changes more.

    Saved as:
      regime_compare_{metric_name}_{activation}_m{m}_scale_{beta_scaling}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    keys_present = [
        k for k in results
        if k[1] == activation and k[2] == m and k[3] == beta_scaling
    ]
    if not keys_present:
        return

    regimes = sorted({k[0] for k in keys_present})
    fig, ax = plt.subplots(figsize=(8, 5))

    sample_key = keys_present[0]
    meta = results[sample_key]["metadata"]

    ax.set_title(
        f"{metric_name} vs checkpoint — {activation} activation — m={m}\n"
        f"scale={beta_scaling:g}, β0={meta['default_beta']:g}, β={meta['effective_beta']:g}",
        fontsize=12,
        fontweight="bold"
    )

    any_plotted = False
    for regime in regimes:
        key = (regime, activation, m, beta_scaling)
        if key not in results:
            continue

        metrics = results[key].get("metrics", {})
        if not metrics:
            continue

        checkpoints, values = _extract_metric_series(metrics, metric_name)
        ax.plot(
            checkpoints,
            values,
            marker="o",
            linewidth=2,
            color=REGIME_COLORS.get(regime, "gray"),
            label=REGIME_LABELS.get(regime, regime)
        )
        any_plotted = True

    if any_plotted:
        ax.set_xlabel("Checkpoint / Iteration")
        ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name))
        ax.legend()
        plt.tight_layout()

        scale_str = f"{beta_scaling:g}"
        fname = os.path.join(
            save_dir,
            f"regime_compare_{metric_name}_{activation}_m{m}_scale_{scale_str}.png"
        )
        fig.savefig(fname, dpi=150, bbox_inches="tight")

    plt.close(fig)


def plot_kernel_metrics_by_width(results: dict, save_dir: str,
                                 regime: str, activation: str, beta_scaling: float):
    """
    For a fixed (regime, activation, beta_scaling), create one 2x2 figure with:
      - Frobenius norm
      - Absolute change
      - Relative change
      - Mean kernel value

    Each subplot contains one curve per width m.

    Saved as:
      kernel_metrics_by_width_{regime}_{activation}_scale_{beta_scaling}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    keys_present = [
        k for k in results
        if k[0] == regime and k[1] == activation and k[3] == beta_scaling
    ]
    if not keys_present:
        return

    widths = sorted({k[2] for k in keys_present})
    w_colors = _width_colormap(widths)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    sample_key = (regime, activation, widths[0], beta_scaling)
    meta = results[sample_key]["metadata"]
    default_beta = meta["default_beta"]
    effective_beta = meta["effective_beta"]

    fig.suptitle(
        f"{REGIME_LABELS.get(regime, regime)} — {activation} activation\n"
        f"default β={default_beta:g}, scale={beta_scaling:g}, effective β={effective_beta:g}",
        fontsize=12,
        fontweight="bold",
        y=1.02
    )

    plot_specs = [
        ("fro_norm", "Frobenius Norm", axes[0, 0]),
        ("abs_change", "Absolute Change from $K_0$", axes[0, 1]),
        ("rel_change", "Relative Change from $K_0$", axes[1, 0]),
        ("mean_value", "Mean Kernel Value", axes[1, 1]),
    ]

    for metric_name, title, ax in plot_specs:
        for m in widths:
            key = (regime, activation, m, beta_scaling)
            if key not in results:
                continue

            metrics = results[key].get("metrics", {})
            if not metrics:
                continue

            checkpoints, values = _extract_metric_series(metrics, metric_name)
            ax.plot(
                checkpoints,
                values,
                marker="o",
                linewidth=1.8,
                color=w_colors[m],
                label=f"m={m}"
            )

        ax.set_title(title)
        ax.set_xlabel("Checkpoint / Iteration")
        ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=min(len(widths), 6),
            bbox_to_anchor=(0.5, 0.98),
            frameon=True
        )

    plt.tight_layout()
    scale_str = f"{beta_scaling:g}"
    fname = os.path.join(
        save_dir,
        f"kernel_metrics_by_width_{regime}_{activation}_scale_{scale_str}.png"
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rel_change_by_beta_scaling(results: dict, save_dir: str,
                                    regime: str, activation: str, m: int):
    """
    For a fixed (regime, activation, m), compare different beta scalings
    using relative change vs checkpoint, with one curve per beta scaling.

    Saved as:
      rel_change_by_beta_{regime}_{activation}_m{m}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    keys_present = [
        k for k in results
        if k[0] == regime and k[1] == activation and k[2] == m
    ]
    if not keys_present:
        return

    beta_scalings = sorted({k[3] for k in keys_present})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        f"Relative change vs checkpoint — {REGIME_LABELS.get(regime, regime)} — "
        f"{activation} — m={m}",
        fontsize=12,
        fontweight="bold"
    )

    for beta_scaling in beta_scalings:
        key = (regime, activation, m, beta_scaling)
        if key not in results:
            continue

        metrics = results[key].get("metrics", {})
        if not metrics:
            continue

        checkpoints, values = _extract_metric_series(metrics, "rel_change")
        meta = results[key]["metadata"]

        ax.plot(
            checkpoints,
            values,
            marker="o",
            linewidth=1.8,
            label=(
                f"scale={beta_scaling:g} "
                f"(β0={meta['default_beta']:g}, β={meta['effective_beta']:g})"
            )
        )

    ax.set_xlabel("Checkpoint / Iteration")
    ax.set_ylabel("Relative change")
    ax.legend(fontsize=8)
    plt.tight_layout()

    fname = os.path.join(
        save_dir,
        f"rel_change_by_beta_{regime}_{activation}_m{m}.png"
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ntk_final_rel_change_vs_width(results: dict, save_dir: str,
                                       activation: str = "tanh",
                                       beta_scaling: float = 1.0):
    """
    Plot final relative change vs width m for NTK regime only.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Filter relevant runs
    keys_present = [
        k for k in results
        if k[0] == "NTK" and k[1] == activation and k[3] == beta_scaling
    ]
    if not keys_present:
        return

    # Sort by width
    widths = sorted([k[2] for k in keys_present])

    final_rel_changes = []

    for m in widths:
        key = ("NTK", activation, m, beta_scaling)
        if key not in results:
            continue

        metrics = results[key].get("metrics", {})
        if not metrics:
            continue

        checkpoints = sorted(metrics.keys())
        final_ckpt = checkpoints[-1]
        final_rel_change = metrics[final_ckpt]["rel_change"]

        final_rel_changes.append(final_rel_change)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        widths,
        final_rel_changes,
        marker="o",
        linewidth=2,
        color=REGIME_COLORS["NTK"]
    )

    ax.set_title(
        "NTK kernel constancy: width vs final relative change",
        fontsize=12,
        fontweight="bold"
    )
    ax.set_xlabel("Width m")
    ax.set_ylabel("Final relative change")

    # ✅ KEY FIXES
    # Remove log scale → equal spacing
    # ax.set_xscale("log")  ← DELETE THIS

    # Force ticks to be exactly your widths (100, 200, ...)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])

    plt.tight_layout()

    fname = os.path.join(
        save_dir,
        f"ntk_final_rel_change_vs_width_{activation}_scale_{beta_scaling:g}.png"
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mf_rel_change_vs_checkpoint_by_width(results: dict, save_dir: str,
                                              activation: str = "tanh",
                                              beta_scaling: float = 1.0):
    """
    Plot relative change vs checkpoint for MF regime, with one curve per width.

    Default:
      - regime = MF
      - activation = tanh
      - beta_scaling = 1

    Saved as:
      mf_rel_change_vs_checkpoint_by_width_{activation}_scale_{beta_scaling}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    keys_present = [
        k for k in results
        if k[0] == "MF" and k[1] == activation and k[3] == beta_scaling
    ]
    if not keys_present:
        return

    widths = sorted({k[2] for k in keys_present})
    w_colors = _width_colormap(widths)

    fig, ax = plt.subplots(figsize=(8, 5))

    for m in widths:
        key = ("MF", activation, m, beta_scaling)
        if key not in results:
            continue

        metrics = results[key].get("metrics", {})
        if not metrics:
            continue

        checkpoints = sorted(metrics.keys())
        values = [metrics[t]["rel_change"] for t in checkpoints]

        ax.plot(
            checkpoints,
            values,
            marker="o",
            linewidth=2,
            color=w_colors[m],
            label=f"m={m}"
        )

    ax.set_title(
        f"MF relative change vs checkpoint — {activation} — scale={beta_scaling:g}",
        fontsize=12,
        fontweight="bold"
    )
    ax.set_xlabel("Checkpoint / Iteration")
    ax.set_ylabel("Relative change")
    ax.legend(fontsize=9)

    plt.tight_layout()

    fname = os.path.join(
        save_dir,
        f"mf_rel_change_vs_checkpoint_by_width_{activation}_scale_{beta_scaling:g}.png"
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
"""
utils.py — Logging, plotting, and results utilities for MATH562 Experiment 3.
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


# TODO: add whatever is needed


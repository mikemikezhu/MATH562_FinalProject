"""
main.py — Experiment 1: Performance Comparison Across Regimes (MATH562)

Usage examples
--------------
# Conservative default run
python main.py

# Custom widths and more iterations
python main.py --m_values 50 100 200 400 800 --n_iters 2000

# Single regime / activation for quick testing
python main.py --regimes NTK --activations relu --m_values 100 200 --n_iters 200

# Override learning rates per regime
python main.py --lr_ntk 0.5 --lr_mf 0.01 --lr_rf 1.0
"""

import argparse
import os
import time
import numpy as np

from regimes import NTKRegime, MeanFieldRegime, RandomFeaturesRegime
from utils import (
    setup_logger,
    save_results,
    print_summary_table,
    plot_training_curves,
    plot_test_loss_vs_width,
    plot_regime_comparison,
    plot_final_loss_heatmap,
    plot_final_loss_bars,
)

# ──────────────────────────────────────────────────────────────────────────────
# Regime registry
# ──────────────────────────────────────────────────────────────────────────────

REGIME_CLASSES = {
    "NTK": NTKRegime,
    "MF":  MeanFieldRegime,
    "RF":  RandomFeaturesRegime,
}

# Default per-regime learning rates (motivated by Chapter 12 theory:
#   NTK ~ O(1), MF ~ O(1/m), RF ~ O(1))
DEFAULT_LR = {
    "NTK": 0.1,
    "MF":  0.5,
    "RF":  0.5,
}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Experiment 1 — Regime comparison (NTK / MF / RF)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data / ground-truth
    data = p.add_argument_group("Data & ground-truth network")
    data.add_argument("--d",        type=int,   default=10,
                      help="Input dimension")
    data.add_argument("--n",        type=int,   default=3000,
                      help="Number of training samples")
    data.add_argument("--n_test",   type=int,   default=500,
                      help="Number of test samples")
    data.add_argument("--m_star",   type=int,   default=20,
                      help="Hidden width of the ground-truth network")
    data.add_argument("--gt_activation", type=str, default="relu",
                      choices=["relu", "erf", "tanh"],
                      help="Activation of the ground-truth network")
    data.add_argument("--seed",     type=int,   default=42,
                      help="Global RNG seed")

    # ── Experiment grid
    grid = p.add_argument_group("Experiment grid")
    grid.add_argument("--m_values",   type=int, nargs="+",
                      default=[100, 200, 400, 800],
                      help="List of hidden widths m to sweep over")
    grid.add_argument("--activations", type=str, nargs="+",
                      default=["relu", "erf", "tanh"],
                      choices=["relu", "erf", "tanh"],
                      help="Activation functions to test")
    grid.add_argument("--regimes",    type=str, nargs="+",
                      default=["NTK", "MF", "RF"],
                      choices=["NTK", "MF", "RF"],
                      help="Regimes to include")

    # ── Training
    train = p.add_argument_group("Training")
    train.add_argument("--n_iters",   type=int,   default=1000,
                       help="Number of gradient descent iterations")
    train.add_argument("--lr_ntk",    type=float, default=DEFAULT_LR["NTK"],
                       help="Learning rate for the NTK regime")
    train.add_argument("--lr_mf",     type=float, default=DEFAULT_LR["MF"],
                       help="Learning rate for the MF regime")
    train.add_argument("--lr_rf",     type=float, default=DEFAULT_LR["RF"],
                       help="Learning rate for the RF regime")
    train.add_argument("--log_every", type=int,   default=50,
                       help="Record train/test loss every N iterations")

    # ── Output
    out = p.add_argument_group("Output")
    out.add_argument("--out_dir", type=str, default="results",
                     help="Root directory for logs, plots, and JSON results")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────────────────────────────────────

def make_dataset(args, logger):
    """Generate train/test splits from a fixed ground-truth two-layer net."""
    from regimes import BaseRegime

    rng = np.random.default_rng(args.seed)

    logger.info("=" * 60)
    logger.info("DATA GENERATION")
    logger.info("=" * 60)
    logger.info(f"  d={args.d}, n={args.n}, n_test={args.n_test}, "
                f"m*={args.m_star}, activation={args.gt_activation}")

    # Ground-truth network (fixed, not trained)
    gt = BaseRegime(d=args.d, m=args.m_star,
                    activation=args.gt_activation, seed=args.seed)

    # X ~ N(0, I/d)
    cov = np.eye(args.d) / args.d
    X_train = rng.multivariate_normal(np.zeros(args.d), cov, size=args.n)
    X_test  = rng.multivariate_normal(np.zeros(args.d), cov, size=args.n_test)

    y_train = gt.forward(X_train)
    y_test  = gt.forward(X_test)

    logger.info(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
    logger.info(f"  X_test : {X_test.shape}    y_test : {y_test.shape}")
    logger.info(f"  y_train — mean={y_train.mean():.4f}, std={y_train.std():.4f}")
    logger.info(f"  Null MSE (predict 0): {float(np.mean(y_test**2)):.6f}")

    return X_train, y_train, X_test, y_test


# ──────────────────────────────────────────────────────────────────────────────
# Single-run training
# ──────────────────────────────────────────────────────────────────────────────

def run_one(regime_name, activation, m, lr, n_iters, log_every,
            X_train, y_train, X_test, y_test, seed, logger):
    """
    Train one (regime, activation, m) configuration.
    Returns dicts of per-checkpoint train and test losses.
    """
    RegimeClass = REGIME_CLASSES[regime_name]
    model = RegimeClass(d=X_train.shape[1], m=m, activation=activation, seed=seed)

    train_losses, test_losses = [], []
    t0 = time.perf_counter()

    for it in range(1, n_iters + 1):
        model.gradient_descent_step(X_train, y_train, lr)

        if it % log_every == 0 or it == n_iters:
            y_pred_train = model.forward(X_train)
            y_pred_test  = model.forward(X_test)
            trl = model.mse_loss(y_train, y_pred_train)
            tel = model.mse_loss(y_test,  y_pred_test)
            train_losses.append(trl)
            test_losses.append(tel)

            elapsed = time.perf_counter() - t0
            logger.debug(
                f"  [{regime_name:3s}|{activation:4s}|m={m:4d}] "
                f"iter {it:5d}/{n_iters}  "
                f"train={trl:.6f}  test={tel:.6f}  ({elapsed:.1f}s)"
            )

    elapsed = time.perf_counter() - t0
    logger.info(
        f"  DONE  [{regime_name:3s}|{activation:4s}|m={m:4d}]  "
        f"train={train_losses[-1]:.6f}  test={test_losses[-1]:.6f}  "
        f"({elapsed:.1f}s)"
    )
    return train_losses, test_losses


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(args, logger, X_train, y_train, X_test, y_test):
    lr_map = {"NTK": args.lr_ntk, "MF": args.lr_mf, "RF": args.lr_rf}

    total = len(args.regimes) * len(args.activations) * len(args.m_values)
    logger.info("=" * 60)
    logger.info("EXPERIMENT GRID")
    logger.info("=" * 60)
    logger.info(f"  Regimes    : {args.regimes}")
    logger.info(f"  Activations: {args.activations}")
    logger.info(f"  Widths m   : {args.m_values}")
    logger.info(f"  Iterations : {args.n_iters}  (log every {args.log_every})")
    logger.info(f"  LR — NTK={args.lr_ntk}, MF={args.lr_mf}, RF={args.lr_rf}")
    logger.info(f"  Total runs : {total}")
    logger.info("")

    results = {}
    run_idx = 0

    for regime in args.regimes:
        lr = lr_map[regime]
        for activation in args.activations:
            for m in sorted(args.m_values):
                run_idx += 1
                logger.info(
                    f"[{run_idx}/{total}]  regime={regime}  "
                    f"activation={activation}  m={m}  lr={lr}"
                )
                train_losses, test_losses = run_one(
                    regime_name=regime,
                    activation=activation,
                    m=m,
                    lr=lr,
                    n_iters=args.n_iters,
                    log_every=args.log_every,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    seed=args.seed,
                    logger=logger,
                )
                results[(regime, activation, m)] = {
                    "train_losses": [float(v) for v in train_losses],
                    "test_losses":  [float(v) for v in test_losses],
                }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Directories
    plots_dir = os.path.join(args.out_dir, "plots")
    logs_dir  = os.path.join(args.out_dir, "logs")
    os.makedirs(plots_dir, exist_ok=True)

    logger = setup_logger(logs_dir, name="experiment1")

    logger.info("=" * 60)
    logger.info("MATH562 — Experiment 1: Regime Comparison")
    logger.info("=" * 60)
    logger.info(f"Output directory: {os.path.abspath(args.out_dir)}")

    # ── Data
    X_train, y_train, X_test, y_test = make_dataset(args, logger)

    # ── Train
    t_start = time.perf_counter()
    results = run_experiment(args, logger, X_train, y_train, X_test, y_test)
    logger.info(f"\nTotal wall-clock time: {time.perf_counter() - t_start:.1f}s")

    # ── Summary table
    print_summary_table(results, logger)

    # ── Save JSON
    json_path = save_results(results, args.out_dir)
    logger.info(f"Results saved to: {json_path}")

    # ── Plots
    logger.info("Generating plots …")

    plot_training_curves(results, plots_dir, log_interval=args.log_every)
    logger.info("  [1/5] Training curves done")

    plot_test_loss_vs_width(results, plots_dir)
    logger.info("  [2/5] Width scaling plots done")

    plot_regime_comparison(results, plots_dir)
    logger.info("  [3/5] Regime comparison plots done")

    plot_final_loss_heatmap(results, plots_dir)
    logger.info("  [4/5] Heatmap done")

    plot_final_loss_bars(results, plots_dir)
    logger.info("  [5/5] Bar charts done")

    logger.info(f"\nAll plots saved to: {os.path.abspath(plots_dir)}")
    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()

"""
main.py — Experiment 1: Performance Comparison Across Regimes (MATH562)

Usage examples
--------------
# Conservative default run
python component2/exp_1/main.py

# Custom widths and more iterations
python component2/exp_1/main.py --m_values 50 100 200 400 800 --n_iters 2000

# Single regime / activation for quick testing
python component2/exp_1/main.py --regimes NTK --activations relu --m_values 100 200 --n_iters 200

# Override beta (scales learning rate per regime)
python component2/exp_1/main.py --beta 0.5
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import time

import numpy as np

from component2.regimes import NTKRegime, MeanFieldRegime, RandomFeaturesRegime
from component2.utils import (
    setup_logger,
)

from component2.exp_1.plot import (
    plot_training_curves_grid,
    plot_regime_comparison_grid,
    plot_width_scaling_grid,
    plot_final_loss_heatmap,
)

from component2.exp_1.save import (
    save_results,
    print_summary_table,
)

from component2.data_generator import SyntheticDataGenerator, MNISTDataGenerator, CIFAR10DataGenerator

# ──────────────────────────────────────────────────────────────────────────────
# Regime registry
# ──────────────────────────────────────────────────────────────────────────────

REGIME_CLASSES = {
    "NTK": NTKRegime,
    "MF": MeanFieldRegime,
    "RF": RandomFeaturesRegime,
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
    data.add_argument("--dataset", type=str, default="synthetic",
                      choices=["synthetic", "mnist", "cifar10"],
                      help="Dataset to use")
    data.add_argument("--classes", type=int, nargs=2, default=[0, 1],
                      metavar=("CLASS_A", "CLASS_B"),
                      help="Two classes for binary classification (mnist/cifar10 only)")
    data.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                      help="Normalize real-data features to zero mean, unit std")
    data.add_argument("--d", type=int, default=10,
                      help="Input dimension (synthetic only; derived from data for real datasets)")
    data.add_argument("--n", type=int, default=None,
                      help="Training samples. Defaults: 3000 (synthetic), 5000 (real)")
    data.add_argument("--n_test", type=int, default=None,
                      help="Test samples. Defaults: 500 (synthetic), 1250 (real)")
    data.add_argument("--m_star", type=int, default=20,
                      help="Hidden width of the ground-truth network")
    data.add_argument("--gt_activation", type=str, default="relu",
                      choices=["relu", "erf", "tanh"],
                      help="Activation of the ground-truth network")
    data.add_argument("--seed", type=int, default=42,
                      help="Global RNG seed")

    # ── Experiment grid
    grid = p.add_argument_group("Experiment grid")
    grid.add_argument("--m_values", type=int, nargs="+",
                      default=[100, 200, 400, 800],
                      help="List of hidden widths m to sweep over")
    grid.add_argument("--activations", type=str, nargs="+",
                      default=["relu", "erf", "tanh"],
                      choices=["relu", "erf", "tanh"],
                      help="Activation functions to test")
    grid.add_argument("--regimes", type=str, nargs="+",
                      default=["NTK", "MF", "RF"],
                      choices=["NTK", "MF", "RF"],
                      help="Regimes to include")

    # ── Training
    train = p.add_argument_group("Training")
    train.add_argument("--n_iters", type=int, default=1000,
                       help="Number of gradient descent iterations")
    train.add_argument("--beta", type=float, default=0.1,
                       help="LR scale: MF uses beta*m, NTK uses beta, RF uses beta")
    train.add_argument("--log_every", type=int, default=50,
                       help="Record train/test loss every N iterations")

    # ── Output
    out = p.add_argument_group("Output")
    out.add_argument("--out_dir", type=str, default="results",
                     help="Root directory for logs, plots, and JSON results")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Single-run training
# ──────────────────────────────────────────────────────────────────────────────

def run_one(regime_name, activation, m, beta, n_iters, log_every,
            X_train, y_train, X_test, y_test, seed, logger,
            is_classification=False):
    """
    Train one (regime, activation, m) configuration.
    Returns (train_losses, test_losses, train_accs, test_accs).
    train_accs/test_accs are populated only when is_classification=True.
    """
    RegimeClass = REGIME_CLASSES[regime_name]
    model = RegimeClass(d=X_train.shape[1], m=m, activation=activation, seed=seed)

    lr = beta * m if regime_name == "MF" else beta

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    t0 = time.perf_counter()

    for it in range(1, n_iters + 1):
        model.gradient_descent_step(X_train, y_train, lr)

        if it % log_every == 0 or it == n_iters:
            y_pred_train = model.forward(X_train)
            y_pred_test = model.forward(X_test)
            trl = model.mse_loss(y_train, y_pred_train)
            tel = model.mse_loss(y_test, y_pred_test)
            train_losses.append(trl)
            test_losses.append(tel)

            elapsed = time.perf_counter() - t0
            if is_classification:
                acc_train = float(np.mean((y_pred_train >= 0.5) == y_train))
                acc_test  = float(np.mean((y_pred_test  >= 0.5) == y_test))
                train_accs.append(acc_train)
                test_accs.append(acc_test)
                logger.debug(
                    f"  [{regime_name:3s}|{activation:4s}|m={m:4d}] "
                    f"iter {it:5d}/{n_iters}  "
                    f"train={trl:.6f}  test={tel:.6f}  "
                    f"acc_train={acc_train:.4f}  acc_test={acc_test:.4f}  ({elapsed:.1f}s)"
                )
            else:
                logger.debug(
                    f"  [{regime_name:3s}|{activation:4s}|m={m:4d}] "
                    f"iter {it:5d}/{n_iters}  "
                    f"train={trl:.6f}  test={tel:.6f}  ({elapsed:.1f}s)"
                )

    elapsed = time.perf_counter() - t0
    if is_classification:
        logger.info(
            f"  DONE  [{regime_name:3s}|{activation:4s}|m={m:4d}]  "
            f"train={train_losses[-1]:.6f}  test={test_losses[-1]:.6f}  "
            f"acc_train={train_accs[-1]:.4f}  acc_test={test_accs[-1]:.4f}  "
            f"({elapsed:.1f}s)"
        )
    else:
        logger.info(
            f"  DONE  [{regime_name:3s}|{activation:4s}|m={m:4d}]  "
            f"train={train_losses[-1]:.6f}  test={test_losses[-1]:.6f}  "
            f"({elapsed:.1f}s)"
        )
    return train_losses, test_losses, train_accs, test_accs


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logs_dir = os.path.join(args.out_dir, "logs")
    logger, log_stem = setup_logger(logs_dir, name="experiment1")

    # Plots live in a subdirectory named after the log file
    plots_dir = os.path.join(args.out_dir, "plots", log_stem)
    os.makedirs(plots_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MATH562 — Experiment 1: Regime Comparison")
    logger.info("=" * 60)
    logger.info(f"Output directory: {os.path.abspath(args.out_dir)}")
    logger.info(f"Plots directory : {os.path.abspath(plots_dir)}")

    # ── Data
    if args.n is None:
        args.n = 3000 if args.dataset == "synthetic" else 4000
    if args.n_test is None:
        args.n_test = 500 if args.dataset == "synthetic" else 1000

    is_classification = args.dataset != "synthetic"

    if args.dataset == "synthetic":
        gen = SyntheticDataGenerator(logger)
        X_train, y_train, X_test, y_test = gen.make_dataset(**vars(args))
    elif args.dataset == "mnist":
        gen = MNISTDataGenerator(logger)
        X_train, y_train, X_test, y_test = gen.make_dataset(
            seed=args.seed, classes=tuple(args.classes),
            n=args.n, n_test=args.n_test, normalize=args.normalize,
        )
    elif args.dataset == "cifar10":
        gen = CIFAR10DataGenerator(logger)
        X_train, y_train, X_test, y_test = gen.make_dataset(
            seed=args.seed, classes=tuple(args.classes),
            n=args.n, n_test=args.n_test, normalize=args.normalize,
        )

    # ── Experiment loop (plots emitted as soon as each slice is complete)
    total = len(args.regimes) * len(args.activations) * len(args.m_values)

    logger.info("=" * 60)
    logger.info("EXPERIMENT GRID")
    logger.info("=" * 60)
    logger.info(f"  Dataset    : {args.dataset}"
                + (f"  classes={args.classes}" if is_classification else ""))
    logger.info(f"  Regimes    : {args.regimes}")
    logger.info(f"  Activations: {args.activations}")
    logger.info(f"  Widths m   : {args.m_values}")
    logger.info(f"  Iterations : {args.n_iters}  (log every {args.log_every})")
    logger.info(f"  Beta       : {args.beta}  (MF lr=beta*m, NTK/RF lr=beta)")
    logger.info(f"  Total runs : {total}")
    logger.info("")

    results = {}
    run_idx = 0
    t_start = time.perf_counter()

    for regime in args.regimes:
        for activation in args.activations:
            for m in sorted(args.m_values):
                run_idx += 1
                lr = args.beta * m if regime == "MF" else args.beta
                logger.info(
                    f"[{run_idx}/{total}]  regime={regime}  "
                    f"activation={activation}  m={m}  beta={args.beta}  lr={lr}"
                )
                train_losses, test_losses, train_accs, test_accs = run_one(
                    regime_name=regime,
                    activation=activation,
                    m=m,
                    beta=args.beta,
                    n_iters=args.n_iters,
                    log_every=args.log_every,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    seed=args.seed,
                    logger=logger,
                    is_classification=is_classification,
                )
                entry = {
                    "train_losses": [float(v) for v in train_losses],
                    "test_losses": [float(v) for v in test_losses],
                }
                if is_classification:
                    entry["train_accs"] = [float(v) for v in train_accs]
                    entry["test_accs"] = [float(v) for v in test_accs]
                results[(regime, activation, m)] = entry


    logger.info(f"\nTotal wall-clock time: {time.perf_counter() - t_start:.1f}s")

    # ── Summary table
    print_summary_table(results, logger)

    # ── Save JSON
    json_path = save_results(results, args.out_dir, log_stem=log_stem)
    logger.info(f"Results saved to: {json_path}")

    # ── Summary plots (need full results)
    logger.info("Generating summary plots …")

    plot_training_curves_grid(results, plots_dir, log_interval=args.log_every)
    logger.info("  [1/4] Training curves grid done")

    plot_regime_comparison_grid(results, plots_dir, log_interval=args.log_every)
    logger.info("  [2/4] Regime comparison grid done")

    plot_width_scaling_grid(results, plots_dir)
    logger.info("  [3/4] Width scaling grid done")

    plot_final_loss_heatmap(results, plots_dir)
    logger.info("  [4/4] Heatmap done")

    logger.info(f"\nAll plots saved to: {os.path.abspath(plots_dir)}")
    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()

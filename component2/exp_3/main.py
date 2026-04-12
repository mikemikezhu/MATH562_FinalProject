import os
import sys
import time
import argparse

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from component2.regimes import NTKRegime, MeanFieldRegime, RandomFeaturesRegime
from component2.utils import setup_logger
from component2.data_generator import SyntheticDataGenerator

from component2.exp_3.kernels import compute_kernel
from component2.exp_3.eval import kernel_trajectory
from component2.exp_3.save import save_results, save_kernel_matrices, print_summary_table

from component2.exp_3.plot import (
    plot_kernel_metrics_by_width,
    plot_metric_vs_checkpoint_by_regime,
    plot_rel_change_by_beta_scaling,
)


REGIME_CLASSES = {
    "NTK": NTKRegime,
    "MF": MeanFieldRegime,
    "RF": RandomFeaturesRegime,
}

#TODO: figure out learning rates
'''
DEFAULT_BETA_NTK = 0.01
DEFAULT_BETA_RF = 0.01
DEFAULT_BETA_MF = 0.0001
'''

DEFAULT_BETA_NTK = 0.3
DEFAULT_BETA_RF = 3.5
DEFAULT_BETA_MF = 0.2


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Experiment 3 — Kernel Evolution and Consistency",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data / ground-truth
    data = p.add_argument_group("Data & ground-truth network")
    data.add_argument("--d", type=int, default=10,
                      help="Input dimension")
    data.add_argument("--n", type=int, default=3000,
                      help="Number of training samples")
    data.add_argument("--n_test", type=int, default=500,
                      help="Number of test samples")
    data.add_argument("--n_kernel", type=int, default=50,
                      help="Number of training samples used for kernel computation")
    data.add_argument("--m_star", type=int, default=20,
                      help="Hidden width of the ground-truth network")
    data.add_argument("--gt_activation", type=str, default="tanh",
                      choices=["relu", "erf", "tanh"],
                      help="Activation of the ground-truth network")
    data.add_argument("--seed", type=int, default=42,
                      help="Global RNG seed")

    # ── Experiment grid
    grid = p.add_argument_group("Experiment grid")
    grid.add_argument("--m_values",
                      type=int,
                      nargs="+",
                      default=[100, 200, 400, 800, 1600],
                      help="List of hidden widths m to sweep over")
    grid.add_argument("--activations",
                      type=str,
                      nargs="+",
                      default=["relu", "erf", "tanh"],
                      choices=["relu", "erf", "tanh"],
                      help="Activation functions to test")
    grid.add_argument("--regimes",
                      type=str,
                      nargs="+",
                      default=["NTK", "MF", "RF"],
                      choices=["NTK", "MF", "RF"],
                      help="Regimes to include")
    grid.add_argument("--beta_scalings",
        type=float,
        nargs="+",
        default=[1.0],
        help="Multiplicative scalings applied to the default beta for each regime"
    )

    # ── Training
    train = p.add_argument_group("Training")
    train.add_argument("--n_iters", type=int, default=100,
                       help="Number of gradient descent iterations")
    train.add_argument("--beta_ntk", type=float, default=DEFAULT_BETA_NTK,
                       help="Base learning rate for NTK")
    train.add_argument("--beta_rf", type=float, default=DEFAULT_BETA_RF,
                       help="Base learning rate for RF")
    train.add_argument("--beta_mf", type=float, default=DEFAULT_BETA_MF,
                       help="Base learning rate for MF; actual LR is beta_mf * m")
    train.add_argument("--log_every", type=int, default=10,
                       help="Record train/test loss every N iterations")
    train.add_argument("--kernel_every", type=int, default=25,
                       help="Record the kernel every N iterations")
    train.add_argument("--save_kernel_matrices", action="store_true",
                       help="If set, save the full kernel matrices in an .npz file")

    # ── Output
    out = p.add_argument_group("Output")
    out.add_argument("--out_dir", type=str, default="results",
                     help="Root directory for logs and saved results")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Single-run training
# ──────────────────────────────────────────────────────────────────────────────

def get_learning_rate(regime_name, m, effective_beta):
    if regime_name == "NTK":
        return effective_beta
    elif regime_name == "RF":
        return effective_beta
    elif regime_name == "MF":
        return effective_beta * m
    else:
        raise ValueError(f"Unknown regime: {regime_name}")


def get_default_beta(regime_name, args):
    if regime_name == "NTK":
        return args.beta_ntk
    elif regime_name == "RF":
        return args.beta_rf
    elif regime_name == "MF":
        return args.beta_mf
    else:
        raise ValueError(f"Unknown regime: {regime_name}")


def run_one(regime_name, activation, m,
            default_beta, beta_scaling, effective_beta, lr,
            n_iters, log_every, kernel_every,
            X_train, y_train, X_test, y_test, X_kernel,
            d, n_train, n_test, n_kernel, m_star, gt_activation,
            seed, logger):
    """
    Train one (regime, activation, m) configuration.

    Returns a dict with the structure expected by exp_3/save.py:
        {
            "metadata": {...},
            "metrics": {...},
            "kernels": {...},   # optional to save later
        }
    """
    RegimeClass = REGIME_CLASSES[regime_name]
    model = RegimeClass(d=X_train.shape[1], m=m, activation=activation, seed=seed)

    train_losses = {}
    test_losses = {}
    kernels_by_checkpoint = {}

    t0 = time.perf_counter()

    # Record checkpoint 0 before training starts.
    y_pred_train = model.forward(X_train)
    y_pred_test = model.forward(X_test)
    train_losses[0] = float(model.mse_loss(y_train, y_pred_train))
    test_losses[0] = float(model.mse_loss(y_test, y_pred_test))
    kernels_by_checkpoint[0] = compute_kernel(model, X_kernel, regime_name)

    for it in range(1, n_iters + 1):
        model.gradient_descent_step(X_train, y_train, lr)

        if it % log_every == 0 or it == n_iters:
            y_pred_train = model.forward(X_train)
            y_pred_test = model.forward(X_test)
            trl = float(model.mse_loss(y_train, y_pred_train))
            tel = float(model.mse_loss(y_test, y_pred_test))
            train_losses[it] = trl
            test_losses[it] = tel

            elapsed = time.perf_counter() - t0
            logger.debug(
                f"  [{regime_name:3s}|{activation:4s}|m={m:4d}] "
                f"iter {it:5d}/{n_iters}  "
                f"train={trl:.6f}  test={tel:.6f}  ({elapsed:.1f}s)"
            )

        if it % kernel_every == 0 or it == n_iters:
            kernels_by_checkpoint[it] = compute_kernel(model, X_kernel, regime_name)

    metrics = kernel_trajectory(kernels_by_checkpoint)
    elapsed = time.perf_counter() - t0

    logger.info(
        f"  DONE  [{regime_name:3s}|{activation:4s}|m={m:4d}]  "
        f"train={train_losses[max(train_losses.keys())]:.6f}  "
        f"test={test_losses[max(test_losses.keys())]:.6f}  "
        f"snapshots={len(kernels_by_checkpoint)}  ({elapsed:.1f}s)"
    )

    return {
        "metadata": {
            "regime": regime_name,
            "activation": activation,
            "m": int(m),
            "d": int(d),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "n_kernel": int(n_kernel),
            "m_star": int(m_star),
            "gt_activation": gt_activation,
            "seed": int(seed),
            "default_beta": float(default_beta),
            "beta_scaling": float(beta_scaling),
            "effective_beta": float(effective_beta),
            "lr": float(lr),
            "n_iters": int(n_iters),
            "log_every": int(log_every),
            "kernel_every": int(kernel_every),
            "train_losses": train_losses,
            "test_losses": test_losses,
            "final_train_loss": train_losses[max(train_losses.keys())],
            "final_test_loss": test_losses[max(test_losses.keys())],
        },
        "metrics": metrics,
        "kernels": kernels_by_checkpoint,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logs_dir = os.path.join(args.out_dir, "logs")
    logger, log_stem = setup_logger(logs_dir, name="experiment3")

    plots_dir = os.path.join(args.out_dir, "plots", log_stem)
    os.makedirs(plots_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MATH562 — Experiment 3: Kernel Evolution and Consistency")
    logger.info("=" * 60)
    logger.info(f"Output directory: {os.path.abspath(args.out_dir)}")
    logger.info(f"Plots directory: {os.path.abspath(plots_dir)}")

    synthetic_data_generator = SyntheticDataGenerator(logger)
    X_train, y_train, X_test, y_test = synthetic_data_generator.make_dataset(
        d=args.d,
        n=args.n,
        n_test=args.n_test,
        m_star=args.m_star,
        gt_activation=args.gt_activation,
        seed=args.seed,
    )

    # Generate random subset of X_train for kernel calculation
    if args.n_kernel > len(X_train):
        raise ValueError(
            f"n_kernel={args.n_kernel} cannot exceed number of training samples={len(X_train)}"
        )

    rng = np.random.default_rng(args.seed)
    kernel_indices = rng.choice(len(X_train), size=args.n_kernel, replace=False)
    X_kernel = X_train[kernel_indices]

    total = (
            len(args.regimes)
            * len(args.activations)
            * len(args.m_values)
            * len(args.beta_scalings)
    )

    logger.info("=" * 60)
    logger.info("EXPERIMENT GRID")
    logger.info("=" * 60)
    logger.info(f"  Regimes      : {args.regimes}")
    logger.info(f"  Activations  : {args.activations}")
    logger.info(f"  Widths m     : {args.m_values}")
    logger.info(f"  Iterations   : {args.n_iters}")
    logger.info(f"  Log every    : {args.log_every}")
    logger.info(f"  Kernel every : {args.kernel_every}")
    logger.info(f"  Kernel subset size : {args.n_kernel}")
    logger.info(
        f"  Betas       : NTK={args.beta_ntk}, RF={args.beta_rf}, MF={args.beta_mf}")
    logger.info("  LR rule     : NTK -> beta_ntk, RF -> beta_rf, MF -> beta_mf * m")
    logger.info(f"  Beta scalings : {args.beta_scalings}")
    logger.info(f"  Total runs   : {total}")
    logger.info("")

    results = {}
    run_idx = 0
    t_start = time.perf_counter()

    for regime in args.regimes:
        default_beta = get_default_beta(regime, args)

        for activation in args.activations:
            for beta_scaling in args.beta_scalings:
                effective_beta = default_beta * beta_scaling

                for m in sorted(args.m_values):
                    lr = get_learning_rate(regime, m, effective_beta)

                    run_idx += 1
                    logger.info(
                        f"[{run_idx}/{total}]  regime={regime}  "
                        f"activation={activation}  m={m}  "
                        f"default_beta={default_beta}  "
                        f"scale={beta_scaling}  "
                        f"effective_beta={effective_beta}  lr={lr}"
                    )

                    run_result = run_one(
                        regime_name=regime,
                        activation=activation,
                        m=m,
                        default_beta=default_beta,
                        beta_scaling=beta_scaling,
                        effective_beta=effective_beta,
                        lr=lr,
                        n_iters=args.n_iters,
                        log_every=args.log_every,
                        kernel_every=args.kernel_every,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        X_kernel=X_kernel,
                        d=args.d,
                        n_train=args.n,
                        n_test=args.n_test,
                        n_kernel=args.n_kernel,
                        m_star=args.m_star,
                        gt_activation=args.gt_activation,
                        seed=args.seed,
                        logger=logger,
                    )

                    results[(regime, activation, m, beta_scaling)] = run_result

    # Plotting

    # 1) Width comparison (fixed regime, activation, beta scaling)
    #    → For each (regime, activation, beta_scaling), produce one 2x2 grid
    #      showing kernel metrics vs checkpoint, with one curve per width m.
    for regime in args.regimes:
        for activation in args.activations:
            for beta_scaling in args.beta_scalings:
                plot_kernel_metrics_by_width(
                    results,
                    plots_dir,
                    regime=regime,
                    activation=activation,
                    beta_scaling=beta_scaling,
                )

    # 2) Regime comparison (fixed activation, width, beta scaling)
    #    → For each (activation, m, beta_scaling), compare regimes (NTK/RF/MF)
    #      using relative change vs checkpoint.
    #    NOTE: this assumes plot_metric_vs_checkpoint_by_regime
    #          has been updated to take beta_scaling as input.
    for activation in args.activations:
        for m in args.m_values:
            for beta_scaling in args.beta_scalings:
                plot_metric_vs_checkpoint_by_regime(
                    results,
                    plots_dir,
                    activation=activation,
                    m=m,
                    beta_scaling=beta_scaling,
                    metric_name="rel_change",
                )

    # 3) Learning-rate (beta scaling) comparison (fixed regime, activation, width)
    #    → For each (regime, activation, m), compare different beta scalings
    #      by plotting relative change vs checkpoint, one curve per scaling.
    for regime in args.regimes:
        for activation in args.activations:
            for m in args.m_values:
                plot_rel_change_by_beta_scaling(
                    results,
                    plots_dir,
                    regime=regime,
                    activation=activation,
                    m=m,
                )

    logger.info(f"\nTotal wall-clock time: {time.perf_counter() - t_start:.1f}s")

    print_summary_table(results, logger)

    json_path = save_results(results, args.out_dir, log_stem=log_stem)
    logger.info(f"Results saved to: {json_path}")

    if args.save_kernel_matrices:
        kernels_path = save_kernel_matrices(results, args.out_dir, log_stem=log_stem)
        logger.info(f"Kernel matrices saved to: {kernels_path}")

    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
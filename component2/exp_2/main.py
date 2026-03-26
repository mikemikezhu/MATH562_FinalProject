import argparse
import os
import time

from component2.regimes import NTKRegime, MeanFieldRegime, RandomFeaturesRegime
from component2.utils import (
    setup_logger,
)
from component2.exp_2.plot import plot_training_curves
from component2.exp_2.save import (
    save_results,
    save_variance_results,
    print_summary_table,
    print_variance_table
)

from component2.exp_2.eval import compute_mean_variance
from component2.data_generator import SyntheticDataGenerator

# ──────────────────────────────────────────────────────────────────────────────
# Regime registry
# ──────────────────────────────────────────────────────────────────────────────

REGIME_CLASSES = {
    "NTK": NTKRegime,
    "MF": MeanFieldRegime,
    "RF": RandomFeaturesRegime,
}

# Default per-regime learning rates (motivated by Chapter 12 theory:
#   NTK ~ O(1), MF ~ O(1/m), RF ~ O(1))
DEFAULT_LR = {
    "NTK": 0.1,
    "MF": 0.5,
    "RF": 0.5,
}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Experiment 2 — Learning Rate Scaling Analysis",
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
    data.add_argument("--m_star", type=int, default=200,
                      help="Hidden width of the ground-truth network")
    data.add_argument("--gt_activation", type=str, default="tanh",
                      choices=["relu", "erf", "tanh"],
                      help="Activation of the ground-truth network")
    data.add_argument("--activation", type=str, default="tanh",
                      choices=["relu", "erf", "tanh"],
                      help="Activation for the trained networks")
    data.add_argument("--seed", type=int, default=42,
                      help="Global RNG seed")
    data.add_argument("--beta", type=float, default=1e-3,
                      help="Base learning rate")

    # ── Experiment grid
    grid = p.add_argument_group("Experiment grid")
    grid.add_argument("--m_values", type=int, nargs="+",
                      default=[100, 200, 400, 800, 1600, 3200],
                      help="List of hidden widths m to sweep over")
    grid.add_argument("--regimes", type=str, nargs="+",
                      default=["NTK", "MF", "RF"],
                      choices=["NTK", "MF", "RF"],
                      help="Regimes to include")
    grid.add_argument("--alphas", type=float, nargs="+",
                      default=[-0.5, 0.5, 0.0, 1.0, 2.0],
                      help="Scaling exponents to test: eta = prefactor * beta * m^alpha. ")
    grid.add_argument("--prefactors", type=float, nargs="+",
                      default=[0.01, 0.1, 0.5, 1.0],
                      help="Prefactors to test: eta = prefactor * beta * m^alpha. ")

    # ── Training
    train = p.add_argument_group("Training")
    train.add_argument("--n_iters", type=int, default=1000,
                       help="Number of gradient descent iterations")
    train.add_argument("--lr_ntk", type=float, default=DEFAULT_LR["NTK"],
                       help="Learning rate for the NTK regime")
    train.add_argument("--lr_mf", type=float, default=DEFAULT_LR["MF"],
                       help="Learning rate for the MF regime")
    train.add_argument("--lr_rf", type=float, default=DEFAULT_LR["RF"],
                       help="Learning rate for the RF regime")
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
            y_pred_test = model.forward(X_test)
            trl = model.mse_loss(y_train, y_pred_train)
            tel = model.mse_loss(y_test, y_pred_test)
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
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logs_dir = os.path.join(args.out_dir, "logs")
    logger, log_stem = setup_logger(logs_dir, name="experiment2")

    # Plots live in a subdirectory named after the log file
    plots_dir = os.path.join(args.out_dir, "plots", log_stem)
    os.makedirs(plots_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MATH562 — Experiment 2: Learning Rate Scaling Analysis")
    logger.info("=" * 60)
    logger.info(f"Output directory: {os.path.abspath(args.out_dir)}")
    logger.info(f"Plots directory : {os.path.abspath(plots_dir)}")

    # ── Data
    synthetic_data_generator = SyntheticDataGenerator(logger)
    X_train, y_train, X_test, y_test = synthetic_data_generator.make_dataset(**vars(args))

    # ── Experiment loop (plots emitted as soon as each slice is complete)
    lr_map = {"NTK": args.lr_ntk, "MF": args.lr_mf, "RF": args.lr_rf}
    total = len(args.regimes) * len(args.m_values) * len(args.alphas) * len(args.prefactors)

    logger.info("=" * 60)
    logger.info("EXPERIMENT GRID")
    logger.info("=" * 60)
    logger.info(f"  Regimes    : {args.regimes}")
    logger.info(f"  Widths m   : {args.m_values}")
    logger.info(f"  Beta : {args.beta}")
    logger.info(f"  Alphas : {args.alphas}")
    logger.info(f"  Prefactors : {args.prefactors}")
    logger.info(f"  Iterations : {args.n_iters}  (log every {args.log_every})")
    logger.info(f"  LR — NTK={args.lr_ntk}, MF={args.lr_mf}, RF={args.lr_rf}")
    logger.info(f"  Total runs : {total}")
    logger.info("")

    results = {}
    run_idx = 0
    t_start = time.perf_counter()

    for regime in args.regimes:
        lr = lr_map[regime]
        for alpha in args.alphas:
            for prefactor in args.prefactors:
                for m in sorted(args.m_values):
                    run_idx += 1
                    logger.info(
                        f"[{run_idx}/{total}]  regime={regime}  "
                        f"activation={args.activation}  m={m}  lr={lr}  "
                        f"beta={args.beta}  alpha={alpha}  prefactor={prefactor}]"
                    )

                    # Test scaling: eta = prefactor * beta * m^alpha
                    # e.g. eta = beta * m^2 / 100
                    # Then, prefactor = 0.01, alpha = 2.0
                    lr = prefactor * args.beta * (m ** alpha)
                    train_losses, test_losses = run_one(
                        regime_name=regime,
                        activation=args.activation,
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
                    results[(regime, alpha, prefactor, m)] = {
                        "train_losses": [float(v) for v in train_losses],
                        "test_losses": [float(v) for v in test_losses],
                    }

            # ── After every (regime, alpha, prefactor) pair: training curves
            pair_results = {k: v for k, v in results.items() if k[0] == regime and k[1] == alpha and k[2] == prefactor}
            plot_training_curves(args, pair_results, plots_dir, log_interval=args.log_every)
            logger.info(f"  Saved curves_{regime}_beta_{args.beta}_alpha_{alpha}_prefactor_{prefactor}.png")

    logger.info(f"\nTotal wall-clock time: {time.perf_counter() - t_start:.1f}s")

    # Calculate mean variance
    variance_results = compute_mean_variance(results)
    save_variance_results(variance_results, args.out_dir, log_stem=log_stem)
    print_variance_table(variance_results, logger)

    # ── Summary table
    print_summary_table(results, logger)

    # ── Save JSON
    json_path = save_results(results, args.out_dir, log_stem=log_stem)
    logger.info(f"Results saved to: {json_path}")

    logger.info(f"\nAll plots saved to: {os.path.abspath(plots_dir)}")
    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()

import numpy as np


def compute_mean_variance(results: dict):
    regimes = sorted({k[0] for k in results})
    alphas = sorted({k[1] for k in results})
    prefactors = sorted({k[2] for k in results})

    var_results = {}
    for regime in regimes:
        for alpha in alphas:
            for prefactor in prefactors:
                keys_present = [k for k in results if k[0] == regime and k[1] == alpha and k[2] == prefactor]
                if not keys_present:
                    continue

                train_curves = []
                test_curves = []
                widths_present = []

                for key in sorted(keys_present, key=lambda x: x[3]):
                    _, _, _, m = key
                    r = results[key]

                    train_curves.append(r["train_losses"])
                    test_curves.append(r["test_losses"])
                    widths_present.append(m)

                train_loss_array = np.array(train_curves, dtype=float)
                test_loss_array = np.array(test_curves, dtype=float)

                # Calculate variance across widths for each iteration
                train_variance = np.var(train_loss_array, axis=0)
                test_variance = np.var(test_loss_array, axis=0)

                # Calculate mean variance across iterations
                mean_train_variance = np.mean(train_variance)
                mean_test_variance = np.mean(test_variance)

                var_results[(regime, alpha, prefactor)] = {
                    "mean_train_variance": mean_train_variance,
                    "mean_test_variance": mean_test_variance,
                }

    return var_results

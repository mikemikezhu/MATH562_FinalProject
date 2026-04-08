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

                log_train_loss_array = np.log(train_loss_array + 1e-8)
                log_test_loss_array = np.log(test_loss_array + 1e-8)

                # Calculate variance across widths for each iteration
                train_variance = np.var(train_loss_array, axis=0)
                test_variance = np.var(test_loss_array, axis=0)

                log_train_variance = np.var(log_train_loss_array, axis=0)
                log_test_variance = np.var(log_test_loss_array, axis=0)

                # Calculate mean variance across iterations
                mean_train_variance = np.mean(train_variance)
                mean_test_variance = np.mean(test_variance)

                mean_log_train_variance = np.mean(log_train_variance)
                mean_log_test_variance = np.mean(log_test_variance)

                var_results[(regime, alpha, prefactor)] = {
                    "mean_train_variance": mean_train_variance,
                    "mean_test_variance": mean_test_variance,
                    "mean_log_train_variance": mean_log_train_variance,
                    "mean_log_test_variance": mean_log_test_variance,
                }

    return var_results


def compute_learn_score(results: dict):
    regimes = sorted({k[0] for k in results})
    alphas = sorted({k[1] for k in results})
    prefactors = sorted({k[2] for k in results})

    learn_score_results = {}
    for regime in regimes:
        for alpha in alphas:
            for prefactor in prefactors:
                keys_present = [k for k in results if k[0] == regime and k[1] == alpha and k[2] == prefactor]
                if not keys_present:
                    continue

                train_learn_score_across_widths = []
                test_learn_score_across_widths = []

                widths_present = []

                for key in sorted(keys_present, key=lambda x: x[3]):
                    _, _, _, m = key
                    r = results[key]

                    train_losses = r["train_losses"]
                    test_losses = r["test_losses"]

                    train_learn_score = train_losses[0] - train_losses[-1]
                    test_learn_score = test_losses[0] - test_losses[-1]

                    train_learn_score_across_widths.append(train_learn_score)
                    test_learn_score_across_widths.append(test_learn_score)

                    widths_present.append(m)

                train_learn_score_array = np.array(train_learn_score_across_widths, dtype=float)
                test_learn_score_array = np.array(test_learn_score_across_widths, dtype=float)

                log_train_learn_score_array = np.log(train_learn_score_array + 1e-8)
                log_test_learn_score_array = np.log(test_learn_score_array + 1e-8)

                # Calculate mean learn score across widths
                mean_train_learn_score = np.mean(train_learn_score_array)
                mean_test_learn_score = np.mean(test_learn_score_array)

                mean_log_train_learn_score = np.mean(log_train_learn_score_array)
                mean_log_test_learn_score = np.mean(log_test_learn_score_array)

                # Calculate worst learn score across widths
                worst_train_learn_score = np.min(train_learn_score_array)
                worst_test_learn_score = np.min(test_learn_score_array)

                worst_log_train_learn_score = np.min(log_train_learn_score_array)
                worst_log_test_learn_score = np.min(log_test_learn_score_array)

                learn_score_results[(regime, alpha, prefactor)] = {
                    "mean_train_learn_score": mean_train_learn_score,
                    "mean_test_learn_score": mean_test_learn_score,
                    "mean_log_train_learn_score": mean_log_train_learn_score,
                    "mean_log_test_learn_score": mean_log_test_learn_score,
                    "worst_train_learn_score": worst_train_learn_score,
                    "worst_test_learn_score": worst_test_learn_score,
                    "worst_log_train_learn_score": worst_log_train_learn_score,
                    "worst_log_test_learn_score": worst_log_test_learn_score,
                }

    return learn_score_results

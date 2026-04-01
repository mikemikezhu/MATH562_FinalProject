import numpy


def frobenius_norm(K: numpy.ndarray):
    return numpy.linalg.norm(K)


def relative_change(K_old: numpy.ndarray, K_new: numpy.ndarray):
    return numpy.linalg.norm(K_new - K_old) / numpy.linalg.norm(K_old)


def absolute_change(K_old: numpy.ndarray, K_new: numpy.ndarray):
    return numpy.linalg.norm(K_new - K_old)


def mean_kernel_value(K: numpy.ndarray):
    return 1.0 / K.size * numpy.sum(K)


def kernel_trajectory(kernels_by_checkpoint: dict):
    """
    Given a dict:
        {t: K_t}  where K_t is an (n x n) numpy array,

    returns:
        {t: {
            "fro_norm": ...,
            "abs_change": ...,
            "rel_change": ...,
            "mean_value": ...
        }}
    """

    # Sort checkpoints (important for consistency)
    checkpoints = sorted(kernels_by_checkpoint.keys())

    # Initial kernel (baseline)
    t0 = checkpoints[0]
    K0 = kernels_by_checkpoint[t0]

    results = {}

    for t in checkpoints:
        Kt = kernels_by_checkpoint[t]

        fro_norm = frobenius_norm(Kt)
        mean_val = mean_kernel_value(Kt)

        # For t = 0, changes should be 0
        if t == t0:
            abs_change = 0.0
            rel_change = 0.0
        else:
            abs_change = absolute_change(K0, Kt)

            # Avoid division by zero just in case
            denom = frobenius_norm(K0)
            rel_change = abs_change / denom if denom != 0 else 0.0

        results[t] = {
            "fro_norm": float(fro_norm),
            "abs_change": float(abs_change),
            "rel_change": float(rel_change),
            "mean_value": float(mean_val),
        }

    return results

import numpy
from component2.regimes import *


def compute_rf_kernel(model: RandomFeaturesRegime, X: numpy.ndarray):
    """
        Compute the Random Features kernel matrix on a set of inputs X.
        Parameters
        ----------
        model : RandomFeaturesRegime
            Must have attributes:
            - W of shape (m, d)
            - m (number of hidden neurons)
            - sigma(z) activation function
        X : np.ndarray
            Input matrix of shape (n_samples, d)

        Returns
        -------
        K : np.ndarray
            RF kernel matrix of shape (n_samples, n_samples)
        """
    Z = X @ model.W.T
    A = model.sigma(Z)
    K = (1.0 / model.m) * (A @ A.T)
    return K


def compute_ntk_kernel(model: NTKRegime, X: numpy.ndarray):
    n, m = X.shape[0], model.m

    # Forward pass
    pre_activation = X @ model.W.T
    hidden = model.sigma(pre_activation)
    hidden_prime = model.sigma_prime(pre_activation)

    # K_a contribution
    K_a = (1.0 / m) * (hidden @ hidden.T)

    # K_w contribution
    X_inner = X @ X.T
    a_sq = model.a ** 2
    weighted_hprime = hidden_prime * a_sq[None, :]
    K_w = (1.0 / m) * (weighted_hprime @ hidden_prime.T) * X_inner

    return K_a + K_w


def compute_mf_kernel(model, X):
    n, m = X.shape[0], model.m

    # Forward pass
    pre_activation = X @ model.W.T
    hidden = model.sigma(pre_activation)
    hidden_prime = model.sigma_prime(pre_activation)

    # K_a contribution
    K_a = (1.0 / (m ** 2)) * (hidden @ hidden.T)

    # K_w contribution
    X_inner = X @ X.T
    a_sq = model.a ** 2
    weighted_hprime = hidden_prime * a_sq[None, :]
    K_w = (1.0 / (m ** 2)) * (weighted_hprime @ hidden_prime.T) * X_inner

    return K_a + K_w


def compute_kernel(model, X: numpy.ndarray, regime: str):
    if regime == 'RF':
        return compute_rf_kernel(model, X)
    elif regime == 'NTK':
        return compute_ntk_kernel(model, X)
    elif regime == 'MF':
        return compute_mf_kernel(model, X)
    else:
        raise ValueError(f"Unknown regime: {regime}")

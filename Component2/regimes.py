import numpy as np

try:
    from scipy.special import erf as _scipy_erf
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class BaseRegime:
    """
    Two-layer neural network: f(X) = (1/alpha) * sigma(X W^T) a

    Parameters
    ----------
    d : int       Input dimension
    m : int       Hidden width
    activation :  'relu' | 'erf' | 'tanh'
    seed : int    RNG seed for reproducibility
    """

    def __init__(self, d, m, activation="relu", seed=42):
        rng = np.random.default_rng(seed)
        self.d = d
        self.m = m
        self.activation_name = activation
        self.alpha = 1.0  # overridden by subclasses

        # W: (m, d),  each row ~ N(0, I_d)
        self.W = rng.multivariate_normal(np.zeros(d), np.eye(d), size=m)
        # a: (m,),   ~ N(0, 1)
        self.a = rng.standard_normal(m)

        # Cached from last forward pass (needed by backward)
        self._X = None
        self._Z = None
        self._A = None

    # ------------------------------------------------------------------
    # Activation and its derivative
    # ------------------------------------------------------------------

    def sigma(self, z):
        if self.activation_name == "relu":
            return np.maximum(0.0, z)
        elif self.activation_name == "tanh":
            return np.tanh(z)
        elif self.activation_name == "erf":
            if not _HAS_SCIPY:
                raise ImportError("scipy is required for the 'erf' activation. "
                                  "Install with: pip install scipy")
            return _scipy_erf(z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name!r}. "
                             f"Choose from 'relu', 'erf', 'tanh'.")

    def sigma_prime(self, z):
        if self.activation_name == "relu":
            return (z > 0).astype(float)
        elif self.activation_name == "tanh":
            return 1.0 - np.tanh(z) ** 2
        elif self.activation_name == "erf":
            # d/dz erf(z) = (2/sqrt(pi)) * exp(-z^2)
            return (2.0 / np.sqrt(np.pi)) * np.exp(-(z ** 2))
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name!r}.")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X):
        """
        Compute f(X) = (1/alpha) * sigma(X W^T) a.

        Parameters
        ----------
        X : (n, d)

        Returns
        -------
        y_pred : (n,)
        """
        self._X = X
        self._Z = X @ self.W.T          # (n, m)
        self._A = self.sigma(self._Z)   # (n, m)
        return (1.0 / self.alpha) * (self._A @ self.a)  # (n,)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, y_true, y_pred):
        """
        Compute gradients of MSE loss L = (1/n)||y_pred - y_true||^2
        w.r.t. W and a. Requires a preceding call to forward().

        Returns
        -------
        dL_dW : (m, d)
        dL_da : (m,)
        """
        n = len(y_true)
        # dL/dy_pred_i = 2(y_pred_i - y_true_i)/n  — shape (n,)
        dL_dy = 2.0 * (y_pred - y_true) / n

        # dL/da = (1/alpha) A^T dL_dy,  shape (m,)
        dL_da = (1.0 / self.alpha) * (self._A.T @ dL_dy)

        # dL/dA_ij = dL_dy_i * (1/alpha) * a_j  — shape (n, m)
        dL_dA = dL_dy[:, None] * (self.a[None, :] / self.alpha)

        # dL/dZ = dL/dA * sigma'(Z),  shape (n, m)
        dL_dZ = dL_dA * self.sigma_prime(self._Z)

        # dL/dW = dL/dZ^T @ X,  shape (m, d)
        dL_dW = dL_dZ.T @ self._X

        return dL_dW, dL_da

    # ------------------------------------------------------------------
    # Gradient descent step
    # ------------------------------------------------------------------

    def gradient_descent_step(self, X, y_true, lr):
        """One step of full gradient descent (updates both W and a)."""
        y_pred = self.forward(X)
        dL_dW, dL_da = self.backward(y_true, y_pred)
        self.W -= lr * dL_dW
        self.a -= lr * dL_da

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def mse_loss(self, y_true, y_pred):
        return float(np.mean((y_true - y_pred) ** 2))

    def train(self, X, y, learning_rate, n_iterations):
        """Convenience training loop. Returns list of per-iteration MSE losses."""
        losses = []
        for _ in range(n_iterations):
            self.gradient_descent_step(X, y, learning_rate)
            y_pred = self.forward(X)
            losses.append(self.mse_loss(y, y_pred))
        return losses


# --------------------------------------------------------------------------
# Regime subclasses — differ only in alpha and which parameters are trained
# --------------------------------------------------------------------------

class NTKRegime(BaseRegime):
    """
    Neural Tangent Kernel (NTK) regime.
    alpha = sqrt(m); both W and a are trained.
    The network stays close to its initialisation → linearised dynamics.
    """

    def __init__(self, d, m, activation="relu", seed=42):
        super().__init__(d, m, activation, seed)
        self.alpha = float(np.sqrt(m))


class MeanFieldRegime(BaseRegime):
    """
    Mean-Field (MF) regime.
    alpha = m; both W and a are trained.
    Neurons evolve independently — captures feature learning.
    """

    def __init__(self, d, m, activation="relu", seed=42):
        super().__init__(d, m, activation, seed)
        self.alpha = float(m)


class RandomFeaturesRegime(BaseRegime):
    """
    Random Features (RF) regime.
    alpha = sqrt(m); ONLY a is trained (W is frozen at initialisation).
    Reduces to a linear model with random features phi_j(x) = sigma(w_j^T x).
    """

    def __init__(self, d, m, activation="relu", seed=42):
        super().__init__(d, m, activation, seed)
        self.alpha = float(np.sqrt(m))

    def gradient_descent_step(self, X, y_true, lr):
        """Update only the output layer a; W stays frozen."""
        y_pred = self.forward(X)
        _, dL_da = self.backward(y_true, y_pred)
        self.a -= lr * dL_da

import numpy as np

class BaseRegime:
    def __init__(self, d, m, activation="relu", seed=0):
        np.random.seed(seed)

        self.d = d
        self.m = m
        self.activation_name = activation

        self.W = np.random.randn(m, d)
        self.a = np.random.randn(m)

        self.alpha = 1

    def sigma(self, z):
        if self.activation_name == "relu":
            return np.maximum(0, z)
        elif self.activation_name == "tanh":
            return np.tanh(z)
        elif self.activation_name == "sigmoid":
            return 1 / (1 + np.exp(-z))
        else:
            raise ValueError("Unsupported activation")

    def sigma_prime(self, z):
        if self.activation_name == "relu":
            return (z > 0).astype(float)
        elif self.activation_name == "tanh":
            return 1 - np.tanh(z) ** 2
        elif self.activation_name == "sigmoid":
            s = 1 / (1 + np.exp(-z))
            return s * (1 - s)
        else:
            raise ValueError("Unsupported activation")

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W.T # (n, m)
        self.A = self.sigma(self.Z) # (n, m)
        return (1 / self.alpha) * self.A @ self.a # (n,)

    def backward(self, y_true, y_pred):
        n = y_true.shape[0]

        dL_dy = 2 * (y_pred - y_true) / n # (n,)

        # dL/da
        dy_da = (1 / self.alpha) * self.A # (n, m)
        dL_da = dL_dy @ dy_da # (m,)

        # dL/dW
        dy_dA = (1 / self.alpha) * np.tile(self.a, (n, 1))  # (n, m)
        dA_dZ = self.sigma_prime(self.Z) # (n, m)
        dZ_dW = self.X # (n, d)

        # need (m, d) for dL/dW
        dL_dW = dL_dy @ (dy_dA * dA_dZ) @ dZ_dW # (m, d)
        
        dL_dZ = dy_dA * dA_dZ # (n, m)
        dL_dW = dL_dZ.T @ self.X # (m, d)

        pass

    def gradient_descent_step(self, X, y_true, learning_rate):
        y_pred = self.forward(X)
        dW1, db1, dW2, db2 = self.backward(y_true, y_pred)
        pass

    def train(self, X, y, learning_rate, n_iterations):
        losses = []
        for _ in range(n_iterations):
            y_pred = self.gradient_descent_step(X, y, learning_rate)
            losses.append(self.mse_loss(y, y_pred))
        return losses

    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
import numpy as np

class BaseRegime:
    def __init__(self, d, m, activation="relu", seed=0):
        np.random.seed(seed)

        self.d = d
        self.m = m
        self.activation_name = activation

        self.W = np.random.randn(d, m)
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
        self.Z = X @ self.W # (n, m)
        self.A = self.sigma(self.Z) # (n, m)
        return (1 / self.alpha) * self.A @ self.a # (n,)

    def backward(self, y_true, y_pred):

        dL_dy = 2 * np.mean(y_pred - y_true) # (n,)

        # dL/da — a is (m,), dy/da = (1/alpha) * A^T @ dL_dy
        dy_da = (1 / self.alpha) * self.A          # (n, m)
        dL_da = dy_da.T @ dL_dy                    # (m,) 

        # dL/dA — elementwise: dL/dA_ij = dL_dy_i * (1/alpha) * a_j
        dy_dA = (1 / self.alpha) * self.a          # (m,)
        dL_dA = dL_dy[:, None] * dy_dA[None, :]   # (n, m) 

        # dL/dZ — elementwise multiply with sigma'
        dA_dZ = self.sigma_prime(self.Z)           # (n, m)
        dL_dZ = dL_dA * dA_dZ                     # (n, m)

        # dL/dW — X^T @ dL_dZ
        dZ_dW = self.X
        dL_dW = dZ_dW.T @ dL_dZ                   # (d, m)

        # Store gradients
        self.grad_W = dL_dW
        self.grad_a = dL_da

        return dL_dW, dL_da

    def gradient_descent_step(self, X, y_true, learning_rate):
        y_pred = self.forward(X)
        dL_dW, dL_da = self.backward(y_true, y_pred)

        self.W -= learning_rate * dL_dW
        self.a -= learning_rate * dL_da

    def train(self, X, y, learning_rate, n_iterations):
        losses = []
        for _ in range(n_iterations):
            y_pred = self.gradient_descent_step(X, y, learning_rate)
            losses.append(self.mse_loss(y, y_pred))
        return losses

    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
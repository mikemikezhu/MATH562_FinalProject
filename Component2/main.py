import numpy as np
from regimes import BaseRegime

d = 10
m_star = 10
m = 5
n = 1000
test_size = 0.2
n_test = int(n * test_size)
seed = 42

ground_truth_regime = BaseRegime(d=d, m=m, activation="relu", seed=seed)

# Sample xj ~ N(0, Id/d): covariance is (1/d)*I
X_train = np.random.multivariate_normal(np.zeros(d), np.eye(d) / d, size=n)  # (n, d)
y_train = ground_truth_regime.forward(X_train)  # (n,)

X_test = np.random.multivariate_normal(np.zeros(d), np.eye(d) / d, size=n_test)  # (n_test, d)
y_test = ground_truth_regime.forward(X_test)  # (n_test,)


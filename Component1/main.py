"""
Imports 
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1) # Set seed

def generate_data(n, d, cov_a, sigma2, beta_n):
    """
    :param cov_a: dxd covariance matrix of a
    :sigma_2: variance parameter for epsilon
    :beta_n: scaling for epsilon variance
    """

    A = np.random.multivariate_normal(np.zeros(d), cov_a, size=n) # Design matrix
    theta_star = np.random.multivariate_normal(np.zeros(d), np.eye(d)) # True param
    epsilon = np.random.multivariate_normal(np.zeros(n), sigma2 * beta_n * np.eye(n)) # Noise

    b = A @ theta_star + epsilon

    return A, b, theta_star


def compute_empirical_risk(A, b, theta):
    n = A.shape[0]
    return 1/(2*n) * np.linalg.norm(A@theta - b)**2

def test_scaling_empirical(d_list, rho, sigma2):
    """
    Test if the empirical risk remains stable when d varies
    """
    for d in d_list:
        theta = np.random.randn(d)
        n = int(d/rho)
        alpha_d = 1/d
        cov_a = alpha_d * np.eye(d)
        beta_n = 1
        A, b, theta_star = generate_data(n, d, cov_a, sigma2, beta_n)
        print("Empirical risk, d = " + str(d) + ": " + str(compute_empirical_risk(A, b, theta)))

def compute_true_risk(theta_star, theta, cov_a, sigma2, beta_n):
    return 1/2 * (theta_star - theta).T @ cov_a @ (theta_star - theta) + 1/2 * sigma2 * beta_n

def plot_risk(d_list, rho, num_runs, num_epochs, sigma2, sgd_algo, gamma, sigma_hat, step_type=None):

    """
    Plot the empirical and true risk evolution per epoch
    :gamma: learning rate
    :sgd_algo: The chosen sgd algo (classical, single shuffle, multiple shuffle...)
    :sigma_hat: See projection description. 
    """

    results = {}

    for d in d_list:

        n = int(d / rho)
        alpha_d = 1/d
        cov_a = alpha_d * sigma_hat(d)
        beta_n = 1

        all_runs_empirical_risk = []
        all_runs_true_risk = []

        for run in range(num_runs):
            empirical_risk_hist, true_risk_hist = sgd_algo(
                rho, d, cov_a, sigma2, beta_n, num_epochs, gamma, step_type
            )
            all_runs_empirical_risk.append(empirical_risk_hist)
            all_runs_true_risk.append(true_risk_hist)

        all_runs_empirical_risk = np.array(all_runs_empirical_risk)
        empirical_risk_hist_mean = np.mean(all_runs_empirical_risk, axis=0)
        empirical_risk_hist_std = np.std(all_runs_empirical_risk, axis=0)

        all_runs_true_risk = np.array(all_runs_true_risk)
        true_risk_hist_mean = np.mean(all_runs_true_risk, axis=0)
        true_risk_hist_std = np.std(all_runs_true_risk, axis=0)

        results[d] = {
            "empirical_mean": empirical_risk_hist_mean,
            "empirical_std": empirical_risk_hist_std,
            "true_mean": true_risk_hist_mean,
            "true_std": true_risk_hist_std,
        }
        
    # Plot empirical risk for all d
    plt.figure(figsize=(5, 3))
    for d in d_list:
        mean = results[d]["empirical_mean"]
        std = results[d]["empirical_std"]
        plt.plot(np.arange(num_epochs), mean, label=f"d={d}")
        plt.fill_between(np.arange(num_epochs), mean - std, mean + std, alpha=0.3)

    plt.xlabel("Num of epochs")
    plt.ylabel("Empirical risk")
    plt.title("Training error for different d")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()

    # Plot true risk for all d
    plt.figure(figsize=(5, 3))
    for d in d_list:
        mean = results[d]["true_mean"]
        std = results[d]["true_std"]
        plt.plot(np.arange(num_epochs), mean, label=f"d={d}")
        plt.fill_between(np.arange(num_epochs), mean - std, mean + std, alpha=0.1)

    plt.xlabel("Num of epochs")
    plt.ylabel("True risk")
    plt.title("True risk for different d")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.show()


###############################################################################
# Experiment 1 - Figuring out the scaling gamma in small batch SGD
###############################################################################

def run_sgd(rho, d, cov_a, sigma2, beta_n, num_epochs, gamma, step_type):

    empirical_risk_hist = []
    true_risk_hist = []

    n = int(d/rho)

    A, b, theta_star = generate_data(n, d, cov_a, sigma2, beta_n)

    theta = np.random.randn(d) # Random initialization

    # Set absolute constant L (for the learning rate gamma(d))
    U, S, Vh = np.linalg.svd(A) # To find the singular values of A^T A (and thus its eigenvalues)
    if step_type == "max":
        L = S[0]**2
    
    elif step_type == "avg":
        L = np.mean(S**2)

    else:
        L = 1

    for epoch in range(num_epochs):
        for _ in range(n):
            idx  = np.random.randint(0, n)
            a_idx = A[idx, :]
            b_idx = b[idx]
            theta -= 1/L * gamma(d) * 1/n * (np.dot(a_idx, theta) - b_idx)* a_idx 

        empirical_risk_hist.append(compute_empirical_risk(A, b, theta))
        true_risk_hist.append(compute_true_risk(theta_star, theta, cov_a, sigma2, beta_n))

    return empirical_risk_hist, true_risk_hist


###############################################################################
# Experiment 2 - Effect of randomness in SGD
###############################################################################

def single_suffle_sgd(rho, d, cov_a, sigma2, beta_n, num_epochs, gamma):

    empirical_risk_hist = []
    true_risk_hist = []

    n = int(d/rho)

    A, b, theta_star = generate_data(n, d, cov_a, sigma2, beta_n)

    theta = np.random.randn(d) # Random initialization

    permutation = np.random.permutation(n)

    for epoch in range(num_epochs):
        for i in range(n):
            idx  = permutation[i]
            a_idx = A[idx, :]
            b_idx = b[idx]
            theta -= gamma(d) * 1/n * (np.dot(a_idx, theta) - b_idx)* a_idx

        empirical_risk_hist.append(compute_empirical_risk(A, b, theta))
        true_risk_hist.append(compute_true_risk(theta_star, theta, cov_a, sigma2, beta_n))

    return empirical_risk_hist, true_risk_hist


def multiple_suffle_sgd(rho, d, cov_a, sigma2, beta_n, num_epochs, gamma):

    empirical_risk_hist = []
    true_risk_hist = []

    n = int(d/rho)

    A, b, theta_star = generate_data(n, d, cov_a, sigma2, beta_n)

    theta = np.random.randn(d) # Random initialization

    for epoch in range(num_epochs):
        permutation = np.random.permutation(n)
        for i in range(n):
            idx  = permutation[i]
            a_idx = A[idx, :]
            b_idx = b[idx]
            theta -= gamma(d) * 1/n * (np.dot(a_idx, theta) - b_idx)* a_idx

        empirical_risk_hist.append(compute_empirical_risk(A, b, theta))
        true_risk_hist.append(compute_true_risk(theta_star, theta, cov_a, sigma2, beta_n))

    return empirical_risk_hist, true_risk_hist


###############################################################################
# Experiment 3 - Repeating experiment 1 but with SGD with momentum
###############################################################################

def sgd_momentum(rho, d, cov_a, sigma2, beta_n, num_epochs, gamma, step_type, delta):

    empirical_risk_hist = []
    true_risk_hist = []

    n = int(d/rho)

    A, b, theta_star = generate_data(n, d, cov_a, sigma2, beta_n)

    theta = np.random.randn(d) # Random initialization
    theta_prev = theta.copy()

    # Set absolute constant L (for the learning rate gamma(d))
    U, S, Vh = np.linalg.svd(A) # To find the singular values of A^T A (and thus its eigenvalues)
    if step_type == "max":
        L = S[0]**2
    
    elif step_type == "avg":
        L = np.mean(S**2)

    else:
        L = 1

    for epoch in range(num_epochs):
        for _ in range(n):
            
            idx  = np.random.randint(0, n)
            a_idx = A[idx, :]
            b_idx = b[idx]
            theta_old = theta.copy()
            theta = theta - 1/L * gamma(d) * 1/n * (np.dot(a_idx, theta) - b_idx)* a_idx + delta * (theta - theta_prev)
            theta_prev = theta_old

        empirical_risk_hist.append(compute_empirical_risk(A, b, theta))
        true_risk_hist.append(compute_true_risk(theta_star, theta, cov_a, sigma2, beta_n))

    return empirical_risk_hist, true_risk_hist


def sgd_momentum_fixed_delta(rho, d, cov_a, sigma2, beta_n, num_epochs, gamma, step_type):
    """
    Wrapper function
    """
    return sgd_momentum(rho, d, cov_a, sigma2, beta_n, num_epochs, gamma, step_type, delta=0.5)


"""
Imports 
"""
import numpy as np
import matplotlib.pyplot as plt

def gamma(d):
    return d

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


def run_sgd(rho, d, cov_a, sigma2, beta_n, num_epochs, gamma):

    empirical_risk_hist = []
    true_risk_hist = []

    n = int(d/rho)

    A, b, theta_star = generate_data(n, d, cov_a, sigma2, beta_n)

    theta = np.random.randn(d) # Random initialization

    for epoch in range(num_epochs):
        for _ in range(n):
            idx  = np.random.randint(0, n)
            a_idx = A[idx, :]
            b_idx = b[idx]
            theta -= gamma(d) * 1/n * (np.dot(a_idx, theta) - b_idx)* a_idx 

        empirical_risk_hist.append(compute_empirical_risk(A, b, theta))
        true_risk_hist.append(compute_true_risk(theta_star, theta, cov_a, sigma2, beta_n))

    return empirical_risk_hist, true_risk_hist

def plot_risk(d_list, rho, num_runs, num_epochs, sigma2, gamma, sgd_algo):

    """
    Plot the empirical and true risk evolution per epoch
    :gamma: learning rate
    :sgd_algo: The chosen sgd algo (classical, single shuffle, multiple shuffle...)
    """

    results = {}

    for d in d_list:

        n = int(d / rho)
        alpha_d = 1/d
        cov_a = alpha_d * np.eye(d)
        beta_n = 1

        all_runs_empirical_risk = []
        all_runs_true_risk = []

        for run in range(num_runs):
            empirical_risk_hist, true_risk_hist = sgd_algo(
                rho, d, cov_a, sigma2, beta_n, num_epochs, gamma
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
    plt.figure(figsize=(10, 6))
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

rho = 1
d_list = [100, 200, 400]
num_runs = 10
num_epochs = 10
sigma2 = 0.01

plot_risk(d_list, rho, num_runs, num_epochs, sigma2, gamma, run_sgd)



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


rho = 2
d_list = [100, 200, 400, 800, 1600]
d_list = [100, 200, 400]
num_runs = 10
num_epochs = 10
sigma2 = 1

#plot_risk(d_list, rho, num_runs, num_epochs, sigma2, gamma, single_suffle_sgd)



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


rho = 2
d_list = [100, 200, 400]
num_runs = 10
num_epochs = 10
sigma2 = 1

# plot_risk(d_list, rho, num_runs, num_epochs, sigma2, gamma, multiple_suffle_sgd)
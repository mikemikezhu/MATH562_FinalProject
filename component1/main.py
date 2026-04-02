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


def compute_first_term(n, d, cov_a, sigma2, beta_n):
    """
    Computes the first term of the expression s.t. we must choose gamma(d)
    s.t. this term doesn't vanish.
    More precisely, computes the squared norm of the gradient of the empirical risk.
    Goal : find the order of the squared norm of gradient of the empirical risk
    """
    A, b, theta_star = generate_data(n, d, cov_a, sigma2, beta_n)
    theta = np.random.randn(d)

    return np.linalg.norm(1/n * A.T @ (A @ theta - b))**2

def compute_second_term(n, d, cov_a, sigma2, beta_n):
    A, b, theta_star = generate_data(n, d, cov_a, sigma2, beta_n)
    theta = np.random.randn(d)
    
    grad = 1/n * A.T @ (A @ theta - b)
    H = H = 1/n * A.T @ A  # Hessian

    return grad.T @ H @ grad


def compute_third_term_sgd(n, d, cov_a, sigma2, beta_n, num_samples=10000):
    """
    Approximate trace(H @ Var(g_t)) by Monte Carlo over num_samples random i's
    """
    A, b, theta_star = generate_data(n, d, cov_a, sigma2, beta_n)
    theta = np.random.randn(d)

    H = 1/n * A.T @ A  # Hessian

    gts = np.zeros((num_samples, d))  # store g_t samples

    for k in range(num_samples):
        idx = np.random.randint(0, n)
        a_i = A[idx, :]
        b_i = b[idx]
        gts[k, :] = (a_i * (np.dot(a_i, theta) - b_i))/n 

    cov_gt = np.cov(gts, rowvar=False)  
    return np.trace(H @ cov_gt) 

def compute_third_term_batch(n, d, cov_a, sigma2, beta_n, batch_size, num_samples=10000):
    """
    Approximate trace(H @ Var(g_t_batch)) by Monte Carlo over num_samples random batches
    """
    A, b, theta_star = generate_data(n, d, cov_a, sigma2, beta_n)
    theta = np.random.randn(d)

    H = 1/(2*n) * A.T @ A  # Hessian

    gts = np.zeros((num_samples, d))  # store batch gradient samples

    for k in range(num_samples):
        idx_batch = np.random.choice(n, batch_size, replace=False)  # sample a batch
        A_batch = A[idx_batch, :]
        b_batch = b[idx_batch]

        # Batch gradient (scaled by 1/n as in your single-sample version)
        gts[k, :] = (A_batch.T @ (A_batch @ theta - b_batch)) / (n * batch_size)

    cov_gt = np.cov(gts, rowvar=False)
    return np.trace(H @ cov_gt)




def plot_risk(d_list, rho, num_runs, num_epochs, sigma2, sgd_algo, gamma, sigma_hat, step_type=None, batch_size=None):

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

            if batch_size is None:
                empirical_risk_hist, true_risk_hist = sgd_algo(
                    rho, d, cov_a, sigma2, beta_n, num_epochs, gamma, step_type
                )
            else:
                empirical_risk_hist, true_risk_hist = sgd_algo(
                    rho, d, cov_a, sigma2, beta_n, num_epochs, gamma, step_type, batch_size
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

        empirical_risk_hist.append(compute_empirical_risk(A, b, theta))
        true_risk_hist.append(compute_true_risk(theta_star, theta, cov_a, sigma2, beta_n))

        for _ in range(n):
            idx  = np.random.randint(0, n)
            a_idx = A[idx, :]
            b_idx = b[idx]
            theta -= 1/L * gamma(d) * 1/n * (np.dot(a_idx, theta) - b_idx)* a_idx 

    return empirical_risk_hist, true_risk_hist


###############################################################################
# Experiment 2 - Effect of randomness in SGD
###############################################################################

def single_suffle_sgd(rho, d, cov_a, sigma2, beta_n, num_epochs, gamma, step_type):

    empirical_risk_hist = []
    true_risk_hist = []

    n = int(d/rho)

    A, b, theta_star = generate_data(n, d, cov_a, sigma2, beta_n)

    theta = np.random.randn(d) # Random initialization

    permutation = np.random.permutation(n)

    # Set absolute constant L (for the learning rate gamma(d))
    U, S, Vh = np.linalg.svd(A) # To find the singular values of A^T A (and thus its eigenvalues)
    if step_type == "max":
        L = S[0]**2
    
    elif step_type == "avg":
        L = np.mean(S**2)

    else:
        L = 1

    for epoch in range(num_epochs):
        empirical_risk_hist.append(compute_empirical_risk(A, b, theta))
        true_risk_hist.append(compute_true_risk(theta_star, theta, cov_a, sigma2, beta_n))

        for i in range(n):
            idx  = permutation[i]
            a_idx = A[idx, :]
            b_idx = b[idx]
            theta -= 1/L * gamma(d) * 1/n * (np.dot(a_idx, theta) - b_idx)* a_idx

    return empirical_risk_hist, true_risk_hist


def multiple_suffle_sgd(rho, d, cov_a, sigma2, beta_n, num_epochs, gamma, step_type):

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
        empirical_risk_hist.append(compute_empirical_risk(A, b, theta))
        true_risk_hist.append(compute_true_risk(theta_star, theta, cov_a, sigma2, beta_n))

        permutation = np.random.permutation(n)
        for i in range(n):
            idx  = permutation[i]
            a_idx = A[idx, :]
            b_idx = b[idx]
            theta -= 1/L * gamma(d) * 1/n * (np.dot(a_idx, theta) - b_idx)* a_idx

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

        empirical_risk_hist.append(compute_empirical_risk(A, b, theta))
        true_risk_hist.append(compute_true_risk(theta_star, theta, cov_a, sigma2, beta_n))

        for _ in range(n):
            
            idx  = np.random.randint(0, n)
            a_idx = A[idx, :]
            b_idx = b[idx]
            theta_old = theta.copy()
            theta = theta - 1/L * gamma(d) * 1/n * (np.dot(a_idx, theta) - b_idx)* a_idx + delta * (theta - theta_prev)
            theta_prev = theta_old

    return empirical_risk_hist, true_risk_hist


def sgd_momentum_fixed_delta(rho, d, cov_a, sigma2, beta_n, num_epochs, gamma, step_type):
    """
    Wrapper function
    """
    return sgd_momentum(rho, d, cov_a, sigma2, beta_n, num_epochs, gamma, step_type, delta=0.5)


###############################################################################
# Experiment 4 - Small vs large batch sizes
###############################################################################

def sgd_batch(rho, d, cov_a, sigma2, beta_n, num_epochs, gamma, step_type, batch_size):

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

        empirical_risk_hist.append(compute_empirical_risk(A, b, theta))
        true_risk_hist.append(compute_true_risk(theta_star, theta, cov_a, sigma2, beta_n))

        for _ in range(int(n/batch_size(n))):
            idx  = np.random.choice(n, batch_size(n), replace=False) # Batch
            a_idx = A[idx, :]
            b_idx = b[idx]

            grad = (a_idx.T @ (a_idx @ theta - b_idx)) / (batch_size(n) * n)
            theta -= (gamma(d) / L) * grad

    return empirical_risk_hist, true_risk_hist
"""
Imports 
"""
import numpy as np
import matplotlib.pyplot as plt


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

def compute_true_risk(theta_star, theta, cov_a, sigma2, beta_n):
    return 1/2 * (theta_star - theta).T @ cov_a @ (theta_star - theta) + 1/2 * sigma2 * beta_n


def run_sgd(rho, d, cov_a, sigma2, beta_n, num_epochs, gamma):

    empirical_risk_hist = []
    true_risk_hist = []

    n = int(d/rho)

    A, b, theta_star = generate_data(n, d, cov_a, sigma2, beta_n)

    theta = np.random.randn(d) # Random initialization. TODO : Maybe change this

    for epoch in range(num_epochs):
        for _ in range(n):
            idx  = np.random.randint(0, n)
            a_idx = A[idx, :]
            b_idx = b[idx]
            theta -= gamma * 1/n * (np.dot(a_idx, theta) - b_idx)* a_idx

        empirical_risk_hist.append(compute_empirical_risk(A, b, theta))
        true_risk_hist.append(compute_true_risk(theta_star, theta, cov_a, sigma2, beta_n))

    return empirical_risk_hist, true_risk_hist


"""
Experiment 1
"""
rho = 1
d_list = [100, 200, 400, 800, 1600]
num_runs = 10
all_runs_empirical_risk = []
all_runs_true_risk = []
num_epochs = 100



# Testing
d = 100
cov_a = np.eye(d) # TODO: Change alpha_d
sigma2 = 1
beta_n =1
gamma = 1/d

for run in range(num_runs): # TODO : wrap in d 
    empirical_risk_hist, true_risk_hist = run_sgd(rho, d, cov_a, sigma2, beta_n, num_epochs, gamma)
    all_runs_empirical_risk.append(empirical_risk_hist)
    all_runs_true_risk.append(true_risk_hist)


all_runs_empirical_risk = np.array(all_runs_empirical_risk)
empirical_risk_hist_mean = np.mean(all_runs_empirical_risk, axis=0)
empirical_risk_hist_std = np.std(all_runs_empirical_risk, axis=0)

all_runs_true_risk = np.array(all_runs_true_risk)
true_risk_hist_mean = np.mean(all_runs_true_risk, axis=0)
true_risk_hist_std = np.std(all_runs_true_risk, axis=0)

# Plot empirical
plt.plot(np.arange(num_epochs), empirical_risk_hist_mean)
plt.fill_between(np.arange(num_epochs), 
                 empirical_risk_hist_mean - empirical_risk_hist_std, 
                 empirical_risk_hist_mean + empirical_risk_hist_std, 
                 alpha=0.2)
plt.xlabel("Num of epochs")
plt.ylabel("Empirical risk")
plt.title("Training error")
plt.yscale("log")
plt.show()

# Plot true risk
plt.plot(np.arange(num_epochs), true_risk_hist_mean)
plt.fill_between(np.arange(num_epochs), 
                 true_risk_hist_mean - true_risk_hist_std, 
                 true_risk_hist_mean + true_risk_hist_std, 
                 alpha=0.2)
plt.xlabel("Num of epochs")
plt.ylabel("True risk")
plt.title("True risk")
plt.yscale("log")
plt.show()

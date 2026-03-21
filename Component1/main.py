"""
Imports 
"""
import numpy as np


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


def run_sgd(rho, d, cov_a, sigma2, beta_n, num_epochs):

    empirical_risk_hist = []
    true_risk_hist = []

    n = int(d/rho)

    A, b, theta_star = generate_data(n, d, cov_a, sigma2, beta_n)

    theta = np.random.randn(d) # Random initialization. TODO : Maybe change this
    gamma = 1/d # Learning TODO : Optimize the constant

    for epoch in range(num_epochs):
        for _ in range(n):
            idx  = np.random.randint(0, n)
            a_idx = A[idx, :]
            b_idx = b[idx]
            theta -= gamma * 1/n * (np.dot(a_idx, theta) - b_idx)* a_idx

        empirical_risk_hist.append(compute_empirical_risk(A, b, theta))
        true_risk_hist.append(compute_true_risk(theta_star, theta, cov_a, sigma2, beta_n))

    return empirical_risk_hist, true_risk_hist

"""
Imports 
"""
import numpy as np


def generate_data(n, d, cov_a, sigma_2, beta_n):
    """
    :param cov_a: dxd covariance matrix of a
    :sigma_2: variance parameter for epsilon
    :beta_n: scaling for epsilon variance
    """

    A = np.random.multivariate_normal(np.zeros(d), cov_a, size=n) # Design matrix
    theta_star = np.random.multivariate_normal(np.zeros(d), np.eye(d)) # True param
    epsilon = np.random.multivariate_normal(np.zeros(n), sigma_2 * beta_n * np.eye(n)) # Noise

    b = A @ theta_star + epsilon

    return A, b, theta_star


def compute_empirical_risk(A, b, theta):

    n = A.shape[0]
    
    return 1/(2*n) * np.linalg.norm(A@theta - b)**2

def compute_true_risk(theta_star, theta, cov_a, sigma2, beta_n):
    return 1/2 * (theta_star - theta).T @ cov_a @ (theta_star - theta) + 1/2 * sigma2 * beta_n



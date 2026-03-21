"""
Imports 
"""
import numpy as np


def generate_data(n, d, cov_A, sigma_2, beta_n):
    """
    :param cov_A: dxd covariance matrix of A
    :sigma_2: variance parameter for epsilon
    :beta_n: scaling for epsilon variance
    """

    A = np.random.multivariate_normal(np.zeros(d), cov_A, size=n) # Design matrix
    theta_star = np.random.multivariate_normal(np.zeros(d), np.eye(d)) # True param
    epsilon = np.random.multivariate_normal(np.zeros(n), sigma_2 * beta_n * np.eye(n)) # Noise

    b = A @ theta_star + epsilon

    return A, b, theta_star
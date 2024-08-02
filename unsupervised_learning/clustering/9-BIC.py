#!/usr/bin/env python3
"""Gaussian Mixture Model"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian Information Criterion (BIC).

    X: np.ndarray of shape (n, d) containing the data set
    kmin: int containing the minimum number of clusters to check for (inclusive)
    kmax: int containing the maximum number of clusters to check for (inclusive)
    iterations: int containing the maximum number of iterations for the EM algorithm
    tol: float containing the tolerance for the EM algorithm
    verbose: bool, if True, prints information about the EM algorithm

    Returns:
    best_k: int, the best number of clusters based on BIC
    best_result: tuple containing (pi, m, S) for the best number of clusters
    l: np.ndarray of shape (kmax - kmin + 1) containing the log likelihood for each cluster size tested
    b: np.ndarray of shape (kmax - kmin + 1) containing the BIC value for each cluster size tested
    """

    # Set default value for kmax if not provided
    if kmax is None:
        kmax = X.shape[0]

    n, d = X.shape
    k_range = range(kmin, kmax + 1)
    
    log_likelihoods = []
    bic_values = []

    for k in k_range:
        # Run EM algorithm
        pi, m, S, g, l = expectation_maximization(X, k, iterations, tol, verbose)
        if pi is None or m is None or S is None or g is None or l is None:
            return None, None, None, None

        # Compute number of parameters p
        p = k * (d + d * (d + 1) / 2) + k - 1  # k * d (means) + k * d * (d + 1) / 2 (covariances) + k - 1 (priors)

        # Calculate BIC
        bic = p * np.log(n) - 2 * l

        log_likelihoods.append(l)
        bic_values.append(bic)

    # Convert lists to numpy arrays
    log_likelihoods = np.array(log_likelihoods)
    bic_values = np.array(bic_values)

    # Find the best number of clusters (minimum BIC)
    best_k = k_range[np.argmin(bic_values)]
    best_result = (pi, m, S)

    return best_k, best_result, log_likelihoods, bic_values

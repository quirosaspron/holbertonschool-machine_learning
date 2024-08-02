#!/usr/bin/env python3
"""Gaussian Mixture Model"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the Expectation-Maximization (EM) algorithm for a GMM.

    X: np.ndarray of shape (n, d) containing the data set
    k: int containing the number of clusters
    iterations: int containing the maximum number of iterations
    tol: float containing the tolerance for the log likelihood
    verbose: bool, if True, prints the log likelihood every 10 iterations

    Returns:
    pi: np.ndarray of shape (k,) containing the
    priors for each cluster
    m: np.ndarray of shape (k, d) containing the
    centroid means for each cluster
    S: np.ndarray of shape (k, d, d) containing the
    covariance matrices for each cluster
    g: np.ndarray of shape (k, n) containing the posterior
    probabilities for each data point
    l: float, the log likelihood of the model
    """

    # Initialize parameters
    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    log_likelihood_prev = None

    for i in range(iterations):
        # E-step
        g, log_likelihood = expectation(X, pi, m, S)
        if g is None or log_likelihood is None:
            return None, None, None, None, None

        # M-step
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        # Check for convergence
        log_l_prev = log_likelihood_prev
        log_l = log_likelihood
        if log_l_prev is not None and np.abs(log_l - log_l_prev) <= tol:
            break

        log_l_prev = log_l

        # Verbose output
        if verbose and (i + 1) % 10 == 0:
            print(f"Log Likelihood after {i + 1} iterations: {log_l:.5f}")

    # Final verbose output
    if verbose:
        print(f"Log Likelihood after {i + 1} iterations: {log_l:.5f}")

    return pi, m, S, g, log_l

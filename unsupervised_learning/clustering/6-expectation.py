#!/usr/bin/env python3
"""Gaussian Mixture Model"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Performs the expectation step of the EM algorithm for a GMM.
    X: np.ndarray of shape (n, d) containing the data set
    pi: np.ndarray of shape (k,) containing the priors for each cluster
    m: np.ndarray of shape (k, d) containing the centroid
    means for each cluster
    S: np.ndarray of shape (k, d, d) containing the covariance
    matrices for each cluster
    Returns:
    g: np.ndarray of shape (k, n) containing the posterior
    probabilities for each data point in each cluster
    l: float, the total log likelihood
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None

    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None

    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    # Initialize the posterior probabilities array
    g = np.zeros((k, n))

    # Calculate the PDF for each cluster and each data point
    for j in range(k):
        pdf_values = pdf(X, m[j], S[j])
        g[j, :] = pi[j] * pdf_values

    # Normalize the posterior probabilities
    g_sum = np.sum(g, axis=0)
    if np.any(g_sum == 0):  # To avoid division by zero
        return None, None
    g = g / g_sum

    # Calculate the total log likelihood
    likelihood = np.sum(np.log(g_sum))

    return g, likelihood

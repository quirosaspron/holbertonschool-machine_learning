#!/usr/bin/env python3
"""Gaussian Mixture Model"""
import numpy as np


def maximization(X, g):
    """
    Performs the maximization step of the EM algorithm for a GMM.

    X: np.ndarray of shape (n, d) containing the data set
    g: np.ndarray of shape (k, n) containing the posterior probabilities
    for each data point in each cluster

    Returns:
    pi: np.ndarray of shape (k,) containing the updated priors
    for each cluster
    m: np.ndarray of shape (k, d) containing the updated centroid
    means for each cluster
    S: np.ndarray of shape (k, d, d) containing the updated covariance
    matrices for each cluster
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    # Calculate the sum of posterior probabilities for each cluster
    Nk = np.sum(g, axis=1)
    if np.any(Nk == 0):
        return None, None, None  # Avoid division by zero
    # Update priors
    pi = Nk / n
    # Update means
    m = np.dot(g, X) / Nk[:, np.newaxis]
    # Update covariance matrices
    S = np.zeros((k, d, d))
    for j in range(k):
        X_centered = X - m[j]
        weighted_cov = np.dot(g[j] * X_centered.T, X_centered) / Nk[j]
        S[j] = weighted_cov

    return pi, m, S

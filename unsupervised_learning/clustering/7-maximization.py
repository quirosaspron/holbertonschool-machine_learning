#!/usr/bin/env python3
"""Gaussian Mixture Model"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM.
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
        not isinstance(g, np.ndarray) or g.ndim != 2 or
        X.shape[0] != g.shape[1] or
            not np.allclose(g.sum(axis=0), 1.0)):
        return None, None, None

    n, d = X.shape
    k, _ = g.shape

    # Calculate the update priors
    pi = np.sum(g, axis=1) / n

    # Calculate the updated means
    m = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]

    # Calculate the updated covariance matrices
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = np.dot(g[i] * diff.T, diff) / np.sum(g[i])

    return pi, m, S

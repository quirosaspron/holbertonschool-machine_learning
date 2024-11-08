#!/usr/bin/env python3
"""K-means clustering"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-meansi
    X: numpy.ndarray containing the dataset
    k: number of clusters
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    centroids = np.random.uniform(low, high, size=(k, d))
    return centroids

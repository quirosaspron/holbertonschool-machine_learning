#!/usr/bin/env python3
import numpy as np
"intra-cluster variance"


def variance(X, C):
    """Calculates the intra-cluster variance
       X: np.ndarray containing the dataset
       C: np.ndarray with centroid means for each cluster"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    # Adds a dimension to X and broadcasts X and C
    distance = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    # Distance shape (n, k) closest_centroids shape (n,)
    closest_centroids = np.argmin(distances, axis=1)
    variances = np.sum((X - C[closest_centroids]) ** 2, axis=1)
    total_variance = np.mean(variances)
    return total_variance

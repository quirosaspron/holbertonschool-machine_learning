#!/usr/bin/env python3
"""intra-cluster variance"""
import numpy as np


def variance(X, C):
    """
    Calculates the intra-cluster variance
    X: np.ndarray containing the dataset
    C: np.ndarray with centroid means for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    if X.shape[1] != C.shape[1]:
        return None

    centroids_extended = C[:, np.newaxis]
    distances = np.linalg.norm(X - centroids_extended, axis=2)
    min_distances = np.min(distances, axis=0)
    variance = np.sum(min_distances ** 2)

    return variance

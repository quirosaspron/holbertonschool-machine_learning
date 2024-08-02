#!/usr/bin/env python3
import numpy as np
"K-means clustering"


def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset
       X: numpy.ndarray containing the dataset
       k: number of clusters
       iterations: maximum number of iterations"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if type(k) != int or k <= 0:
        return None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    C = np.random.uniform(low, high, size=(k, d))

    for _ in range(iterations):
        # Assignment Step: Assign each point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)
        new_C = np.zeros((k, d))

        # Update Step: Calculate the new centroids
        for j in range(k):
            # Boolean Indexing
            points_in_cluster = X[clss == j]
            if len(points_in_cluster) == 0:
                new_C[j] = np.random.uniform(low, high, d)
            else:
                new_C[j] = points_in_cluster.mean(axis=0)

        # Check for convergence (if centroids do not change)
        if np.all(C == new_C):
            break
        
        C = new_C
    
    return C, clss

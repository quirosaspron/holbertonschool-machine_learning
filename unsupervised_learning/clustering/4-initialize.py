#!/usr/bin/env python3
"""GMM"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans  # Import the kmeans function


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.

    Parameters:
    - X: np.ndarray of shape (n, d) containing the dataset
    - k: int, the number of clusters

    Returns:
    - pi: np.ndarray of shape (k,) containing the priors for each
      cluster, initialized evenly
    - m: np.ndarray of shape (k, d) containing the centroid means
      for each cluster, initialized with K-means
    - S: np.ndarray of shape (k, d, d) containing the covariance
      matrices for each cluster, initialized as identity matrices
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    n, d = X.shape 
    # Initialize centroids using K-means
    C, _ = kmeans(X, k)
    # Initialize priors with equal values
    pi = np.full(k, 1 / k)
    # Initialize covariance matrices as identity matrices
    S = np.tile(np.identity(d), (k, 1, 1))
    return pi, C, S

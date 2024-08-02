#!/usr/bin/env python3
"""K-means"""
import numpy as np
from sklearn.cluster import KMeans


def kmeans(X, k):
    """
    Perform K-means clustering on a dataset.

    X: np.ndarray of shape (n, d) containing the dataset
    k: int, number of clusters

    Returns:
    C: np.ndarray of shape (k, d) containing the centroid means for each cluster
    clss: np.ndarray of shape (n,) containing the index of the cluster in C that each data point belongs to
    """
    # Ensure input is a numpy array
    X = np.asarray(X)
    
    # Initialize KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    
    # Fit the KMeans model
    kmeans.fit(X)
    
    # Retrieve the cluster centers (centroids) and labels (cluster assignments)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    
    return C, clss


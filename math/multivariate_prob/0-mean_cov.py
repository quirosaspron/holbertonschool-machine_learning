#!/usr/bin/env python3
"Mean and covariance calculation"
import numpy as np


def mean_cov(X):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    if X.shape[0] < 2:
        raise ValueError('X must contain multiple data points')

    mean = np.mean(X, axis=0, keepdims=True)
    cov = 1/(X.shape[0]-1) * np.matmul(X.T-mean.T, X-mean)
    return mean, cov

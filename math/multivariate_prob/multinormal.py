#!/usr/bin/env python3
"Multinormal class"
import numpy as np


class MultiNormal:
    def __init__(self, data):
        """Initializes the class"""

        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a \
2D numpy.ndarray')
        if data.shape[1] < 2:
            raise ValueError('data must contain \
multiple data points')
        mean = np.mean(data, axis=1, keepdims=True)
        self.mean = mean
        cov = np.matmul(data-mean, data.T-mean.T)/(data.shape[1]-1)
        self.cov = cov

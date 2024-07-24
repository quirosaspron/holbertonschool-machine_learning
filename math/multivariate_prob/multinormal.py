#!/usr/bin/env python3
"Multinormal class"
import numpy as np


class MultiNormal:
    """Multivariate Normal distribution representation"""

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

    def pdf(self, x):
        """Calculates the PDF at a data point"""

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        test_d, one = x.shape
        if test_d != d or one != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        pdf = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
        mult = np.matmul(np.matmul((x - self.mean).T, inv), (x - self.mean))
        pdf *= np.exp(-0.5 * mult)
        pdf = pdf[0][0]
        return pdf

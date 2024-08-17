#!/usr/bin/env python3
"""
Hyperparameters tunings
"""

import numpy as np


class GaussianProcess:
    """
    This class represents a noiseless Gaussian process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):

        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Radial Basis Function Kernel
        reutrn
        """

        # covariance kernel matrix, Gaussian process
        sqdist = np.sum(X1**2, 1).reshape(-1, 1)\
            + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp((-0.5 / self.l**2) * sqdist)

    def predict(self, X_s):
        """
        X_star: new data point
        """
        s = X_s.shape[0]
        mu = np.zeros(s)
        sigma = np.zeros(s)
        K = self.kernel(self.X, self.X)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)

        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        mu_s = mu_s.T[0, :]  # reshape

        # covariance non demandée mais utilisée pour la variance
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        # la variance se trouve sur la digonale de la matrice de covariance
        sigma = np.diag(cov_s)
        return mu_s, sigma

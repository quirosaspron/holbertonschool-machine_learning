#!/usr/bin/env python3
"""
Hyperparameters tunings
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """
    initilise the new class
    """
    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
        self.Y_init = Y_init
        self.X_init = X_init

    def acquisition(self):
        """
        Donne le prochain test a effectuer de maniere a trouver
        le min ou le max d'une fonction
        """
        mu, sigma = self.gp.predict(self.X_s)

        # Calculer la meilleure valeur observée jusqu'à présent
        if self.minimize:
            mu_sample_opt = np.min(self.Y_init)
            imp = mu_sample_opt - mu - self.xsi
        else:
            mu_sample_opt = np.max(self.Y_init)
            imp = mu - mu_sample_opt - self.xsi

        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei

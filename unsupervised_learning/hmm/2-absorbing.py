#!/usr/bin/env python3
"""
Defines function that determines if the Markov Chain is absorbing
"""
import numpy as np


def absorbing(P):
    """
    decide if a transition matrice is or not absorbing
    """
    n = P.shape[0]
    t = 300
    limit = np.zeros((n, n))
    limit = np.linalg.matrix_power(P, t)
    for i in range(n):
        if limit[i, i] == 1:
            absorb = True
            for j in range(n):
                if i != j and (not np.isclose(limit[i, j], 0.0)):
                    absorb = False
            if absorb:
                return True
    return False

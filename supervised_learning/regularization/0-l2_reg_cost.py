#!/usr/bin/env python3
"""calculates the cost with L2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Returns: Cost with L2 regularization"""
    reg_term = 0
    for i in range(1, L+1):
        reg_term += np.sum(weights[f'W{i}']**2)
    return cost + lambtha / (2 * m) * reg_term

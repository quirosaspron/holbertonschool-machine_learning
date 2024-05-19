#!/usr/bin/env python3
"""updates a variable using gd with momentum"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Returns: updated variable and the new moment"""
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v

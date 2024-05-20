#!/usr/bin/env python3
"""normalizes an unactivated output of a neural
network using batch normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Returns: the normalized Z matrix"""
    m, nx = Z.shape
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    Z_tilda = gamma * Z_norm + beta
    return Z_tilda

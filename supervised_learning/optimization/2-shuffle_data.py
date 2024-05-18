#!/usr/bin/env python3
"""shuffles the data points in two matrices the same way"""
import numpy as np


def shuffle_data(X, Y):
    """Returns: the shuffled matrices"""
    X = np.random.permutation(X)
    Y = np.random.permutation(Y)
    return X, Y

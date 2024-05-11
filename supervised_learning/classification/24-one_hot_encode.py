#!/usr/bin/env python3
"""converts a numeric label vector into a one-hot matrix"""
import numpy as np


def one_hot_encode(Y, classes):
    """Returns: a one-hot encoding of Y"""
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    m = Y.shape[0]
    if not isinstance(classes, int) or classes < m:
        return None
    one_hot = np.zeros((classes, m))
    for i in range(m):
        if Y[i] < 0 or Y[i] >= classes:
            return None
        one_hot[Y[i], i] = 1
    return one_hot

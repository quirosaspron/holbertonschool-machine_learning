#!/usr/bin/env python3
"""Converts a one-hot matrix into a vector of labels"""
import numpy as np


def one_hot_decode(one_hot):
    """Returns: An array with the numeric labels for each example"""
    if not isinstance(one_hot, np.ndarray) or not one_hot.ndim == 2:
        return None
    if not np.array_equal(one_hot, one_hot.astype(bool)):
        return None
    if not np.allclose(np.sum(one_hot, axis=0), 1):
        return None
    decoded = np.empty(one_hot.shape[1], dtype=int)
    for j in range(one_hot.shape[1]):
        for i in range(one_hot.shape[0]):
            if one_hot[i][j] == 1:
                decoded[j] = i
    return decoded

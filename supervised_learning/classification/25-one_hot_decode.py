#!/usr/bin/env python3
"""Converts a one-hot matrix into a vector of labels"""
import numpy as np


def one_hot_decode(one_hot):
    """Returns: An array with the numeric labels for each example"""
    decoded = np.array([])
    for j in range(one_hot.shape[1]):
        for i in range(one_hot.shape[0]):
            if one_hot[i][j] == 1:
               decoded = np.append(decoded, i)
    return decoded.astype(int)

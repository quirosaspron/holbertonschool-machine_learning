#!/usr/bin/env python3
"""Creates mini-batches to be used for training"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Returns: list of mini-batches"""
    X, Y = shuffle_data(X, Y)
    batches = []
    m = X.shape[0]
    for i in range(0, m, batch_size):
        batch_X = X[i:i+batch_size]
        batch_Y = Y[i:i+batch_size]
        batches.append((batch_X, batch_Y))
    return batches

#!/usr/bin/env python3
"""Concatenates two matrices along a specific axis"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Returns the matrices concatenated"""
    return np.concatenate((mat1, mat2), axis)

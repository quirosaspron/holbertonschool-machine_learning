#!/usr/bin/env python3
"""Calculates the definiteness of a matrix"""
import numpy as np


def definiteness(matrix):
    """Returns the definiteness of a matrix"""

    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')

    size = matrix.shape[0]

    if matrix.shape != (size, size):
        return None

    dets_list = []
    for i in range(size):
        det = np.linalg.det(matrix[:i+1, :i+1])
        dets_list.append(det)

    definiteness = 'Positive definite'
    for i in range(len(dets_list)):
        if dets_list[-1] == 0:
            if dets_list[i] <= 0 and definiteness != 'Positive semi-definite':
                definiteness = 'Negative semi-definite'
            elif dets_list[i] >= 0 and definiteness != 'Negative \
semi-definite':
                definiteness = 'Positive semi-definite'
            else:
                return None

        if dets_list[i] < 0:
            if (i+1) % 2 == 0:
                definiteness = 'Indefinite'
                break
            else:
                definiteness = 'Negative definite'

    return definiteness

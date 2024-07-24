#!/usr/bin/env python3
"Mean and covariance calculation"
import numpy as np


def correlation(C):
    """Returns the correlation matrix of C
     C: shape (d, d)
     n: number of dimensions"""

    if not isinstance(C, np.ndarray):
        raise TypeError('C must be a numpy.ndarray')
    d = C.shape[0]
    if C.shape != (d, d):
        raise ValueError('C must be a 2D square matrix')

    D = np.sqrt(np.diag(C))
    D_inverse = 1 / np.outer(D, D)
    corr = D_inverse * C
    return corr

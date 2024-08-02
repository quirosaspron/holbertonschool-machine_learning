#!/usr/bin/env python3
"""Gaussian Mixture Model"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution
    Parameters:
    - X: np.ndarray of shape (n, d) containing the data points
    - m: np.ndarray of shape (d,) containing the mean of the distribution
    - S: np.ndarray of shape (d, d) containing the covariance
      of the distribution
    Returns:
    - P: np.ndarray of shape (n,) containing the PDF values
      for each data point
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape

    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None

    # Calculate the determinant of S
    det_S = np.linalg.det(S)
    if det_S == 0:
        return None
    # Calculate the inverse of S
    inv_S = np.linalg.inv(S)
    # Center the data points
    X_centered = X - m
    # Calculate the exponent term
    exponent = -0.5 * np.sum(np.dot(X_centered, inv_S) * X_centered, axis=1)
    # Calculate the coefficient
    coefficient = 1 / np.sqrt((2 * np.pi) ** d * det_S)
    # Calculate the PDF values
    P = coefficient * np.exp(exponent)
    # Ensure the PDF values have a minimum value of 1e-300
    P = np.maximum(P, 1e-300)
    return P

#!/usr/bin/env python3
""" Marginal calculation """
import numpy as np


def marginal(x, n, P, Pr):
    """"Calculates the marginal probability of obtaining the
     data:
     x:number of patients that develop severe side effects
     n: total number of patients observed
     P: numpy.ndarray containing the probabilities of
     developing severe side effects
     Pr is a 1D numpy.ndarray containing the
     prior beliefs of P
     Returns:  numpy.ndarray containing the intersection
     of obtaining the data, x and n, for each probability
     in P, respectively"""

    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError('x must be an integer \
that is greater than or equal to 0')
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError('Pr must be a numpy.ndarray with the \
same shape as P')
    for a, b in zip(P, Pr):
        if not (0 <= a <= 1):
            raise ValueError('All values in P must be in the range [0, 1]')
        if not (0 <= b <= 1):
            raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Get likelihood
    factorial = np.math.factorial
    fact_coefficient = factorial(n) / (factorial(n - x) * factorial(x))
    likelihood = fact_coefficient * (P ** x) * ((1 - P) ** (n - x))
    # Get intersection
    intersection = likelihood * Pr
    # Calculate marginal
    marginal = np.sum(intersection)
    return marginal

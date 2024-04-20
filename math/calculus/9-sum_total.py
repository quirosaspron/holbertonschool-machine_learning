#!/usr/bin/env python3
"""Calculates a sum"""


def summation_i_squared(n):
    """Sums i squared n times"""
    if n < 1 or type(n) != int:
        return None
    else:
        return int((n*(n+1)*(2*n+1))/6)

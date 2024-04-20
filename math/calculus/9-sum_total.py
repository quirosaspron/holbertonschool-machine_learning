#!/usr/bin/env python3
"""Calculates a sum"""


def summation_i_squared(n):
    """Sums i squared n times"""
    if not n <= 1:
        return None
    else:
        return int((n*(n+1)*(2*n+1))/6)

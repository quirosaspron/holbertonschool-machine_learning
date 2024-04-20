#!/usr/bin/env python3
"""Calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """Derives a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    elif len(poly) == 1:
        return [0]
    else:
        poly[0] = 0
        for i, x in enumerate(poly):
            if not isinstance(x, int):
                return None
            elif i > 0:
                poly[i-1] = i*x
        poly.pop()
        return poly

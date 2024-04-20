#!/usr/bin/env python3
"""Calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Integrates a polynomial"""
    if not isinstance(poly, list) or not all(isinstance(coeff, (int, float)) for coeff in poly) or not isinstance(C, int):
        return None
    if len(poly) == 0:
        return None
    integral_coeffs = [C]
    for i in range(len(poly)):
        if i == 0:
            integral_coeffs.append(poly[i] / (i + 1))
        else:
            integral_coeffs.append(poly[i] / (i + 1))
    while integral_coeffs[-1] == 0 and len(integral_coeffs) > 1:
        integral_coeffs.pop()
    
    return integral_coeffs

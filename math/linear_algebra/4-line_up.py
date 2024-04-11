#!/usr/bin/env python3
"""Adds two arrays together"""


def add_arrays(arr1, arr2):
    """Adds two arrays"""
    if len(arr1) != len(arr2):
        return None
    suma = []
    for i in range(len(arr1)):
        suma.append(arr1[i] + arr2[i])
    return suma

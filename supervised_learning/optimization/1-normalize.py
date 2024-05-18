#!/usr/bin/env python3
"""normalizes (standardizes) a matrix"""
import numpy as np


def normalize(X, m, s):
    """Returns: the normalized matrix"""
    return (X - m) / s

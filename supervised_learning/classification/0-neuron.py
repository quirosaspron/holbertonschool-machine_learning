#!/usr/bin/env python3
import numpy as np
"""Contains a single neuron class"""


class Neuron:
    """Defines a single neuron"""
    def __init__(self, nx):
        """Initializes the weights, bias and activated output"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0

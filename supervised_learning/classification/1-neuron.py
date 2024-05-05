#!/usr/bin/env python3
"""Contains a single neuron class performing binary classification"""
import numpy as np


class Neuron:
    """Defines a single neuron"""
    def __init__(self, nx):
        """Initializes the weights, bias and activated output"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Gets the weight attribute"""
        return self.__W

    @property
    def b(self):
        """Gets the bias attribute"""
        return self.__b

    @property
    def A(self):
        """Gets the activated output attribute"""
        return self.__A

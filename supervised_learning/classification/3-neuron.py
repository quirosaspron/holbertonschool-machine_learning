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

    def forward_prop(self, X):
        """Peforms forward propagation using the sigmoid activation"""
        op = np.dot(self.__W, X) + self.__b
        self.__A = 1/(1 + np.exp(-op))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost function"""
        cost = -1/Y.shape[1] * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost





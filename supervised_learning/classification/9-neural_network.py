#!/usr/bin/env python3
"""Builds a neural network"""
import numpy as np


class NeuralNetwork():
    """Neural network class"""
    def __init__(self, nx, nodes):
        """Initializes the neural net"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Gets the weight 1 attribute"""
        return self.__W1

    @property
    def b1(self):
        """Gets the bias 1 attribute"""
        return self.__b1

    @property
    def A1(self):
        """Gets the hidden activated output"""
        return self.__A1

    @property
    def W2(self):
        """Gets the weight 2 attribute"""
        return self.__W2

    @property
    def b2(self):
        """Gets the bias 2 attribute"""
        return self.__b2

    @property
    def A2(self):
        """Gets the final activated output"""
        return self.__A2

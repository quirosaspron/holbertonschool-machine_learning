#!/usr/bin/env python3
"""Defines a deep neural network"""
import numpy as np


class DeepNeuralNetwork():
    """Deep neural network class"""
    def __init__(self, nx, layers):
        """Initializes the deep neural network"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(len(layers)):
            if not isinstance(i, int) or layers[i] <= 0:
                raise TypeError('layers must be a list of positive integers')
            if i == 0:
                self.__weights['W1'] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
                self.__weights['b1'] = np.zeros((layers[i], 1))
            else:
                self.__weights[f'W{i+1}'] = np.random.randn(
                    layers[i], layers[i-1]) * np.sqrt(2 / layers[i-1])
                self.__weights[f'b{i+1}'] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Returns the number of layers"""
        return self.__L

    @property
    def cache(self):
        """The getter for the cache attribute"""
        return self.__cache

    @property
    def weights(self):
        """The getter for the weights and biases"""
        return self.__weights

    def forward_prop(self, X):
        """Performs the forward propagation"""
        self.__cache['A0'] = X
        for i in range(self.__L):
            W = 'W' + str(i+1)
            b = 'b' + str(i+1)
            A = 'A' + str(i)
            activation = np.dot(self.__weights[W],
                                self.__cache[A]) + self.__weights[b]
            self.__cache[f'A{i+1}'] = 1 / (1 + np.exp(-activation))
        return self.__cache[f'A{self.__L}'], self.__cache

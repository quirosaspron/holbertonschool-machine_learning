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

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(len(layers)):
            if not isinstance(i, int) or layers[i] < 1:
                raise TypeError('Layers must be a list of positive integers')
            if i == 0:
                self.weights['W1'] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
                self.weights['b1'] = np.zeros((layers[i], 1))
            else:
                self.weights[f'W{i+1}'] = np.random.randn(
                    layers[i], layers[i-1]) * np.sqrt(2 / layers[i-1])
                self.weights[f'b{i+1}'] = np.zeros((layers[i], 1))

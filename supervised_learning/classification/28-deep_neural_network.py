#!/usr/bin/env python3
"""Defines a deep neural network"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class DeepNeuralNetwork():
    """Deep neural network class"""
    def __init__(self, nx, layers, activation='sig'):
        """Initializes the deep neural network"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

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
    def activation(self):
        """Activation atribute getter"""
        return self.__activation

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

    def softmax(self, Z):
        """Softmax activation function, reducing max for overflow"""
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward_prop(self, X):
        """Performs forward prop with softmax for multiclass classification"""
        self.__cache['A0'] = X
        for i in range(self.__L):
            W = 'W' + str(i+1)
            b = 'b' + str(i+1)
            A = 'A' + str(i)
            activation = np.dot(self.__weights[W],
                                self.__cache[A]) + self.__weights[b]
            if i == self.__L - 1:
                self.__cache[f'A{i+1}'] = self.softmax(activation)
            elif self.__activation == 'sig':
                self.__cache[f'A{i+1}'] = 1 / (1 + np.exp(-activation))
            else:
                self.__cache[f'A{i+1}'] = np.tanh(activation)
        return self.__cache[f'A{self.__L}'], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model with cross entropy"""
        m = Y.shape[1]
        cost_function = -1 / m * np.sum(Y * np.log(1.0000001 - A))
        return cost_function

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions"""
        self.forward_prop(X)
        cost = self.cost(Y, self.__cache[f'A{self.__L}'])
        predictions = np.where(self.__cache[f'A{self.__L}'] == np.max(
                    self.__cache[f'A{self.__L}'], axis=0), 1, 0)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        dz = self.cache[f'A{self.L}'] - Y
        for i in range(self.L, 0, -1):
            A_prev = cache[f'A{i-1}']
            dW = (1 / m) * np.dot(dz, A_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            W = self.weights[f'W{i}']
            b = self.weights[f'b{i}']
            dz = np.dot(W.T, dz) * (A_prev * (1 - A_prev))
            self.__weights[f'W{i}'] -= alpha * dW
            self.__weights[f'b{i}'] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neural network"""
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        y_costs = []
        for i in range(0, iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            if i % step == 0 or i == iterations:
                cst = self.cost(Y, self.__cache[f'A{self.__L}'])
                y_costs.append(cst)
                if verbose:
                    print(f'Cost after {i} iterations: {cst}')
        if graph:
            x_iterations = [i for i in range(0, iterations // step)]
            plt.plot(x_iterations, y_costs, color='skyblue')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if ".pkl" not in filename:
            filename += ".pkl"
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, "rb") as file:
                unpickled_obj = pickle.load(file)
            return unpickled_obj

        except FileNotFoundError:
            return None

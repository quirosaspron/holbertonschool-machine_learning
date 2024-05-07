#!/usr/bin/env python3
"""Builds a neural network"""

import numpy as np
import matplotlib.pyplot as plt


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

    def forward_prop(self, X):
        """Calculates the forward propagation"""
        activation_1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1+np.exp(-activation_1))
        activation_2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1+np.exp(-activation_2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model with logreg"""
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions"""
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        predictions = np.where(self.__A2 >= 0.5, 1, 0)
        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Computes the gradient descent"""
        m = X.shape[1]

        dz2 = A2 - Y
        dW2 = 1/m * np.dot(dz2, A1.T)
        db2 = 1/m * np.sum(dz2)

        dz1 = np.dot(self.__W2.T, dz2) * A1 * (1 - A1)
        dW1 = 1/m * np.dot(dz1, X.T)
        db1 = 1/m * np.sum(dz1)

        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2

        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1

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
            output = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            if i % step == 0:
                y_costs.append(self.cost(Y, self.__A2))
                if verbose:
                    cst = self.cost(Y, self.__A2)
                    print(f'Cost after {i} iterations: {cst}')
        if graph:
            x_iterations = [i for i in range(0, iterations // step)]
            plt.plot(x_iterations, y_costs, color='skyblue')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

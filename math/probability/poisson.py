#!/usr/bin/env python3
"""Creates a poisson distribution class"""


class Poisson:
    """Poisson class"""
    def __init__(self, data=None, lambtha=1):
        "Sets the lambtha attribute"
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.lambtha = self.mean(data)

    def mean(self, data):
        "Calculates the mean of dataset"
        sumation = 0
        for i in data:
            sumation += i
        return sumation / len(data)

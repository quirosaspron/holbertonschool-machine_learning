#!/usr/bin/env python3
"""Creates a exponential distribution class"""


class Exponential:
    """Exponential class"""
    e = 2.7182818285

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
                self.lambtha = 1/self.mean(data)

    def mean(self, data):
        "Calculates the mean of dataset"
        sumation = 0
        for i in data:
            sumation += i
        return sumation / len(data)

    def pdf(self, k):
        "Calculates the probability density function"
        if k < 0:
            return 0
        lambtha = self.lambtha
        exp = Exponential.e
        pdf = lambtha * (exp**(-lambtha*k))
        return pdf

    def cdf(self, k):
        "Calculates the cumulative distribution function"
        if k < 0:
            return 0
        e = Exponential.e
        lambtha = self.lambtha
        cdf = 1 - e**-lambtha*k
        return cdf

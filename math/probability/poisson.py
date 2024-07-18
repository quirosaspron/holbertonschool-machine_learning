#!/usr/bin/env python3
"""Creates a poisson distribution class"""


class Poisson:
    """Poisson class"""
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
                self.lambtha = self.mean(data)

    def mean(self, data):
        "Calculates the mean of dataset"
        sumation = 0
        for i in data:
            sumation += i
        return sumation / len(data)

    def factorial(self, number):
        "Calculates the factorial of a number"
        if number == 1 or number == 0:
            return 1
        else:
            return self.factorial(number - 1) * number

    def pmf(self, k):
        "Calculates the probability mass function"
        k = int(k)
        if k < 0:
            return 0
        lambtha = self.lambtha
        exp = Poisson.e
        fact = self.factorial(k)
        pmf = ((lambtha**k)*(exp**-lambtha))/fact
        return pmf

    def cdf(self, k):
        "Calculates the cumulative distribution function"
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        while k >= 0:
            cdf += self.pmf(k)
            k += -1
        return cdf

#!/usr/bin/env python3
"""Creates a binomial distribution class"""


class Binomial:
    """Binomial class"""
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, n=1, p=0.5):
        "Sets n and p attributes"
        if data is None:
            if not n > 0:
                raise ValueError('n must be a positive value')
            if p > 1 or p < 0:
                raise ValueError('p must be greater than 0 and less than 1')
            else:
                self.n = int(n)
                self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.n, self.p = self.calculate_parameters(data)

    def calculate_parameters(self, data):
        """Calculates n and p from the given data"""
        mean = float(sum(data) / len(data))
        summation = 0
        for x in data:
            summation += ((x - mean) ** 2)
        variance = (summation / len(data))
        q = variance / mean
        p = (1 - q)
        n = round(mean / p)
        p = float(mean / n)
        return n, p

    def pmf(self, k):
        """
        calculates the value of the PMF for a given number of successes

        parameters:
            k [int]: number of successes
                If k is not an int, convert it to int
                If k is out of range, return 0

        return:
            the PMF value for k
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        p = self.p
        n = self.n
        q = (1 - p)
        n_factorial = 1
        for i in range(n):
            n_factorial *= (i + 1)
        k_factorial = 1
        for i in range(k):
            k_factorial *= (i + 1)
        nk_factorial = 1
        for i in range(n - k):
            nk_factorial *= (i + 1)
        binomial_co = n_factorial / (k_factorial * nk_factorial)
        pmf = binomial_co * (p ** k) * (q ** (n - k))
        return pmf

    def cdf(self, k):
        """
        calculates the value of the CDF for a given number of successes

        parameters:
            k [int]: number of successes
                If k is not an int, convert it to int
                If k is out of range, return 0

        return:
            the CDF value for k
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf

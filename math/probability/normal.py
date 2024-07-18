#!/usr/bin/env python3
"""Creates a normal distribution class"""


class Normal:
    """Normal class"""
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        "Sets the lambtha attribute"
        if data is None:
            if stddev < 0:
                raise ValueError('stddev must be a positive value')
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.mean = self.get_mean(data)
                self.stddev = self.get_stddev(data)

    def get_mean(self, data):
        "Calculates the mean of dataset"
        sumation = 0
        for i in data:
            sumation += i
        return sumation / len(data)

    def get_stddev(self, data):
        "Calculates the standard deviation of dataset"
        mean = self.get_mean(data)
        n = len(data)
        sumation = 0
        for i in data:
            sumation += (i - mean)**2
        return (1/n * sumation)**(1/2)

    def z_score(self, x):
        "Calculates the z-score of a given x-value"
        mean = self.mean
        stddev = self.stddev
        return (x - mean) / stddev

    def x_value(self, z):
        "Calculates the x-value of a given z-score"
        mean = self.mean
        stddev = self.stddev
        return mean + z * stddev

    def pdf(self, x):
        "Calculates the probability density function"
        pi = Normal.pi
        exp = Normal.e
        stddev = self.stddev
        mean = self.mean
        return (1 / (stddev * (2 * pi)**(1/2))) * \
            exp**(-0.5 * ((x - mean) / stddev) ** 2)

    def erf(self, x):
        "Erf fucntion"
        pi = Normal.pi
        return 2/((pi)**(1/2))*(x-(x**3/3)+(x**5/10)
                                - (x**7/42)+(x**9/216))

    def cdf(self, x):
        "Calculates the cumulative distribution function"
        stddev = self.stddev
        mean = self.mean
        return 0.5 * (1 + self.erf((x - mean) / (stddev * (2**(1/2)))))

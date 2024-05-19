#!/usr/bin/env python3
"""calculates the weighted moving average of a data set"""
import numpy as np


def moving_average(data, beta):
    """Weighted moving averages of data"""
    averages = []
    v = 0
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        averages.append(v / (1 - beta ** (i + 1)))
    return averages

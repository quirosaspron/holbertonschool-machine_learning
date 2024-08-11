#!/usr/bin/env python3
"""
Defines function that performs the forward algorithm for a Hidden Markov Model
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    forward algorithm
    """
    n = len(Observation)
    num_S = Transition.shape[0]

    a = np.empty([num_S, n], dtype='float')
    # Base case
    a[:, 0] = np.multiply(Initial[:, 0], Emission[:, Observation[0]])

    # Recursive case
    for t in range(1, n):
        a[:, t] = np.multiply(Emission[:, Observation[t]],
                              np.dot(Transition.T, a[:, t-1]))

    return (np.sum(a[:, n - 1]), a)

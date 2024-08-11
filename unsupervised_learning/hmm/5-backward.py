#!/usr/bin/env python3
"""
Defines function that performs the backward algorithm for a Hidden Markov Model
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    '''
    Input: Initial distribution v; Transition probability P; Emission
    probability E;
    Observations from HMM: Obers (Integer list with outcomes).
    Output: (backward probabilities for all the time steps, likelihood value
    based on 'Obser')
    '''
    n = len(Observation)
    num_S = Transition.shape[0]
    b = np.empty([num_S, n], dtype='float')
    # Base case
    b[:, n-1] = 1
    # Recursive case
    for t in reversed(range(n-1)):
        b[:, t] = np.dot(Transition, np.multiply(Emission[:, Observation[t+1]],
                                                 b[:, t+1]))

    return (np.dot(Initial[:, 0], np.multiply(Emission[:, Observation[0]],
                                              b[:, 0])), b)

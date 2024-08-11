#!/usr/bin/env python3
"""
Defines function that calculates that most likely sequence of hidden states for
the Hidden Markov Model
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    '''
    Input: Initial distribution v; Transition probability P;
    Emission probability E;
    Observations from HMM: Obers (Integer list with outcomes).
    Output: (the most likely sequence of states, maximal joint probability)
    '''

    n = len(Observation)
    num_S = Transition.shape[0]
    d = np.empty([num_S, n], dtype='float')
    f = np.empty([num_S, n-1], dtype=int)    # Matrix for backtracking

    # Base case
    d[:, 0] = np.multiply(Initial[:, 0], Emission[:, Observation[0]])

    # Recursive case
    for t in range(1, n):         # NUMBER OF STEP
        for i in range(num_S):    # NUMBER OF STATES
            temp_vec = np.multiply(d[:, t-1], Transition[:, i])
            f[i, t-1] = np.argmax(temp_vec)
            d[i, t] = Emission[i, Observation[t]] * np.max(temp_vec)

    p_star = np.max(d[:, n-1])    # max sequence
    most_lik = []

    # The last element of the most likely sequence of states
    x = np.argmax(d[:, n-1])
    most_lik.append(int(x))
    # Backtracking
    for t in reversed(range(n-1)):
        x = f[x, t]
        most_lik.append(int(x))

    return most_lik[::-1], p_star

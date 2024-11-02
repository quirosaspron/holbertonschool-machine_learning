#!/usr/bin/env python3
"""uses epsilon-greedy to determine the next action"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """"
    Q: np.ndarray q-table
    state: current state
    epsilon: epsilon to use for the calculation
    Returns: the next action index
    """
    p = np.random.uniform()
    if p > epsilon:  # Exploitation
        action = np.argmax(Q[state, :])
    else:  # Exploration
        action = np.random.randint(0, 4)

    return action

#!/usr/bin/env python3
"""initializes the Q-table:"""
import numpy as np


def q_init(env):
    """"
    env is the FrozenLakeEnv instance
    Returns: the Q-table as a numpy.ndarray of zeros
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))
    return q_table

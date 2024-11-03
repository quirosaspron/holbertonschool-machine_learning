#!/usr/bin/env python3
"""Has the trained agent play an episode"""
import numpy as np


def play(env, Q, max_steps=100):
    """"
    env: FrozenLakeEnv
    Q: numpy.ndarray containing the Q-table
    max_steps: maximum number of steps in the episode
    Returns: The total rewards for the episode and
    a list of rendered outputs representing the board
    state at each step
    """
    state = env.reset()[0]
    rewards = 0
    render_outputs = []
    steps = 0
    truncated = False
    terminated = False

    while (steps <= max_steps and not truncated and not terminated):
        render_outputs.append(env.render())

        action = np.argmax(Q[state, :])
        new_state, reward, terminated, truncated, _ = env.step(action)

        rewards += reward
        state = new_state
        steps += steps

    env.close()
    return rewards, render_outputs

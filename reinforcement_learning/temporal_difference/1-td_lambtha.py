#!/usr/bin/env python3
"""Temporal difference lambtha algorithm"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm to estimate the value function V.

    Parameters:
    - env: The environment instance.
    - V: A numpy.ndarray of shape (s,) containing
          the value estimates for each state.
    - policy: A function that takes in a state
          and returns the next action to take.
    - lambtha: The eligibility trace factor.
    - episodes: Total number of episodes to train over.
    - max_steps: Maximum number of steps per episode.
    - alpha: Learning rate.
    - gamma: Discount rate.

    Returns:
    - V: The updated value estimate.
    """
    for episode in range(episodes):
        # Initialize eligibility traces for each state to zero
        eligibility_trace = np.zeros_like(V)

        # Initialize the starting state
        state = env.reset()
        # Handle cases where env.reset() returns a tuple
        if isinstance(state, tuple):
            state = state[0]

        for _ in range(max_steps):
            # Choose an action using the policy
            action = policy(state)

            # Take the action and observe the next state, reward, and done flag
            result = env.step(action)
            if isinstance(result, tuple):
                next_state, reward, done = result[:3]
            else:
                next_state, reward, done = result, None, False

            # Compute the TD error (δ)
            td_error = reward + gamma * V[next_state] - V[state]

            # Update eligibility trace for the current state
            eligibility_trace[state] += 1

            # Update value estimates and eligibility traces for all states
            V += alpha * td_error * eligibility_trace
            eligibility_trace *= gamma * lambtha  # Decay eligibility traces

            # Transition to the next state
            state = next_state

            # End episode if done
            if done:
                break

    return V

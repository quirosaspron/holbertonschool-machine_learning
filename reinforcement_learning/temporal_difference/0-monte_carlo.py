#!/usr/bin/env python3
"""Monte Carlo algorithm"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to estimate the value function V.

    Parameters:
    - env: The environment instance.
    - V: A numpy.ndarray of shape (s,)
          containing the value estimates for each state.
    - policy: A function that takes in a state and
          returns the next action to take.
    - episodes: Total number of episodes to train over.
    - max_steps: Maximum number of steps per episode.
    - alpha: Learning rate.
    - gamma: Discount rate.

    Returns:
    - V: The updated value estimate.
    """
    for episode in range(episodes):
        # Generate an episode
        state = env.reset()
        # If env.reset() returns a tuple, extract the state
        if isinstance(state, tuple):
            state = state[0]

        episode_history = []

        for _ in range(max_steps):
            action = policy(state)
            result = env.step(action)

            # If env step, returns a tuple, extract state, reward and done
            if isinstance(result, tuple):
                next_state, reward, done = result[:3]
            else:
                next_state, reward, done = result, None, False

            episode_history.append((state, reward))
            state = next_state

            if done:
                break

        # Calculate the return G for each state in the episode
        G = 0
        visited_states = set()
        for t in reversed(range(len(episode_history))):
            state, reward = episode_history[t]
            G = reward + gamma * G

            # Check if it's the first occurrence of the state in this episode
            if state not in visited_states:
                visited_states.add(state)

                # Update the value estimate using incremental update rule
                V[state] += alpha * (G - V[state])

    return V

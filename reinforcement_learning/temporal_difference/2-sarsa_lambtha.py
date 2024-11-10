#!/usr/bin/env python3
"""
Defines function to perform the SARSA(λ) algorithm
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm to update the Q table.

    Parameters:
    - env: The environment instance.
    - Q: A numpy.ndarray of shape (s, a) containing the Q table.
    - lambtha: The eligibility trace factor.
    - episodes: Total number of episodes to train over.
    - max_steps: Maximum number of steps per episode.
    - alpha: Learning rate.
    - gamma: Discount rate.
    - epsilon: Initial threshold for epsilon-greedy policy.
    - min_epsilon: Minimum value that epsilon should decay to.
    - epsilon_decay: Decay rate for epsilon.

    Returns:
    - Q: The updated Q table.
    """
    n_states, n_actions = Q.shape

    def epsilon_greedy_policy(state, epsilon):
        """ Selects an action using epsilon-greedy policy. """
        if np.random.rand() < epsilon:
            return np.random.choice(n_actions)  # Explore: random action
        else:
            return np.argmax(Q[state])          # Exploit: best action
    for episode in range(episodes):
        # Reset the environment and initialize the eligibility trace
        state = env.reset()
        # Handle cases where env.reset() returns a tuple
        if isinstance(state, tuple):
            state = state[0]
        eligibility_trace = np.zeros_like(Q)
        # Choose action using epsilon-greedy policy
        action = epsilon_greedy_policy(state, epsilon)
        for step in range(max_steps):
            # Take action, observe reward and next state
            next_state, reward, done, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            # Choose next action using epsilon-greedy policy
            next_action = epsilon_greedy_policy(next_state, epsilon)
            # Calculate the TD error (δ)
            td_error = reward + (
                    gamma * Q[next_state, next_action] - Q[state, action])
            # Update the eligibility trace for the state-action pair
            eligibility_trace[state, action] += 1
            # Update Q values and eligibility traces for all state-action pairs
            Q += alpha * td_error * eligibility_trace
            eligibility_trace *= gamma * lambtha  # Decay eligibility traces
            # Transition to the next state and action
            state, action = next_state, next_action
            # End episode if done
            if done:
                break
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
    return Q

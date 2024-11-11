#!/usr/bin/env python3
"""
Sarsa(λ) algorithm
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    run 5000 episodes of sarsa(λ) algorithm
    """
    # Initialize eligibility traces, Q is given
    n_states, n_actions = Q.shape
    E = np.zeros((n_states, n_actions))

    for episode in range(episodes):
        """
        reset the environment and sample one episode
        player start upperleft
        Q is given
        """

        E.fill(0)  # Reset eligibility traces
        done = False
        truncated = False

        # initialize state action
        state = env.reset()[0]
        action = get_action(state, Q, epsilon)

        while not done:
            # observing next state and next action
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = get_action(next_state, Q, epsilon)

            # SARSA update
            target = reward + gamma * Q[next_state, next_action]
            actual = Q[state, action]
            delta = target - actual

            # Update eligibility trace for the current state
            # and Q values
            E[state, action] += 1  # Update eligibility
            Q += alpha * delta * E  # update Qvalue
            E *= gamma * lambtha

            state, action = next_state, next_action

        # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q


def get_action(state, Q, epsilon):
    """
    Choose action using epsilon-greedy policy
    """
    n_actions = Q.shape[1]
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state])

#!/usr/bin/env python3
"""performs Q-learning"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy

def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    env: FrozenLakeEnv instance
    Q: numpy.ndarray containing the Q-table
    episodes: number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: learning rate
    gamma: discount rate
    epsilon: initial threshold for epsilon greedy
    min_epsilon: minimum value that epsilon should decay to
    epsilon_decay: decay rate for updating epsilon between episodes
    Returns: Q (updated q-table), total_rewards (list of collected rewards)
    """
    total_rewards = [0] * episodes

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63
        terminated = False  # True when fall in hole or reached goal
        truncated = False  # True when actions > max_steps
        episode_reward = 0  # Track cumulative reward per episode

        while not (truncated or terminated):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)

            # When agent falls in a hole, the reward is updated
            if terminated and reward == 0:
                reward = -1

            # Update Q-table with Q-learning formula
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state, :]) - Q[state, action]
            )

            episode_reward += reward  # Accumulate reward
            state = new_state

        # Decay epsilon after each episode
        epsilon = max(epsilon - epsilon_decay, min_epsilon)

        # Store cumulative reward for the episode
        total_rewards[i] = episode_reward

    env.close()
    return Q, total_rewards


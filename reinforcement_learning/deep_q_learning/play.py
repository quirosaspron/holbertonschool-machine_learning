#!/usr/bin/env python3
"""Have a trained agent play Breakout"""

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Permute
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
from rl.core import Processor


class EnvProc(Processor):
    """Custom processor class for the environment"""

    def process_observation(self, observation):
        return np.array(observation)

    def process_state_batch(self, batch):
        return np.array(batch, dtype=np.float32) / 255.0

    def process_reward(self, reward):
        return np.clip(reward, -1, 1)


class CWrapper(gym.Wrapper):
    """A compatibility wrapper to make Gymnasium environments compatible with keras-rl"""

    def step(self, action):
        obs, r, term, truncated, info = self.env.step(action)
        return obs, r, term or truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)[0]

    def render(self, **kwargs):
        """Render the environment"""
        return self.env.render()


def __set_up_env(envname="ALE/Breakout-v5"):
    """Sets up the environment where the model will perform"""
    env = gym.make(envname, frameskip=1, render_mode="human")
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True,
                             frame_skip=1, noop_max=30)
    return CWrapper(env)


def __build_agent_cnn(iS, nactions):
    """Builds a CNN to process the game frames"""
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=iS))
    model.add(Conv2D(32, (8, 8), strides=4, activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nactions, activation='linear'))
    return model


def __set_up_DQN_agent(ws, nba, cnnmodel, ti):
    """Sets up the DQN agent"""
    memory = SequentialMemory(limit=1000000, window_length=ws)
    policy = GreedyQPolicy()
    dqn = DQNAgent(
        model=cnnmodel,
        nb_actions=nba,
        memory=memory,
        processor=EnvProc(),
        nb_steps_warmup=0,
        target_model_update=10000,
        policy=policy,
        delta_clip=1,
        train_interval=ti,
        gamma=0.99,
    )
    dqn.compile(optimizer='adam', metrics=['mae'])
    return dqn


if __name__ == '__main__':
    # 1. Create environment
    env = __set_up_env()

    # 2. Build CNN model for the agent
    cnnmodel = __build_agent_cnn((4, 84, 84), env.action_space.n)

    # 3. Set up the DQN agent
    dqnagent = __set_up_DQN_agent(4, env.action_space.n, cnnmodel, 4)

    # 4. Load the trained weights into the agent
    dqnagent.load_weights('policy.h5')

    # 5. Have the DQN agent play Breakout
    scores = dqnagent.test(env, nb_episodes=20, visualize=False)

    # 6. Close the environment
    env.close()


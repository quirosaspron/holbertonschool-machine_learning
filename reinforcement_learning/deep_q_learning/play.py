#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, StepAPICompatibility
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from keras.optimizers import Adam
from keras import backend as K
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy

# Set image data format
K.set_image_data_format('channels_first')

# Create the environment with rendering enabled
env = gym.make('ALE/Breakout-v5', render_mode='human')
env = AtariPreprocessing(env)
env = StepAPICompatibility(env, output_truncation_bool=False)

nb_actions = env.action_space.n

# Build the model
input_shape = (4, 84, 84)
model = Sequential()
model.add(Permute((2, 3, 1), input_shape=input_shape))
model.add(Conv2D(32, (8, 8), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Conv2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

# Configure and compile the agent
memory = SequentialMemory(limit=1000000, window_length=4)
policy = GreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
               nb_steps_warmup=0, target_model_update=10000,
               policy=policy, gamma=0.99)
dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

# Load the weights
dqn.load_weights('policy.h5')

# Test the agent
dqn.test(env, nb_episodes=10, visualize=True)


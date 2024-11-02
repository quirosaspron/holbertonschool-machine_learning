#!/usr/bin/env python3
"""loads the pre-made evnironment from gymnasium"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """"
    Loads the FrozenLakeEnv evnironment
    desc: list of lists containing a custom description
          of the map to load for the environment
    map_name: string containing the pre-made map to load
    is_slippery is the ice slippery?
    Returns: the environment
    """

    env = gym.make('FrozenLake-v1',
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)
    return env

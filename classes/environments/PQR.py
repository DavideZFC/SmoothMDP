from os import path
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


class PQR(gym.Env):

    def __init__(self):
        self.time_horizon = 20
        self.h = 0

        self.A = np.array([[.7, .7],[-0.7, .7]])
        self.u = np.array([1., 1.])

        self.action_cost = 0.2
        self.state_cost = 1.

        self.high = np.array([1.0, 1.0], dtype=np.float32)
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=-1., high=1., shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-self.high, high=self.high, dtype=np.float32)

    def step(self, u):

        cost = self.state_cost*np.sum(self.state**2) + self.action_cost*u**2

        self.state = np.dot(self.A, self.state) + self.u*u + np.random.normal(scale=0.1, size=2)
        self.h += 1
        done = self.h > self.time_horizon - 1

        # poincarè attacks
        radius = np.sum(self.state**2)**0.5
        self.state = self.state*(1/(1+radius))

        return self._get_obs(), -cost, False, done, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = np.array([-0.7, -0.7])+np.random.uniform(low=0, high=0.2)*np.array([1., 1.])
        self.h = 0

        return self._get_obs(), {}

    def _get_obs(self):
        return np.copy(self.state)

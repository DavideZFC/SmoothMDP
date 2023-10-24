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
    '''
    environment idea: simulate the linear quadratic regulator in an environment with Poincarè (hyperbolic) geometry.
    '''

    def __init__(self):
        '''
        Initialize the environment
        '''

        self.time_horizon = 20
        self.h = 0

        self.A = np.array([[.7, .7],[-0.7, .7]])
        self.u = np.array([1., 1.])

        self.action_cost = 0.2
        self.state_cost = 1.

        self.high = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1., high=1., shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-self.high, high=self.high, dtype=np.float32)

    def step(self, u):
        '''
        Makes one step in the environment

        Parameters:
            u (vector): action choosen by the agent

        Returns:
            self._get_obs() (vector): copy of the current state      
            -cost (double): reward
            _ : the environment has failed? (always False in this case)
            done (bool): reached time horizon?
            _ : useless information
        '''

        cost = self.state_cost*np.sum(self.state**2) + self.action_cost*u**2

        self.state = np.dot(self.A, self.state) + self.u*u + np.random.normal(scale=0.1, size=2)
        self.h += 1
        done = self.h > self.time_horizon - 1

        # poincarè attacks
        radius = np.sum(self.state**2)**0.5
        self.state = self.state*(1/(1+radius))

        return self._get_obs(), -cost, False, done, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        '''
        Resets the whole environment

        Parameters:
            seed (int): Random seed to use
        
        Returns:
            self._get_obs(): copy of the current state
        '''
        super().reset(seed=seed)
        self.state = np.array([-0.7, -0.7])+np.random.uniform(low=0, high=0.2)*np.array([1., 1.])
        self.h = 0

        return self._get_obs(), {}

    def _get_obs(self):
        '''
        Returns copy of the state
        '''
        return np.copy(self.state)

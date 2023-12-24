from os import path
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


class PendulumSimple(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10.0):
        # modificato da me, prima era limitato a 8, ma ora devo raddoppiare l'effetto
        self.max_speed = 1

        # modificato da me, prima era limitato a 2, ma ora devo raddoppiare l'effetto
        self.max_torque = 1.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.time_horizon = 200
        self.theta_normalization = np.pi
        self.thetadot_normalization = 8
        self.reward_normalization = 10

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([1.0, self.max_speed], dtype=np.float32)
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def original_step(self, th, thdot, u):
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        reward = -costs
        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt

        # modificata da me
        newthdot = np.clip(newthdot, -8*self.max_speed, 8*self.max_speed)
        newth = th + newthdot * dt

        new_state = np.concatenate((angle_normalize(newth).reshape(-1,1), newthdot.reshape(-1,1)), axis=1)
        return new_state, reward


    def step(self, u):
        th, thdot = self.state
        th *= self.theta_normalization
        thdot *= self.thetadot_normalization

        self.h += 1
        done = self.h > self.time_horizon - 1
        u = np.clip(u, -self.max_torque, self.max_torque)[0]

        original_action = 2*u
        new_original_state, original_reward = self.original_step(th, thdot, original_action)

        self.state = np.array([new_original_state[0,0]/self.theta_normalization, new_original_state[0,1]/self.thetadot_normalization])

        return self.state, original_reward, False, done, {}
    
    def query_generator(self, state_action):
        state = state_action[:,:self.observation_space.shape[0]]
        th, thdot = state[:,0], state[:,1]
        th *= self.theta_normalization
        thdot *= self.thetadot_normalization

        u = state_action[:,-1]

        u = np.clip(u, -self.max_torque, self.max_torque)
        original_action = 2*u
        new_original_state, original_reward = self.original_step(th, thdot, original_action)
        
        newth = new_original_state[:,0]/self.theta_normalization
        newthdot = new_original_state[:,1]/self.thetadot_normalization

        return np.concatenate((newth.reshape(-1,1), newthdot.reshape(-1,1)), axis=1), original_reward/self.reward_normalization


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        high = np.array([1,1])
        low = -high  # We enforce symmetric limits.
        # self.state = np.zeros(2)
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None
        self.h = 0

        if self.render_mode == "human":
            self.render()
        return self.state, {}



def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
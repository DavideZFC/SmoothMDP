import gym
from classes.auxiliari.Linear_replay_buffer import Linear_replay_buffer
from classes.agents.LSVI import LSVI
import matplotlib.pyplot as plt
import numpy as np
from classes.environments.Pendulum import Pendulum
from classes.environments.Continuous_MountainCarEnv import Continuous_MountainCarEnv
import time
from functions.misc.test_algorithm import test_algorithm
from functions.misc.make_experiment import make_experiment

# env = gym.make('Pendulum-v1')
env = Pendulum()
# env = Continuous_MountainCarEnv()

# buffer = Linear_replay_buffer(basis='legendre', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=10000)
agent = LSVI(basis='legendre', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=25000, discretize=200, time_horizon=200)

make_experiment([agent], env, seeds=5, K=50, labels=['LebesgueSVI'], exp_name='numero3')
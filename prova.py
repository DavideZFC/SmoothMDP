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
agent = LSVI(basis='legendre', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=10000, discretize=200, time_horizon=200)

make_experiment([agent], env, seeds=5, K=2, labels=['LebesgueSVI'], exp_name='provascema')

reward_matrix = test_algorithm(agent, env, seeds=2, K=5)
print(reward_matrix.shape)

K = 0
H = 200
rewards = np.zeros(H*K)
rew_index = 0

t = time.time()

for k in range(K):
    state = env.reset()[0]

    done = False
    h = 0

    while not done:
        # action = env.action_space.sample()
        action = agent.choose_action(state, h)

        next_state, reward, terminated, truncated, _ = env.step(action)

        rewards[rew_index] = reward
        rew_index += 1

        done = terminated or truncated

        agent.replay_buffer.memorize(state, action, next_state, reward)
        state = next_state
        h += 1


    agent.compute_q_values()

    print('Episode {} finished, mean reward is {}, reward of last episode is {}'.format(k, np.mean(rewards[:rew_index]), np.mean(rewards[rew_index-H:rew_index])))

print('time elapsed = {}'.format(time.time()-t))

plt.plot(rewards)
plt.show()
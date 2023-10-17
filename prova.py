import gym
from classes.auxiliari.Linear_replay_buffer import Linear_replay_buffer
from classes.agents.LSVI import LSVI
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('Pendulum-v1')

buffer = Linear_replay_buffer(basis='poly', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=10000)
agent = LSVI(buffer, time_horizon=200)



K = 40
rewards = np.zeros(200*K)
rew_index = 0

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

    print('Episode {} finished, mean reward is {}'.format(k, np.mean(rewards[:rew_index])))

plt.plot(rewards)
plt.show()
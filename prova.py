import gym
from classes.auxiliari.Linear_replay_buffer import Linear_replay_buffer
from classes.agents.LSVI import LSVI

env = gym.make('Pendulum-v1')

buffer = Linear_replay_buffer(basis='poly', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=10000)
agent = LSVI(buffer, time_horizon=200)

state = env.reset()[0]
print(state)
done = False
while not done:
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    agent.replay_buffer.memorize(state, action, next_state, reward)
    state = next_state

agent.compute_q_values()
print(agent.w_vectors)
import gym
from classes.auxiliari.Replay_buffer import Replay_buffer

env = gym.make('Pendulum-v1')

buffer = Replay_buffer(state_space_dim=env.observation_space.shape[0], action_space_dim=env.action_space.shape[0], numel=10000)

state = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    buffer.memorize(state[0], action, next_state[0], reward)
    state = next_state

print(buffer.state_action_buffer)
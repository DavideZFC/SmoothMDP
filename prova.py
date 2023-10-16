import gym
from classes.auxiliari.Linear_replay_buffer import Linear_replay_buffer

env = gym.make('Pendulum-v1')

buffer = Linear_replay_buffer(basis='poly', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space_dim=env.action_space.shape[0], numel=10000)

state = env.reset()[0]
print(state)
done = False
while not done:
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    buffer.memorize(state, action, next_state, reward)
    state = next_state

buffer.linear_converter()
print(buffer.full_feature_map.shape)
buffer.build_action_mesh(env.action_space)
# print(buffer.next_state_buffer[0,:])
S = buffer.build_next_state_action_feature_map()
print(S.shape)

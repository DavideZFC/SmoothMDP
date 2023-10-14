import numpy as np

class Replay_buffer:

    def __init__(self, state_space_dim, action_space_dim, numel):
        self.state_space_dim = state_space_dim

        self.state_action_buffer = np.zeros((numel, state_space_dim + action_space_dim))
        self.next_state_buffer = np.zeros((numel, state_space_dim))
        self.reward_buffer = np.zeros(numel)

        self.current_index = 0

    def memorize(self, state, action, next_state, reward):

        self.state_action_buffer[self.current_index, :self.state_space_dim] = state
        self.state_action_buffer[self.current_index, self.state_space_dim:] = action
        self.next_state_buffer[self.current_index, :self.state_space_dim] = next_state
        self.reward_buffer[self.current_index] = reward

        self.current_index += 1

import numpy as np
from functions.misc.merge_feature_maps import merge_feature_maps
from functions.orthogonal.bases import *

class Linear_replay_buffer:
    # problema, l'ortogonalità se ne vaffanculo

    def __init__(self, basis, approx_degree, state_space_dim, action_space_dim, numel):

        if basis == 'poly':
            self.feature_map = poly_features
        elif basis == 'cosin':
            self.feature_map = cosin_features
        elif basis == 'sincos':
            self.feature_map = sincos_features
        elif basis == 'legendre':
            self.feature_map = legendre_features

        self.approx_degree = approx_degree

        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim

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

    def linear_converter(self):

        feature_maps = []
        for i in range(self.state_space_dim+self.action_space_dim):
            feature_maps.append(self.feature_map(self.state_action_buffer[:self.current_index, i], d=self.approx_degree))

        self.full_feature_map = merge_feature_maps(feature_maps)

    def build_action_mesh(self, action_space, discretize=200):
        
        dim = action_space.shape[0]

        # this only works for action space dim = 1!!
        self.action_grid = np.linspace(action_space.low[0], action_space.high[0], discretize)

        self.action_features = self.feature_map(self.action_grid, d=self.approx_degree)
        


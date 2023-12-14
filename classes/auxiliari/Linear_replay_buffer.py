import numpy as np
from functions.misc.merge_feature_maps import merge_feature_maps
from functions.misc.build_mesh import build_mesh
from functions.orthogonal.bases import *

class Linear_replay_buffer:

    def __init__(self, basis, approx_degree, state_space_dim, action_space, numel, discretize=200):
        '''
        Initializes the algorithm.

        Parameters:
            basis (str): name of the basis function used for the buffer
            approx_degree (int): maximum degree of the polynomials/trigonometric functions used
            state_space_dim (int): length of the state vector
            action_space (gym.ActionSpace): action space of the environment
            numel (int): size of the buffer 
        '''

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
        self.action_space_dim = action_space.shape[0]
        self.action_space = action_space
        self.build_action_mesh(discretize=discretize)

        self.state_action_buffer = np.zeros((numel, state_space_dim + action_space.shape[0]))
        self.next_state_buffer = np.zeros((numel, state_space_dim))
        self.reward_buffer = np.zeros(numel)

        self.current_index = 0

    def reset(self):
        self.current_index = 0
      



    def memorize(self, state, action, next_state, reward):
        '''
        Store one transition of the environment into the buffer

        Parameters:
            state (vector): observed state
            action (vector): action performed
            next_state (vector): state after transition
            reward (double): reward received
        '''

        self.state_action_buffer[self.current_index, :self.state_space_dim] = state
        self.state_action_buffer[self.current_index, self.state_space_dim:] = action
        self.next_state_buffer[self.current_index, :self.state_space_dim] = next_state
        self.reward_buffer[self.current_index] = reward

        self.current_index += 1



    def linear_converter(self):
        '''
        Converts the state_action buffer into the corresponding feature map representation
        '''

        feature_maps = []
        for i in range(self.state_space_dim+self.action_space_dim):
            feature_maps.append(self.feature_map(self.state_action_buffer[:self.current_index, i], d=self.approx_degree))

        self.full_feature_map = merge_feature_maps(feature_maps)
        self.full_feature_map /= (self.full_feature_map.shape[1]**0.5)




    def build_action_mesh(self, discretize=200):
        '''
        Discretizes the action space and converts it according to the feature map

        Parameters:
            discretize (int): how many elements put in the discretization
        '''
        
        dim = self.action_space.shape[0]

        # this only works for action space dim = 1!!
        self.action_grid = np.linspace(self.action_space.low[0], self.action_space.high[0], discretize)

        self.action_features = self.feature_map(self.action_grid, d=self.approx_degree)



    def build_next_state_action_mesh(self, state):
        '''
        Given one state, returns a matrix where the state is repreated in the first 
        columns and in the last we have the discretization of the state space

        Parameters:
            state (vector): state

        Returns:
            _ (array): matrix made in this way
        '''

        state_repeated = np.repeat(state[np.newaxis, :], self.action_grid.shape[0], axis=0)

        return np.append(state_repeated, self.action_grid.reshape(-1,1), axis=1)
    



    def build_next_state_action_feature_map(self, state=0):
        '''
        Calls the function build_next_state_action_mesh() and then converts the result according to the
        feature map

        Parameters:
            state (vector): state

        Returns:
            _ (array): feature map applies to the array       
        '''

        aux_buffer = self.build_next_state_action_mesh(state=state)

        feature_maps = []
        for i in range(self.state_space_dim+self.action_space_dim):
            feature_maps.append(self.feature_map(aux_buffer[:, i], d=self.approx_degree))

        return merge_feature_maps(feature_maps)



    def compute_covariance_matrix(self):
        '''
        Computes the covariance matrix (= design matrix) of the feature transformation of the replay buffer

        Retruns:
            _ (array): covariance matrix
        '''
        self.linear_converter()
        return np.dot(self.full_feature_map.T,self.full_feature_map)



        


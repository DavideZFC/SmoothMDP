import numpy as np
from classes.auxiliari.Linear_replay_buffer import Linear_replay_buffer

class Dummy:

    def __init__(self, basis, approx_degree, state_space_dim, action_space, numel, discretize, v_vector=None):
        '''
        Initialize the algorithm

        Parameters:
            replay_buffer (class Linear_replay_buffer): where to store the information
            time horizon (int): time horizon for the problem
            lam (double): lambda parameter for LSVI
            beta (double): beta parameter for LSVI
        '''
        
        self.replay_buffer = Linear_replay_buffer(basis, approx_degree, state_space_dim, action_space, numel, discretize)
        self.replay_buffer.linear_converter()
        self.dim = self.replay_buffer.full_feature_map.shape[1]
        if v_vector is None:
            v_vector = np.random.uniform(size=self.dim)
        
        self.v_vector = v_vector

    def reset(self):
        pass

    def memorize(self, state, action, next_state, reward):
        pass

    def compute_q_values(self):
        pass

 
    def choose_action(self, state, h):
        '''
        Chooses which action to perform based on the current state

        Parameters:
            state (vector): current state
            h (int): current timestep
        
        Returns:
            _ (vector): action to be performed on the environment
        '''

        variable_action_mesh = self.replay_buffer.build_next_state_action_feature_map(state)

        best_action_index = np.argmax(np.dot(variable_action_mesh, self.v_vector))

        return np.array([self.replay_buffer.action_grid[best_action_index]])


        

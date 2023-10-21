import numpy as np
from classes.auxiliari.Linear_replay_buffer import Linear_replay_buffer

class LSVI:

    def __init__(self, basis, approx_degree, state_space_dim, action_space, numel, discretize, time_horizon, lam=1, beta=10):
        '''
        Initialize the algorithm

        Parameters:
            replay_buffer (class Linear_replay_buffer): where to store the information
            time horizon (int): time horizon for the problem
            lam (double): lambda parameter for LSVI
            beta (double): beta parameter for LSVI
        '''
        
        self.replay_buffer = Linear_replay_buffer(basis, approx_degree, state_space_dim, action_space, numel, discretize)
        self.time_horizon = time_horizon

        self.replay_buffer.linear_converter()
        self.dim = self.replay_buffer.full_feature_map.shape[1]

        self.w_vectors = np.zeros((time_horizon, self.dim))
        self.lam = lam
        self.beta = beta

        self.covariance_matrix = self.lam*np.identity(self.dim)
        self.anticov = np.linalg.inv(self.covariance_matrix)

    def compute_optimal_beta(self, K, c=1, miss=0.01):
        prob = 0.01
        iota = np.log(2*self.dim*(K*self.time_horizon)/prob)
        self.beta = c*(self.dim*iota**0.5+miss*np.sqrt(K*self.dim))*self.time_horizon
        print(self.beta)

    def memorize(self, state, action, next_state, reward):
        self.replay_buffer.memorize(state, action, next_state, reward)

    def reset(self):
        '''
        Come back to the original settings
        '''
        self.replay_buffer.reset()

        self.w_vectors *= 0

        self.covariance_matrix = self.lam*np.identity(self.dim)
        self.anticov = np.linalg.inv(self.covariance_matrix)


    def update_buffer(self):
        '''
        Tells the replay buffer recompute its features
        '''
        self.replay_buffer.linear_converter()


    def compute_q_values(self):
        '''
        Computes the vectors w for every timestep
        '''
        self.covariance_matrix = self.replay_buffer.compute_covariance_matrix() + self.lam*np.identity(self.dim)
        self.anticov = np.linalg.inv(self.covariance_matrix)

        self.w_vectors[self.time_horizon - 1] = self.compute_w_step(next_w=0, last_step=True)
        for h in range(2,self.time_horizon+1):
            self.w_vectors[self.time_horizon - h] = self.compute_w_step(next_w=self.w_vectors[self.time_horizon - h + 1], last_step=False)


    def get_best_future_q(self, state=0, next_w=None):
        '''
        Returns the optimal state value function estimated for a given state

        Parameters:
            state (vector): state in which I want to measure the optimal state value function
            next_w (vector): vector of the coefficients of the Q function for the next iteration

        Returns:
            _ (double): optimal state value function estimated
        '''
        variable_action_mesh = self.replay_buffer.build_next_state_action_feature_map(state)

        # anticov = np.linalg.inv(self.covariance_matrix)
        bonus_vector = np.zeros(variable_action_mesh.shape[0])
        for i in range(variable_action_mesh.shape[0]):
            bonus_vector[i] = np.dot(np.dot(variable_action_mesh[i,:].T, self.anticov), variable_action_mesh[i,:])**0.5

        return np.max(np.dot(variable_action_mesh,next_w) + self.beta*bonus_vector)
    

    def compute_w_step(self, next_w, last_step=False):
        '''
        Computes the w parameter corresponding to this step

        Parameters:
            next_w (vector): w parameter of the next step
            last step (bool): says if this is the last step (in such case, next_w is useless)

        Returns:
            _ (vector): the w vector
        '''
        load = np.zeros(self.dim)
        if last_step:
            for i in range(self.replay_buffer.current_index):
                load += self.replay_buffer.full_feature_map[i]*(self.replay_buffer.reward_buffer[i])            
        else:            
            for i in range(self.replay_buffer.current_index):
                load += self.replay_buffer.full_feature_map[i]*(self.replay_buffer.reward_buffer[i] + self.get_best_future_q(self.replay_buffer.next_state_buffer[i], next_w))

        return np.linalg.solve(self.covariance_matrix, load)
    
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

        anticov = np.linalg.inv(self.covariance_matrix)
        bonus_vector = np.zeros(variable_action_mesh.shape[0])
        for i in range(variable_action_mesh.shape[0]):
            bonus_vector[i] = np.dot(np.dot(variable_action_mesh[i,:].T, anticov), variable_action_mesh[i,:])**0.5

        best_action_index = np.argmax(np.dot(variable_action_mesh, self.w_vectors[h]) + self.beta*bonus_vector)

        return np.array([self.replay_buffer.action_grid[best_action_index]])


        

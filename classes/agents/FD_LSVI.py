import numpy as np
from functions.misc.merge_feature_maps import merge_feature_maps
from functions.misc.build_mesh import build_mesh, generalized_build_mesh
from functions.orthogonal.bases import *
from functions.orthogonal.find_optimal_design import find_optimal_design


def get_dataset(env, disc_numbers, approx_degree = 4, feature_map = legendre_features, action_space_dim=1):
    # dimension of the state-action space
    d = len(disc_numbers)

    coord_mapping_list = []
    for disc in disc_numbers:
        coord_mapping_list.append(np.linspace(-1,1,disc))

    points_list = generalized_build_mesh(coord_mapping_list)
    points_vector = np.array(points_list).T


    ### part relative to the action
    action_list = []
    for i in range(len(disc_numbers)-action_space_dim, len(disc_numbers)):
        action_list.append(np.linspace(-1,1,disc_numbers[i]))

    action_mesh = generalized_build_mesh(action_list)
    action_grid = np.array(action_mesh).T
    #####  

    feature_maps = []
    for i in range(d):
        feature_maps.append(feature_map(points_list[i], d=approx_degree))

    full_feature_map = merge_feature_maps(feature_maps)
    full_feature_map /= (full_feature_map.shape[1]**0.5)

    pi = find_optimal_design(full_feature_map)
    nonzero = np.sum(pi > 0)

    big_constant = 100
    upper_bound = 2*nonzero + big_constant
    fixed_design = np.zeros(int(upper_bound), dtype=int)
    current_index = 0
    for i in range(full_feature_map.shape[0]):
        if pi[i] > 0:
            times_to_pull = 1 + int(pi[i]*big_constant)
            fixed_design[current_index:current_index+times_to_pull] = i
            current_index += times_to_pull
    fixed_design = fixed_design[:current_index]

    query_features = np.zeros((current_index, full_feature_map.shape[1]))
    query_points = np.zeros((current_index, 3))
    for i in range(current_index):
        query_features[i, :] = full_feature_map[fixed_design[i], :]
        query_points[i, :] = points_vector[fixed_design[i], :]

    new_states, rewards = env.query_generator(query_points)

    return query_points, new_states, rewards, query_features, action_grid



class FD_LSVI:
    def __init__(self, env):
        self.env = env
        self.time_horizon = env.time_horizon

    def get_datasets(self, disc_numbers, approx_degree = 4, feature_map = legendre_features):
        # dimension of the state-action space
        self.feature_map = feature_map
        self.d = len(disc_numbers)
        self.query_points, self.new_states, self.rewards, self.query_features, self.action_grid = get_dataset(self.env, disc_numbers, approx_degree, feature_map)
        # how many points we have
        self.N = self.query_points.shape[0]
        self.dim = self.query_features.shape[1]
        self.w_vectors = np.zeros((self.time_horizon, self.dim))
        self.approx_degree = approx_degree

        self.covariance_matrix = np.dot(self.query_features.T,self.query_features)
        self.anticov = np.linalg.inv(self.covariance_matrix)

        print(self.query_points.shape)
        print(self.query_features.shape)
        print(self.rewards.shape)
        print(self.new_states.shape)
        print(self.action_grid.shape)

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
            for i in range(self.N):
                load += self.query_features[i]*(self.rewards[i])            
        else:            
            for i in range(self.N):
                load += self.query_features[i]*(self.rewards[i] + self.get_best_future_q(self.new_states[i], next_w))

        return np.linalg.solve(self.covariance_matrix, load)
    
    def compute_q_values(self):
            '''
            Computes the vectors w for every timestep
            '''
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
        variable_action_mesh = self.build_next_state_action_feature_map(state)
        return np.max(np.dot(variable_action_mesh,next_w))
    
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
        for i in range(self.d):
            feature_maps.append(self.feature_map(aux_buffer[:, i], d=self.approx_degree))

        return merge_feature_maps(feature_maps)



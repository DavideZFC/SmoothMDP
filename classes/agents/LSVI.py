import numpy as np

class LSVI:

    def __init__(self, replay_buffer, time_horizon, lam=1):
        
        self.replay_buffer = replay_buffer
        self.time_horizon = time_horizon

        self.replay_buffer.linear_converter()
        self.dim = self.replay_buffer.full_feature_map.shape[1]

        self.w_vectors = np.zeros((time_horizon, self.dim))
        self.lam = lam

    def update_buffer(self):
        self.replay_buffer.linear_converter()


    def compute_q_values(self):
        self.w_vectors[self.time_horizon - 1] = self.compute_w_step(next_w=0, precomputed=False, covariance_matrix=0, last_step=True)
        for h in range(2,self.time_horizon+1):
            self.w_vectors[self.time_horizon - h] = self.compute_w_step(next_w=self.w_vectors[self.time_horizon - h + 1], precomputed=False, covariance_matrix=0, last_step=True)


    def get_best_future_q(self, state=0, next_w=0):
        variable_action_mesh = self.replay_buffer.build_next_state_action_feature_map(state)

        if next_w == 0:
            next_w = np.ones(self.dim)
        return np.max(np.dot(variable_action_mesh,next_w))


    def compute_w_step(self, next_w, precomputed=False, covariance_matrix=0, last_step=False):

        if not precomputed:
            covariance_matrix = self.replay_buffer.compute_covariance_matrix() + np.identity(self.dim)
        
        load = np.zeros(self.dim)
        if last_step:
            for i in range(self.replay_buffer.current_index):
                load += self.replay_buffer.full_feature_map[i]*(self.replay_buffer.reward_buffer[i])            
        else:            
            for i in range(self.replay_buffer.current_index):
                load += self.replay_buffer.full_feature_map[i]*(self.replay_buffer.reward_buffer[i] + self.get_best_future_q(self.replay_buffer.next_state_buffer[i], next_w))

        return np.linalg.solve(covariance_matrix, load)

        

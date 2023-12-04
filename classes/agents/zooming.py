import numpy as np
from functions.misc.build_mesh import build_mesh

class ZOOM():
    '''
    ZOOMING algorithm, see
    'Multi-Armed Bandits in Metric Spaces'
    '''

    def __init__(self, T=1000, warmup=2):
        '''
        arms = arms of the environment
        iph = exploration parameter corresponding to the order of the time horizon (see the article)
        warmup = inital random arms pulled
        '''
        self.build_arms(numel=int(T**0.5))
        self.warmup = warmup
        self.active_arms_idx = []
        self.mu = []
        self.times_pulled = []
        self.t = 0
        self.iph = int(np.log2(T))

    def build_arms(self, numel=100):
        '''
        discretize arm space

        Parameters:
            numel (int): in how many points to discretize the space in every dimension
        '''
        # build mesh for the space
        x = np.linspace(-1,1,numel)
        y = np.linspace(-1,1,numel)
        self.X,self.Y = build_mesh(x,y)
        self.N = len(self.X)
        self.arms_matrix = np.column_stack((self.X, self.Y))

    def covered_point(self, p):
        for j in range(len(self.active_arms_idx)):
            radius = (8*self.iph/(2+self.times_pulled[j]))**0.5
            arm = self.arms_matrix[self.active_arms_idx[j], :]
            if(np.linalg.norm(arm-p)<radius):
                return True
        return False

    def covering_oracle(self):
        for i in range(self.N):
            if (not self.covered_point(self.arms_matrix[i])):
                return self.arms_matrix[i], i, False
        return 0, 0, True

    def pull_arm(self):
        if (self.t < self.warmup):
            idx = np.random.randint(self.N)
            self.active_arms_idx.append(idx)
            self.times_pulled.append(0)
            self.mu.append(0)
            self.last_arm_pulled = len(self.active_arms_idx)-1

            return self.arms_matrix[idx,:]

        arm, idx, guess = self.covering_oracle()
        if not guess:
            self.active_arms_idx.append(idx)
            self.times_pulled.append(0)
            self.mu.append(0)
            # print('added arm {} with {}'.format(idx, arm))

        current_best = -1000
        for j in range(len(self.active_arms_idx)):
            radius = (8*self.iph/(2+self.times_pulled[j]))**0.5
            if (radius + self.mu[j] > current_best):
                current_best = radius + self.mu[j]
                best_idx = j

        self.last_arm_pulled = best_idx
        # print('pulled arm {} with {} previously pulled {} times'.format(self.active_arms_idx[best_idx], self.arms_matrix[self.active_arms_idx[best_idx],:], self.times_pulled[best_idx]))
        return self.arms_matrix[self.active_arms_idx[best_idx],:]

    def update(self, reward):
        self.t += 1
        self.times_pulled[self.last_arm_pulled] += 1
        self.mu[self.last_arm_pulled] = ((self.times_pulled[self.last_arm_pulled]-1)*self.mu[self.last_arm_pulled] + reward)/(self.times_pulled[self.last_arm_pulled])
        

    def reset(self):
        self.active_arms_idx = []
        self.mu = []
        self.times_pulled = []
        self.t = 0
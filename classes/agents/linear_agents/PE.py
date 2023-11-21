import numpy as np



def compute_induced_norm(Ainv, v):
    results = np.zeros(v.shape[0])
    for i in range(v.shape[0]):
        results[i] = np.dot(v[i,:].T, np.dot(Ainv, v[i,:]))
    return results

def compute_design_matrix(A, pi):
    D = np.zeros((A.shape[1],A.shape[1]))

    for i in range(A.shape[0]):
        D += pi[i]*np.dot(A[i:i+1,:].T,A[i:i+1,:])
    return D

def squeeze_distribution(pi, n):
    # apply noise injection to avoid ties
    pi = pi + np.random.normal(0,scale=1e-4,size=len(pi))

    sorted_vals = sorted(pi, reverse=True)
    nth_largest = sorted_vals[min(n, len(sorted_vals))-1]
    pi[pi<nth_largest] = 0
    pi = pi/np.sum(pi)
    return pi

def onehot(idx, k):
    v = np.zeros(k)
    v[idx] = 1
    return v

def eval_pi(pi, A):
    D = compute_design_matrix(A, pi)
    Dinv = np.linalg.inv(D)
    v = compute_induced_norm(Dinv, A)
    return np.max(v)


def find_optimal_design(A, iter=100, thresh=0):
    k = A.shape[0]
    d = A.shape[1]
    pi = np.ones(k)/k

    for it in range(iter):
        D = compute_design_matrix(A, pi)
        Dinv = np.linalg.inv(D)
        v = compute_induced_norm(Dinv, A)

        best_index = np.argmax(v)
        current = v[best_index]
        if current < (thresh + 1)*A.shape[1]:
            break
        gamma = (current/d-1)/(current-1)

        pi = (1-gamma)*pi + gamma*onehot(best_index, k)
    pi = squeeze_distribution(pi, 2*A.shape[1])
    if eval_pi(pi, A) > 2*d:
        print('Error we are fucked')
    return pi




class dummy_learner:
    def __init__(self, arms, epsilon, delta):
        self.arms = arms
        self.n_arms = arms.shape[0]
        self.d = arms.shape[1]
        pi = self.compute_optimal_design(arms)
        term2 = np.log(2*self.n_arms/delta)

        # self.fixed_design = []
        upper_bound = 2*self.d /(epsilon**2)*term2 + 2*self.d
        self.fixed_design = np.zeros(int(upper_bound), dtype=int)
        current_index = 0
        for i in range(self.n_arms):
            term1 = 2*pi[i]*self.d / (epsilon**2)
            times_to_pull = int(0.25*term1*term2)
            if times_to_pull > 0:
                # print('term 1 = {}, eps = {}, pi = {}, d = {}'.format(term1, epsilon, pi[i], self.d))
                # print('term 2 = {}'.format(term2))
                # self.fixed_design += [i]*times_to_pull
                self.fixed_design[current_index:current_index+times_to_pull] = i
                current_index += times_to_pull
        self.fixed_design = self.fixed_design[:current_index]
        self.idx = 0
        upper_bound = 2*self.d /(epsilon**2)*term2 + 2*self.d
        # print('fixed design of length {} found. eps = {}, upper bound = {}'.format(len(self.fixed_design), epsilon, upper_bound))

        self.design_matrix = np.zeros((self.d, self.d))
        self.load = np.zeros(self.d).reshape(-1,1)

    def update(self, arm, reward):
        self.design_matrix += np.matmul(arm.reshape(-1,1),arm.reshape(1,-1))
        self.load += reward*arm.reshape(-1,1) 

    def get_theta(self):
        return np.squeeze(np.linalg.solve(self.design_matrix, self.load))

    def check_status(self):
        return self.idx < len(self.fixed_design) 

    def pull_arm(self):
        self.idx += 1
        arm_idx = self.fixed_design[self.idx-1]
        return self.arms[arm_idx, :], arm_idx
            
    def compute_optimal_design(self, A):
        if self.n_arms < self.d:
            return np.ones(self.n_arms)/self.n_arms
        return find_optimal_design(A)




class PE:
    def __init__(self, arms_matrix, T=10000.0, base_epsilon=0.3):

        # dimension of the arms
        self.d = arms_matrix.shape[1]

        # matrix of the arms
        self.arms = arms_matrix
        self.n_arms = self.arms.shape[0]
        # time horizon
        self.T = T
        self.t = 0

        # error probability
        self.delta = T**(-1/2)

        # epsilon value
        self.epsilon = base_epsilon

        # initialize_learner
        self.active_arms = np.full(self.n_arms, True, dtype=bool)
        self.index_converter = np.where(self.active_arms)[0]
        self.learner = dummy_learner(arms_matrix, self.epsilon, self.delta)

    def reset(self):
        self.t = 0
        self.epsilon = 1.0
        self.active_arms = np.full(self.n_arms, True, dtype=bool)
        self.index_converter = np.where(self.active_arms)[0]

        # initialize_learner
        self.learner = dummy_learner(self.arms, self.epsilon, self.delta)

    def compute_active_arms(self):
        theta = self.learner.get_theta()
        self.scalar_vector = np.zeros(self.n_arms)

        for i in range(self.n_arms):
            if self.active_arms[i]:
                self.scalar_vector[i] = np.dot(theta, self.arms[i,:])
            else:
                # this condition prevents non-active arms from resurrecting
                self.scalar_vector[i] = -10000

        best = np.max(self.scalar_vector)
        self.active_arms = self.scalar_vector > best - 2*self.epsilon
        self.index_converter = np.where(self.active_arms)[0]

    def update(self, arm, reward):
        self.learner.update(arm, reward)

    def convert_index(self, idx):
        return self.index_converter[idx]

    def pull_arm(self):
        if np.sum(self.active_arms) < self.d:
            return np.argmax(self.scalar_vector)
        if self.learner.check_status():
            _, arm_idx = self.learner.pull_arm()
            return self.convert_index(arm_idx)
        else:
            self.compute_active_arms()
            print('{} arms remain valid when eps = {}'.format(np.sum(self.active_arms), self.epsilon))
            self.epsilon /= 2
            self.learner = dummy_learner(self.arms[self.active_arms], self.epsilon, self.delta)
            _, arm_idx = self.learner.pull_arm()
            return self.convert_index(arm_idx)






    


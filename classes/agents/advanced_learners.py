from sklearn.gaussian_process import GaussianProcessRegressor
from functions.misc.build_mesh import build_mesh
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, ConstantKernel
import numpy as np

class Gauss_Bandit():
    '''
    a.k.a. Gaussian UCB1 is very good in practice but has not the same theoretical guarantees of the others, 
    and so it is widely not considered a valid baseline
    '''
    def __init__(self, arms, update_every=50):
        '''
        arms = arms of the environment
        '''
        self.arms = arms
        self.N = len(arms)
        self.eval_x = []
        self.eval_y = []
        self.gp = GaussianProcessRegressor(normalize_y=True)
        self.step = 0
        self.update_every = update_every

    def pull_arm(self):
        self.step += 1
        if self.step < self.N:
            return self.step
        else:
            mean, std = self.gp.predict(self.arms.reshape(-1,1), return_std = True)
            # here we have to add some coefficient
            return np.argmax(mean + std)

    def update(self, arm, reward):
        self.eval_x.append(self.arms[arm])
        self.eval_y.append(reward)
        if (self.step > 10 and self.step % self.update_every == self.update_every-1):
            self.gp.fit(np.array(self.eval_x).reshape(-1, 1), np.array(self.eval_y))

    def reset(self):     
        self.eval_x = []
        self.eval_y = []
        self.gp = GaussianProcessRegressor(normalize_y=True)
        self.step = 0


class IGP_UCB():
    '''
    From the article "On Kernelized Multi-armed Bandits"
    '''
    def __init__(self, dim=2, T=10000, B=10, R=1, update_every=1, warmup=10):
        '''
        arms = arms of the environment
        '''
        self.eval_x = []
        self.eval_y = []
        self.gp = GaussianProcessRegressor(normalize_y=True)
        self.step = 0
        self.update_every = update_every
        self.T = T
        self.delta = 1/T
        self.B = B
        self.R = R
        self.warmup_steps = warmup

        self.dim = dim
        if dim>2:
            raise NotImplementedError

        self.build_arms(numel = int(T**0.5))

    def build_arms(self, numel=100):
        '''
        Build the discretization of the arms

        Parameters:
            numel (int): in how many points to discretize the space in every dimension
        '''
        # build mesh for the space
        x = np.linspace(-1,1,numel)
        y = np.linspace(-1,1,numel)
        self.X,self.Y = build_mesh(x,y)
        self.N = len(self.X)
        self.arms_matrix = np.column_stack((self.X, self.Y))

    def pull_arm(self):
        self.step += 1
        if self.step < self.warmup_steps:
            self.last_arm_idx = np.random.randint(self.N)
            return self.arms_matrix[self.last_arm_idx,:]
        else:
            mean, std = self.gp.predict(self.arms_matrix, return_std = True)
            # print('step {} maximal std: {} minimal std: {}'.format(self.step ,np.max(std), np.min(std)))
            gamma = np.log(self.step)**(self.dim+1)
            beta = self.B + self.R*np.sqrt(2*gamma + 1 + np.log(1/self.delta))
            self.last_arm_idx = np.argmax(mean + beta*std)
            return self.arms_matrix[self.last_arm_idx,:]

    def update(self, reward):
        self.eval_x.append(self.arms_matrix[self.last_arm_idx,:])
        self.eval_y.append(reward)
        if (self.step > 10 and self.step % self.update_every == self.update_every-1):
            self.gp.fit(np.array(self.eval_x), np.array(self.eval_y))


    def reset(self):     
        self.eval_x = []
        self.eval_y = []
        self.gp = GaussianProcessRegressor(normalize_y=True)
        self.step = 0

class BPE():
    '''
    From the article "Gaussian Process Bandit Optimization with Few Batches"
    '''
    def __init__(self, arms, T=10000, Psi=10, R=1, lam=1):
        '''
        arms = arms of the environment
        '''
        self.arms = arms
        self.active_arms = np.ones_like(arms)
        self.N = len(arms)
        self.eval_x = []
        self.eval_y = []
        self.gp = GaussianProcessRegressor(normalize_y=True)#, kernel = WhiteKernel() + ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed"))
        self.step = 0
        self.T = T
        self.delta = 1/T
        self.Psi = Psi
        self.R = R
        self.lam = lam
        self.next_lim = int(np.sqrt(T))

    def pull_arm(self):
        # we throw away the mean and maximize the variance
        _, std = self.gp.predict(self.arms.reshape(-1,1), return_std = True)
        # print('step {} maximal std: {} minimal std: {} number of active arms: {}'.format(self.step ,np.max(std), np.min(std), np.sum(np.sum(self.active_arms))))
        
        # we put to zero the variance of arms that are not active
        std = std*self.active_arms

        return np.argmax(std)

    def update(self, arm, reward):
        self.eval_x.append(self.arms[arm])
        self.eval_y.append(reward)

        self.step += 1

        if self.step == self.next_lim:
            self.gp.fit(np.array(self.eval_x).reshape(-1, 1), np.array(self.eval_y))
            self.next_lim = int(np.sqrt(self.next_lim*self.T))

            # compute UB and LB
            mean, std = self.gp.predict(self.arms.reshape(-1,1), return_std = True)
            beta = self.Psi + self.R/np.sqrt(self.lam)*np.sqrt(2*np.log(self.N*self.next_lim/self.delta))
            UBvector = mean + beta*std
            LB = np.max((mean - beta*std)*self.active_arms)

            # update set of active arms
            self.active_arms *= np.where(UBvector > LB, 1, 0)
            # print('at step {} there are {} active arms'.format(self.step, np.sum(self.active_arms)))


    
    def reset(self):  
        self.active_arms = np.ones_like(self.arms)  
        self.eval_x = []
        self.eval_y = []
        self.gp = GaussianProcessRegressor(normalize_y=True)
        self.step = 0

class GPTS():
    '''
    Gaussian Thompson Sampling is very good in practice but has not the same theoretical guarantees of the others, 
    and so it is widely not considered a valid baseline
    '''
    
    def __init__(self, arms, update_every=50):
        '''
        arms = arms of the environment
        update_every = how frequently to change the distribution of the gp process (divides the computational time without loss in performance)
        '''
        self.arms = arms
        self.N = len(arms)
        self.eval_x = []
        self.eval_y = []
        self.gp = GaussianProcessRegressor(normalize_y=True)
        self.step = 0
        self.means = np.zeros(self.N)
        self.stds = np.zeros(self.N)
        self.update_every = update_every

    def pull_arm(self):
        self.step += 1
        if (self.step % 1000 == 0):
            print('GPTS siamo allo step = '+str(self.step))
        if self.step < self.N:
            return self.step
        else:
            if(self.step % self.update_every == 0):
                self.means, self.stds = self.gp.predict(self.arms.reshape(-1,1), return_std = True)
            samples = self.stds*np.random.randn(self.N)+self.means
            return np.argmax(samples)

    def update(self, arm, reward):
        self.eval_x.append(self.arms[arm])
        self.eval_y.append(reward)
        if (self.step > 10 and self.step % self.update_every==self.update_every-1):
            self.gp.fit(np.array(self.eval_x).reshape(-1, 1), np.array(self.eval_y))

    def reset(self):     
        self.eval_x = []
        self.eval_y = []
        self.gp = GaussianProcessRegressor(normalize_y=True)
        self.step = 0
        self.means = np.zeros(self.N)
        self.stds = np.zeros(self.N)


    


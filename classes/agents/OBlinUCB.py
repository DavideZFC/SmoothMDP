from classes.agents.linear_agents.linUCB import linUBC
from classes.agents.linear_agents.PE import PE
from functions.misc.build_mesh import build_mesh
from functions.misc.merge_feature_maps import merge_feature_maps
from functions.orthogonal.bases import *
import numpy as np


class OBlinUCB:
    def __init__(self, basis, N, dim=2, lam=1, T=10000, pe=False):
        '''
        Instanciate the OBlinUCB and the OB-PE algorithms

        Parameters:
            basis (str): which basis function to use
            N (int): degree of the basis function
            dim (int): dimension of the problem
            lam (double): lambda parameter for linUCB
            T (int): time horizon
            pe (bool): if True, we make OB-PE instead of OB-LinUCB
        '''

        # what approximation degree to use
        self.N = N
        self.dim = dim
        self.pe = pe

        # time horizon
        self.T = T

        # choose the basis
        self.basis = basis
        self.lam = lam

        if basis == 'poly':
            self.feature_map = poly_features
        elif basis == 'cosin':
            self.feature_map = cosin_features
        elif basis == 'fourier':
            self.feature_map = sincos_features
        elif basis == 'legendre':
            self.feature_map = legendre_features

        self.build_arms(numel=int(T**0.5))
        if pe:
            self.make_PE()
        else:
            self.make_linUCB()
   
    def build_arms(self, numel=100):
        '''
        Build the linear arms to feed into bandit algorithms

        Parameters:
            numel (int): in how many points to discretize the space in every dimension
        '''

        # build mesh for the space
        x = np.linspace(-1,1,numel)
        y = np.linspace(-1,1,numel)
        self.X,self.Y = build_mesh(x,y)

        feature_map_x = self.feature_map(self.X,d=self.N)
        feature_map_y = self.feature_map(self.Y,d=self.N)

        self.linear_arms = merge_feature_maps([feature_map_x, feature_map_y])


    def reset(self):
        '''
            Resets the algorithm
        '''
        self.learner.reset()

    def make_linUCB(self):
        # initialize linUCB
        self.learner = linUBC(self.linear_arms, lam=self.lam, T=self.T)

    def make_PE(self):
        # initialize linUCB
        self.learner = PE(self.linear_arms, T=self.T)

    def pull_arm(self):
        '''
        Asks the linear bandit algorithms which arm to pull, and then converts their index in the actual arm

        Returns:
            _ (vector): vector corresponding to the chosen arm
        '''
        if self.pe:
            arm = self.learner.pull_arm()
        else:
            _, arm = self.learner.pull_arm()
        self.last_arm_pulled = arm
        return np.array([self.X[arm],self.Y[arm]])
    
    def update(self, reward):
        '''
        Updates learner corresponding to the last pulled arm
        '''
        # pass from arm index to arm vector
        arm_vector = self.linear_arms[self.last_arm_pulled,:]
        # update base learner
        self.learner.update(arm_vector, reward)

    



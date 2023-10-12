from classes.agents.linear_agents.linUCB import linUBC
from functions.misc.build_mesh import build_mesh
from functions.misc.merge_feature_maps import merge_feature_maps
from functions.orthogonal.bases import *
import numpy as np


class OBlinUCB:
    def __init__(self, basis, N, dim=2, lam=1, T=10000):

        # what approximation degree to use
        self.N = N
        self.dim = dim

        # time horizon
        self.T = T

        # choose the basis
        self.basis = basis
        self.lam = lam

        if basis == 'poly':
            self.feature_map = poly_features
        elif basis == 'cosin':
            self.feature_map = cosin_features
        elif basis == 'sincos':
            self.feature_map = sincos_features
        elif basis == 'legendre':
            self.feature_map = legendre_features

        self.build_arms()
        self.make_linUCB()
   
    def build_arms(self, numel=100):

        # build mesh for the space
        x = np.linspace(-1,1,numel)
        y = np.linspace(-1,1,numel)
        self.X,self.Y = build_mesh(x,y)

        feature_map_x = self.feature_map(self.X,d=self.N)
        feature_map_y = self.feature_map(self.Y,d=self.N)

        self.linear_arms = merge_feature_maps([feature_map_x, feature_map_y])


    def reset(self):
        self.learner.reset()

    def make_linUCB(self):
        # initialize linUCB
        self.learner = linUBC(self.linear_arms, lam=self.lam, T=self.T)

    def pull_arm(self):        
        # ask what arm to pull
        _, arm = self.learner.pull_arm()
        return np.array([self.X[arm],self.Y[arm]])
    
    def update(self, arm, reward):
        # pass from arm index to arm vector
        arm_vector = self.linarms[arm,:]
        # update base learner
        self.learner.update(arm_vector, reward)

    



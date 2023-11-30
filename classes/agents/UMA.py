from classes.agents.linear_agents.misspec import misSpec
from functions.misc.build_mesh import build_mesh
from functions.misc.merge_feature_maps import merge_feature_maps
from functions.orthogonal.bases import *
import numpy as np


class UMA:
    def __init__(self, N, bins=4, dim=2, lam=1, T=10000):
        '''
        Instanciate the OBlinUCB and the OB-PE algorithms

        Parameters:
            N (int): degree of the basis function
            bins (int): how many discretization bins to build in each dimension
            dim (int): dimension of the problem
            lam (double): lambda parameter for linUCB
            T (int): time horizon
        '''
        self.N = N
        self.bins = bins
        self.dim = dim

        # time horizon
        self.T = T
        self.lam = lam
        self.feature_map = poly_features
        self.bin_dictionary = {}

        self.build_bins(numel=int(T**0.5))

   
    def build_bins(self, numel=100):
        if self.dim > 2:
            raise NotImplementedError
        
        self.bingrid = np.linspace(-1,1,self.bins+1)
        
        for i in range(self.bins):
            for j in range(self.bins):

                x = np.linspace(self.bingrid[i],self.bingrid[i+1],numel)
                y = np.linspace(self.bingrid[j],self.bingrid[j+1],numel)
                self.X,self.Y = build_mesh(x,y)

                feature_map_x = self.feature_map(self.X,d=self.N)
                feature_map_y = self.feature_map(self.Y,d=self.N)

                linear_arms = merge_feature_maps([feature_map_x, feature_map_y])
                self.bin_dictionary[(i,j)] = misSpec(linear_arms, self.X, self.Y)


    def reset(self):
        for i in range(self.bins):
            for j in range(self.bins):
                self.bin_dictionary[(i,j)].reset()

    def pull_arm(self):        
        
        upper_bounds = np.zeros((self.bins,self.bins))
        for i in range(self.bins):
            for j in range(self.bins):
                upper_bounds[i,j] = self.bin_dictionary[i,j].upper_bound

        best_bin = np.unravel_index(np.argmax(upper_bounds), upper_bounds.shape)

        # This line is crucial! It has to return
        # arm: arm as a vector
        arm, idx = self.bin_dictionary[best_bin].pull_arm()

        self.last_bin_pulled = best_bin
        self.last_idx_puller = idx
        return arm
    
    def update(self, reward):
        self.bin_dictionary[self.last_bin_pulled].update(self.last_idx_puller, reward)

    



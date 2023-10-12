import numpy as np
import matplotlib.pyplot as plt

class CMAB:
    '''
    Class to make continous MAB envornment
    '''

    def __init__(self, sigma=0.5, curve='gaussian', seed = 257):
        '''
        Defines the variables in the environment

        Parameters:
        sigma (double): standard deviation of the noise
        curve (str): name of the reward curve to build. Possible values:
            'gaussian',
        seed (int): random seed to fix
        '''

        np.random.seed(seed)

        # standard deviation of the noise
        self.sigma = sigma

        # make curve
        self.generate_curves(curve)
        
            

    def generate_curves(self, curve):
        '''
        Generate the reward curve corresponing to its name

        Parameters:
        curve (str): name of the curve
        '''

        # Cinfty even
        if (curve == 'gaussian'):
            def curve(x):
                return np.exp(-x[0]**2-x[1]**2)
            self.xopt = np.zeros(2)
            self.opt = curve(self.xopt)
            self.d = 2
                
        self.reward_curve = curve



    def pull_arm(self, x):
        '''
        Chooses which arm to pull and get the corresponding reward

        Prameters:
        x (vector): the vector to be fed as input of the reward curve

        Returns:
        reward (double): random reward achieved by pulling the arm
        expected regret (double): expected value of the reward
        '''
        mu = self.reward_curve(x)
        reward = np.random.normal(mu, self.sigma)
        expected_regret = self.opt - mu
        return reward, expected_regret
    
    def plot_curve(self):
        '''
        Plots on screen an image representing the reward function (ONLY FOR 2D REWARD CURVES!)
        '''
        if not self.d == 2:
            raise ValueError('the reward function must be bidimensional to make an heathmap')

        N = 200
        values = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                x = np.array([2*i/N-1, 2*j/N-1])
                values[i,j] = self.reward_curve(x)
        plt.imshow(values, cmap='hot', interpolation='nearest')
        plt.show()



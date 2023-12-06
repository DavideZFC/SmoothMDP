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
        if curve == 'gaussian':
            def curve(x):
                a = 2.0
                b = 1.0
                return a*np.exp(-x[0]**2-x[1]**2)-b
            self.d = 2

        if curve == 'glass':
            def curve(x):
                return np.sin(np.sqrt((5*x[0]) ** 2 + (5*x[1]) ** 2))
            self.d = 2

        if curve == 'concoide':
            def curve(x):
                a = 1
                b = 1
                s = (x[0]) ** 2 + (x[1]) ** 2
                coef = 0.2
                return coef*(- (s-a*x[0])**2 + b*s)
            self.d = 2

        if curve == 'cardioide':
            def curve(x):
                a = 1
                s = 0.5*((x[0]) ** 2 + (x[1]) ** 2)
                return - s*(s-a*x[0]) + a**2*x[1]**2
            self.d = 2

        if curve == 'cardsin':
            def curve(x):
                a = 5
                eps = 0.01
                s = ((a*x[0]) ** 2 + (a*x[1]) ** 2) + eps
                return np.sin(s) / s
            self.d = 2

        if curve == 'faglia':
            def curve(x):
                return 1 - (x[1])**2 - np.abs(x[0])**0.5
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

    def plot_reward_curve(self, dir):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Crea i dati per il grafico
        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(x, y)
        def f(x,y):
            return self.reward_curve(np.array([x,y]))
        Z = f(X, Y)
        self.opt = np.max(Z)

        # Plotta la funzione
        ax.plot_surface(X, Y, Z)

        # Aggiungi etichette e titolo
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')

        # Mostra il grafico
        plt.savefig(dir+'reward_curve.pdf')
        plt.clf()




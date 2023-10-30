import os
import datetime
from joblib import Parallel, delayed
import time
import json
import numpy as np
from functions.misc.plot_data import plot_data
from functions.misc.confidence_bounds import bootstrap_ci
import matplotlib.pyplot as plt
from copy import deepcopy
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from classes.environments.PQR_stable import PQR
    
    

def test_algorithm(env, dir, K=30, seeds=1, first_seed=1):
    '''
    Test a given policy on an environment and returns the regret estimated over some different random seeds

    Parameters:
        env (class environment): environment over which to test the policy
        K (int): number of episodes
        seeds (int): how many random seeds to use
        first seed (int): first seed to use

    Returns:
        regret_matrix (array): matrix having as rows the value of the cumulative regret for one random seed
    '''
    H = env.time_horizon
    T = H*K
    return_matrix = np.zeros((seeds, K))
    np.random.seed(first_seed)


    for seed in range(seeds):
        door = dir
        door = dir+'_seed_{}/'.format(seed)
        os.mkdir(door)
        print(door)
        env = Monitor(env, door)

        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = SAC("MlpPolicy", env, action_noise=action_noise, verbose=0)

        # Train the agent
        model.learn(total_timesteps=int(T))
        xy_list = plot_results([door], T, results_plotter.X_TIMESTEPS, "SAC pendulum")

        return_matrix[seed, :] = xy_list[0][1]
            
    return return_matrix



def make_experiment_SAC(env, seeds, K, labels, window=None, exp_name=''):
    '''
    Performs a RL experiment, estimating the reward curve and saving the data in a given folder

    Parameters:
        env (class environment): environment over which to test the policies
        K (int): number of episodes
        seeds (int): how many random seed to use in the experiment
        labels (list): list with the same length of policies giving a name to each one
        exp_name (string): string to be added to the filder created to same the data    
    '''

    # create folder
    tail = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_")
    dir = 'results/'+'_'+tail+exp_name

    os.mkdir(dir)
    dir = dir+'/'

    # in this dictionary, we store the running times of the algorithms
    running_times = {}

    if window is None:
        window = env.time_horizon

    
    for i in range(len(labels)):

        ####################################
        # actual algorithm simulation

        # evaluate running time of the algorithm
        t0 = time.time()

        # test the algorithm

        results = test_algorithm(env, dir=dir, seeds=seeds, K=K, first_seed=1)

        # store time
        t1 = time.time()
        running_times[labels[i]] = t1 - t0
        
        print(labels[i] + ' finished')

        ####################################
        # part to save data

        # make nonparametric confidence intervals
        # low, high = bootstrap_ci(results)

        # make plot
        # plot_data(np.arange(0,env.time_horizon*K), low, high, col='C{}'.format(i), label=labels[i])

        # save data in given folder
        np.save(dir+labels[i], results)

        # make nonparametric confidence intervals
        low, high = bootstrap_ci(results)

        # make plot
        plot_data(np.arange(0,len(low)), low, high, col='C{}'.format(i+1), label=labels[i]+' smooth')

    with open(dir+"running_times.json", "w") as f:
        # Convert the dictionary to a JSON string and write it to the file
        json.dump(running_times, f)
    
    plt.legend()
    plt.title('Reward curves')
    plt.savefig(dir+'reward_plot.pdf')
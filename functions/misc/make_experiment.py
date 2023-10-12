import os
import datetime
from joblib import Parallel, delayed
import time
import json
import numpy as np
from functions.misc.test_algorithm import test_algorithm
from functions.misc.plot_data import plot_data
from functions.misc.confidence_bounds import bootstrap_ci
import matplotlib.pyplot as plt

def make_experiment(policies, env, T, seeds, labels, exp_name=''):

    # create folder
    tail = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_")
    dir = 'results/'+'_'+tail+exp_name

    os.mkdir(dir)
    dir = dir+'/'

    # in this dictionary, we store the running times of the algorithms
    running_times = {}

    
    for i in range(len(policies)):

        ####################################
        # actual algorithm simulation

        # evaluate running time of the algorithm
        t0 = time.time()

        # test the algorithm
        results = Parallel(n_jobs=seeds)(delayed(test_algorithm)(policies[i], env, T, seeds=1, first_seed=seed) for seed in range(seeds))

        # store time
        t1 = time.time()
        running_times[labels[i]] = t1 - t0
        
        print(labels[i] + ' finished')

        ####################################
        # part to save data

        results = np.concatenate(results, axis=0)

        # make nonparametric confidence intervals
        low, high = bootstrap_ci(results)

        # make plot
        plot_data(np.arange(0,T), low, high, col='C{}'.format(i), label=labels[i])

        # save data in given folder
        np.save(dir+labels[i], results)

    with open(dir+"running_times.json", "w") as f:
        # Convert the dictionary to a JSON string and write it to the file
        json.dump(running_times, f)
    
    plt.legend()
    plt.title('Regret curves')
    plt.savefig(dir+'regret_plot.pdf')
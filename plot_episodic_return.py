import numpy as np
import matplotlib
import json
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from functions.misc.plot_data import plot_data
from functions.misc.confidence_bounds import bootstrap_ci

# things to change
time_horizon = 200
dir = 'results/_23_10_26-13_50_mountain car'
optimum = -0.0

##################
# things to keep

# get the labels
with open(dir+"/running_times.json", "r") as f:
    running_times = json.load(f)

labels = list(running_times.keys())

# plot the data with color
c = 1
for l in labels:
    results = np.load(dir+'/'+l+'.npy')
    seeds = results.shape[0]

    weights = np.repeat(1.0, time_horizon)
    reward_smoothed = np.zeros((seeds, results.shape[1]-time_horizon+1))

    for seed in range(seeds):
        reward_smoothed[seed,:] = np.convolve(results[seed,:], weights, 'valid')
    
    # make nonparametric confidence intervals
    low, high = bootstrap_ci(reward_smoothed, resamples=1000)

    low = low[::time_horizon]
    high = high[::time_horizon]

    T = len(low)
    color = 'C{}'.format(c)
    true_lab = l.replace('_','')

    # make plot
    plot_data(np.arange(0,T), low, high, col=color, label=true_lab)

    # update color
    c += 1

    print(l+' done')

plt.axhline(y=optimum, color='y', linestyle='--', label='opitmum')
plt.legend()
plt.title('Episodic Return')
plt.savefig(dir+'/episodic_return.pdf')
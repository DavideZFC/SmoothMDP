import matplotlib.pyplot as plt
import json
import numpy as np
from functions.misc.confidence_bounds import bootstrap_ci
from functions.misc.plot_data import plot_data
import os

def filter_and_save(x, v1, v2, dir, filter=10):
    ''' 
    Save in a given directory some files containing a table of xy coordinates that can be plotted with latex

    Parameters:
        x (vector): vector of the x axis
        v1 (vector): lower bound on the curve to plot
        v2 (vector): upper bound on the curve to plot
        dir (string): where to save the files
        fitler (int): take one data every filter, to use less memory
    '''

    weights = np.ones(filter)/filter
    v1 = np.convolve(v1, weights, 'valid')
    v1 = v1[::filter]
    v2 = np.convolve(v2, weights, 'valid')
    v2 = v2[::filter]
    x = x[::filter]
    if len(v1) < len(x):
        v1 = np.append(v1,v1[-1])
        v2 = np.append(v2,v2[-1])
    names = ['mean', 'low', 'up']
    for name in names:
        if name == 'mean':
            mat = np.column_stack((x,(v1+v2)/2))
        elif name == 'low':
            mat = np.column_stack((x,v1))
        elif name == 'up':
            mat = np.column_stack((x,v2))
        name = dir+'/'+name+'.txt'
        np.savetxt(name, mat)

def plot_label(label, color, filename = 'TeX/template_plot.txt'):
    '''
    Open template file and replaces letters H and K with given strings

    Parameters:
        label (string): what to replace H with
        color (string): what to replace K with
        filename (string): where to find the original file

    Returns
        content (string): what is contained in the new file
    '''

    with open(filename, 'r') as file:
        # read in the contents of the file
        contents = file.read()
    file.close()

    # replace all occurrences of 'H' with 'my_word'
    contents = contents.replace('H', label)

    # replace all occurrences of 'K' with 'my_other_word'
    contents = contents.replace('K', color)

    return contents

def add_file(filename='TeX/reference_tex.txt'):    
    with open(filename, 'r') as file:
        content = file.read()
    file.close()
    return content


dir = 'results\_31_13_30-11_24_poly vs legendre 2'

with open(dir+"/running_times.json", "r") as f:
    # Convert the dictionary to a JSON string and write it to the file
    running_times = json.load(f)




# makes the barplot
plt.bar(running_times.keys(), np.log(np.array(list(running_times.values()))))

plt.xticks(rotation=15)
plt.tick_params(axis='x', labelsize=8)
plt.ylabel('log(time)')
plt.savefig(dir+'/running_times.pdf')
plt.show()

# questa andrà ridefinita
labels = list(running_times.keys())
togli = ['UCB1', 'FourierUCB', 'LegendreUCB', 'ChebishevUCB']
labels = [x for x in labels if x not in togli]
print(labels)



new_dir = dir+'/TeX'
if not os.path.exists(new_dir):
    os.mkdir(new_dir)

mid_dir = new_dir + '/data'
if not os.path.exists(mid_dir):
    os.mkdir(mid_dir)

c = 0
time_horizon = 20

with open(new_dir+'/main.txt', 'w') as new_file:
        new_file.write(add_file())


for l in labels:
    results = np.load(dir+'/'+l+'.npy')
    seeds = results.shape[0]

    # define uniform function to smooth the reward function
    weights = np.repeat(1.0, time_horizon)
    reward_smoothed = np.zeros((seeds, results.shape[1]-time_horizon+1))

    # make smoothing of the reward
    for seed in range(seeds):
        reward_smoothed[seed,:] = np.convolve(results[seed,:], weights, 'valid')
    
    # make nonparametric confidence intervals
    low, high = bootstrap_ci(reward_smoothed, resamples=1000)

    # take one element every "time_horizon", to have the return of every episode
    low = low[::time_horizon]
    high = high[::time_horizon]

    T = len(low)
    # selects the color for the curve and modifies the label
    color = 'C{}'.format(c)
    true_lab = l.replace('_','')

    # make plot
    plot_data(np.arange(0,T), low, high, col=color, label=true_lab)

    # update color
    c += 1

    # in this part, we crea new folders to save all the necessary info
    this_dir = mid_dir + '/' + true_lab
    if not os.path.exists(this_dir):
        os.mkdir(this_dir)
    filter_and_save(np.arange(0,T), low, high, this_dir, filter=10)
    print(l+' done')

# in this part, we prepare the tex file
# we cycle on the various value of the label, and for each of them write in the fail main.txt 
# the commands used make the plot for the given label, which are taken from the pre-existing file 'TeX/reference_tex.txt'
c = 0
for l in labels:
    color = 'C{}'.format(c)
    c += 1
    true_lab = l.replace('_','')
    with open(new_dir+'/main.txt', 'a') as new_file:
        new_file.write(plot_label(true_lab, color))

# we do as before, but this time instead of just plotting the curve, we make a colores confidence interval
# and the pre-existing file is 'TeX/template_fill.txt'
c = 0
for l in labels:
    color = 'C{}'.format(c)
    c += 1
    true_lab = l.replace('_','')
    with open(new_dir+'/main.txt', 'a') as new_file:
        new_file.write(plot_label(true_lab, color, filename = 'TeX/template_fill.txt'))

with open(new_dir+'/main.txt', 'a') as new_file:
    new_file.write(add_file('TeX/refrence_end.txt'))


plt.legend()
plt.title('Regret curves')
plt.savefig(dir+'/regret_plot.pdf')




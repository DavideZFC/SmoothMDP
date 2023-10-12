import numpy as np
import matplotlib.pyplot as plt

def plot_data(x, v1, v2, col, label):
    ''' Function to plot some data given upper and lower bound
        x = x coordinate
        v1 = lower bound
        v2 = upper bound
        col = color on the plot
        label = label for the image
    '''
    
    plt.plot(x, (v1+v2)/2, label=label, color=col)
    plt.fill_between(x, y1=v1, y2=v2, color=col, alpha=0.3)
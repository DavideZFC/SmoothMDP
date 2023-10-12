import numpy as np

def bootstrap_ci(x, conf=0.95, resamples=10000):
    ''' Function to output bootstrap confidence interval
        x = matrix with each row corresponding to a sample
        
        Returns:
        two vectors low, high with lenght corresponding to the numbe of coluns of x which are lower and upper bound respectively
    '''
    means = [np.mean(x[np.random.choice(x.shape[0], size=x.shape[0], replace=True), :], axis=0) for _ in range(resamples)]
    low = np.percentile(means, (1-conf)/2 * 100, axis=0)
    high = np.percentile(means, (1 - (1-conf)/2) * 100, axis=0)
    low = np.nan_to_num(low)
    high = np.nan_to_num(high)
    return low, high
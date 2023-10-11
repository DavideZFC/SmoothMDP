import numpy as np

def build_mesh(x,y):
    '''
    Builds a mesh given list of x,y cooordinates

    Parameters:
    x (vector): x coordinates
    y (vector): y coordinates

    Returns:
    X (vector): vector containing the x coordinate of each point in the cartesian product of x and y. 
        This vector will have a length equal to the product of the original lengths
    Y (vector): same as Y for the y axis
    '''
    X, Y = np.meshgrid(x, y)
    X = np.concatenate(X)
    Y = np.concatenate(Y)

    return X,Y
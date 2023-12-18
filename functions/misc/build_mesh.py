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

def generalized_build_mesh(arrays):
    '''
    Builds a mesh given list of x,y,z,... cooordinates

    Parameters:
    arrays (list): list of arrays of coordinates

    Returns:
    mesh_list (list): list of arrays after the meshgrid
    '''

    if len(arrays) < 2:
        return arrays[0]


    meshed = np.meshgrid(*arrays)

    mesh_list = []
    d = len(mesh_list)
    for mesh in meshed:
        for cap in range(d+1):
            mesh = np.concatenate(mesh)
            print(mesh)
            print('ok ' + str(cap))
        mesh_list.append(np.concatenate(mesh))

    return mesh_list
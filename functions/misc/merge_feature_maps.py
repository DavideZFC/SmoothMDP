from functions.misc.get_summatory_index import get_summatory_index
import numpy as np

def merge_feature_maps(phi):
    '''
    This function merges some feature maps

    Parameters:
        phi (list): list of feature maps we want to merge. They are collected in a list, each of the elements being a vector of fixed length

    Returns:
        full_feature (array): matrix containing the combined feature matrix evaluated in all the original points
    '''
    num_maps = len(phi)
    evaluation_points = phi[0].shape[0] # number of points in which the features are evaluated
    feature_length = phi[0].shape[1] # length of each feature map
    

    # check that all maps have the same shape
    i = 0
    for p in phi:
        if not p.shape[0] == evaluation_points or not p.shape[1] == feature_length:
            raise ValueError("invalid dimension for map {}".format(i))
        i += 1

    multi_index_matrix = get_summatory_index(feature_length, num_maps)

    mi_matrix_rows = multi_index_matrix.shape[0]

    full_feature = np.zeros((evaluation_points, mi_matrix_rows))
    for i in range(mi_matrix_rows):
        baseline = np.ones(evaluation_points)
        for j in range(num_maps):
            baseline *= phi[j][:,multi_index_matrix[i,j]]
        full_feature[:,i] = baseline

    return full_feature
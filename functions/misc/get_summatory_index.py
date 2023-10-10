import numpy as np

def get_summatory_index(N,l,spy=True):
    '''
    This function takes as input N and l and returns a matrix which has, on the rows, all the ways to
    write N as a sum of l nonnegative numbers

    Parameters:
        N (int): maximal sum accepted
        l (int): length of the vectors

    Returns:
        matrix_return (array): matrix containing the multiindex
    '''
    if spy:
        N = N-1
        if N < 0:
            raise ValueError("N must be strictly positive")
    
    if N == 0:
        return np.zeros((1,l))
    if l == 1:
        return np.expand_dims(np.arange(N+1),axis=1)


    # first iteration is done by hand
    loaded_matrix_return = get_summatory_index(0,l-1,spy=False)
    matrix_return = np.append((N)*np.ones((loaded_matrix_return.shape[0],1)), loaded_matrix_return, axis=1)

    for j in range(1, N+1):

        # fix first value and iterate over the others
        loaded_matrix_return = get_summatory_index(j,l-1,spy=False)
        matrix_return = np.append(matrix_return, np.append((N-j)*np.ones((loaded_matrix_return.shape[0],1)), loaded_matrix_return, axis=1), axis=0)


    return matrix_return.astype(int)
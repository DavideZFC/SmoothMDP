import numpy as np



def compute_induced_norm(Ainv, v):
    results = np.zeros(v.shape[0])
    for i in range(v.shape[0]):
        results[i] = np.dot(v[i,:].T, np.dot(Ainv, v[i,:]))
    return results

def compute_design_matrix(A, pi):
    D = np.zeros((A.shape[1],A.shape[1]))

    for i in range(A.shape[0]):
        D += pi[i]*np.dot(A[i:i+1,:].T,A[i:i+1,:])
    return D

def squeeze_distribution(pi, n):
    # apply noise injection to avoid ties
    pi = pi + np.random.normal(0,scale=1e-4,size=len(pi))

    sorted_vals = sorted(pi, reverse=True)
    nth_largest = sorted_vals[min(n, len(sorted_vals))-1]
    pi[pi<nth_largest] = 0
    pi = pi/np.sum(pi)
    return pi

def onehot(idx, k):
    v = np.zeros(k)
    v[idx] = 1
    return v

def eval_pi(pi, A):
    D = compute_design_matrix(A, pi)
    Dinv = np.linalg.inv(D)
    v = compute_induced_norm(Dinv, A)
    return np.max(v)


def find_optimal_design(A, iter=100, thresh=0):
    k = A.shape[0]
    d = A.shape[1]
    pi = np.ones(k)/k

    for it in range(iter):
        D = compute_design_matrix(A, pi)
        Dinv = np.linalg.inv(D)
        v = compute_induced_norm(Dinv, A)

        best_index = np.argmax(v)
        current = v[best_index]
        if current < (thresh + 1)*A.shape[1]:
            break
        gamma = (current/d-1)/(current-1)

        pi = (1-gamma)*pi + gamma*onehot(best_index, k)
    before = eval_pi(pi, A)
    pi = squeeze_distribution(pi, int(4*d*(np.log(d))))
    after = eval_pi(pi, A)
    print('optimal design found: before squeezing {}, after squeezing {}'.format(before, after))
    if after > 2*d:
        print('Error we are fucked')
    return pi
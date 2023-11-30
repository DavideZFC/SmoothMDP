import numpy as np

class misSpec:
    def __init__(self, arms_matrix, X, Y, C1=128, sigma=1., lam=1., T=1000000, m=1, epsilon=0.01):

        # dimension of the arms
        self.d = arms_matrix.shape[1]

        # lambda parameter of ridge regression
        self.lam = lam

        # matrix of the arms
        self.arms = arms_matrix
        self.n_arms = self.arms.shape[0]

        # initialization of the matrix A of the system
        self.design_matrix = lam*np.identity(self.d)

        # initialization of the known term b of the system
        self.load = np.zeros(self.d).reshape(-1,1)

        # pulled arms history
        self.pulled_arms = []

        # time horizon
        self.T = T
        self.t = 0

        # constants
        self.C1 = C1
        self.sigma = sigma

        # error probability
        self.delta = 1/T

        # upper bound on the norm of theta
        self.m = m

        # misspecification error
        self.epsilon = epsilon

        # maximum norm of an arm vector
        self.L = 0
        for i in range(self.arms.shape[0]):
            self.L = max(self.L,self.norm(self.arms[i,:]))
        
        # print('Highest norm estimated {}'.format(self.L))

        self.X = X
        self.Y = Y

        # random big number
        self.upper_bound = 100.
        self.compute_beta_routine()

    def reset(self):
        self.design_matrix = self.lam*np.identity(self.d)
        self.load = np.zeros(self.d).reshape(-1,1)
        self.t = 0
        self.upper_bound = 100.


    def norm(self,v):
        return (np.sum(v**2))**0.5

    def compute_beta_routine(self):
        self.beta = np.zeros(self.T)

        for t in range(self.T):
            beta = self.C1*self.sigma**2*self.d*np.log(1+t)*2*np.log(4*(t+1)/(self.delta))

            # this is actually sqrt beta
            self.beta[t] = beta**0.5

        # print('Beta routine :'+str(self.beta))

    def update(self, arm, reward):
        arm = self.arms[arm]
        self.design_matrix += np.matmul(arm.reshape(-1,1),arm.reshape(1,-1))
        self.load += reward*arm.reshape(-1,1)

    def find_maximum(self, A, v):
        ''' computes the maximum of the linear form (x.T*v)(x.T*A*x) '''

        def diaginv(A):
            d = A.shape[0]
            B = np.zeros_like(A)
            for i in range(d):
                B[i,i] = A[i,i]**(-1)
            return B

        def makediag(D):
            d = len(D)
            I = np.identity(d)
            for i in range(d):
                I[i,i] = D[i]
            return I
        
        D, U = np.linalg.eig(A)
        if (0 and np.any(np.iscomplex(D))):
            print('Complex number found at step {}'.format(self.t))
            print(D)
            print(A)
        
        
        D = makediag(D)
        invD = diaginv(D)


        matrix = np.matmul(invD**(0.5),U.T)
        v_based = np.matmul(matrix, v.reshape(-1,1))
        
        return (np.sum(v_based**2))**0.5


    def compute_reverse_matrix(self, A):
        def diaginv(A):
            d = A.shape[0]
            B = np.zeros_like(A)
            for i in range(d):
                B[i,i] = A[i,i]**(-1)
            return B

        def makediag(D):
            d = len(D)
            I = np.identity(d)
            for i in range(d):
                I[i,i] = D[i]
            return I
        
        D, U = np.linalg.eig(A)
        if (np.any(np.iscomplex(D))):
            # print('Complex number found at step {}'.format(self.t))
            Dimag = np.imag(D)
            D = np.real(D)
            # print('Eliminated imaginary part of modulus {}'.format(np.sum(np.abs(Dimag))))
            U = np.real(U)
        
        
        D = makediag(D)
        invD = diaginv(D)


        matrix = np.matmul(invD**(0.5),U.T)
        return matrix
    


    def pull_arm(self):
        estimates = np.zeros(self.n_arms)
        thetahat = self.estimate_theta().flatten()
        mat = self.compute_reverse_matrix(self.design_matrix)

        estimates = np.matmul(self.arms, thetahat.reshape(-1,1))
        upper_bounds = self.beta[self.t]*np.matmul(self.arms, mat.T)
        a = np.sum(upper_bounds**2, axis = 1)**0.5
        a = a.reshape(-1,1)
        estimates += a

        ## fin qui abbiamo le stime con l'upper bound con beta. Manca la parte in cui moltiplichiamo il misspecification upper bound
        error_term = np.zeros_like(estimates)

        if self.epsilon>0:
            Ainv = np.linalg.inv(self.design_matrix)
            for j in range(len(self.pulled_arms)):
                old_term = np.dot(Ainv,self.pulled_arms[j])
                error_term += np.abs(np.dot(self.arms, old_term.reshape(-1,1)))

            estimates += self.epsilon*error_term

        # in this way, we have an upper bound for the maximal reward that we expect.
        # this is crucial for the full algorithm
        self.upper_bound = np.max(estimates)

        self.t += 1

        arm_idx = np.argmax(estimates)
        chosen_arm = self.arms[arm_idx]
        self.pulled_arms.append(chosen_arm)

        vector_pulled_arm = np.array([self.X[arm_idx], self.Y[arm_idx]])

        return vector_pulled_arm, arm_idx

    def estimate_theta(self):
        return np.linalg.solve(self.design_matrix, self.load)



    


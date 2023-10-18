import numpy as np

class linUBC:
    '''
    Class for the celebrated LinUCB algorithm from "Improved algorithms for linear stochastic bandits"
    '''
    
    def __init__(self, arms_matrix, lam=0.01, m=1, T=1000000.0, exp=1):
        '''
        Initialize algorithm

        Parameters:
            arms matrix (array): matrix having the vectors corresponding to the arms as rows
            lam (double): lambda coefficient of ridge regression
            m (double): upper bound on the norm of theta
            T (int): time horizon
            exp (double): dependence of the error probability on the time horizon
        '''

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

        # time horizon
        self.T = T
        self.t = 0

        # error probability
        self.delta = T**(-exp)

        # upper bound on the norm of theta
        self.m = m

        # maximum norm of an arm vector
        self.L = 0
        for i in range(self.arms.shape[0]):
            self.L = max(self.L,self.norm(self.arms[i,:]))
        
        # print('Highest norm estimated {}'.format(self.L))

        self.compute_beta_routine()

    def reset(self):
        '''
        Resets the agent, by making all the variables assume their original value
        '''
        self.design_matrix = self.lam*np.identity(self.d)
        self.load = np.zeros(self.d).reshape(-1,1)
        self.t = 0


    def norm(self,v):
        '''
        Computes 2-norm of a vector
    
        Parameters:
            v (vector)

        Returns:
            _: norm of v
        '''
        return (np.sum(v**2))**0.5

    def compute_beta_routine(self):
        '''
        Computes the beta routine that trades off exploration and exploitation
        '''

        self.beta = np.zeros(self.T)

        for t in range(self.T):
            first = self.m * self.lam**0.5
            second = 2*np.log(1/self.delta) + self.d*np.log((self.d*self.lam+t*self.L**2)/(self.d*self.lam))
            second = second**0.5
            # with the notation of Lattimore-Szepesvari, this is beta**0.5
            self.beta[t] = first + second

    def update(self, arm, reward):
        '''
        Updates internal variable after receiving a reward
    
        Prameters:
            arm (int): which arm was pulled
            reward (double): observed reward
        '''
        self.design_matrix += np.matmul(arm.reshape(-1,1),arm.reshape(1,-1))
        self.load += reward*arm.reshape(-1,1)

    def find_maximum(self, A, v):
        ''' 
        Computes the maximum of the linear form (x.T*v)(x.T*A*x) 
        
        Parameters:
            A (array): matrix A
            v (vector): vector v
                
        Returns:
            maximum value of the linear form (x.T*v)(x.T*A*x) over x.
        '''

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
        '''
        Given symmetric positive definite matrix, computes matrix A**(-1/2)
            
        Parameters:
            A (array): matrix A
                
        Retunrs:
            matrix (array): A**(-1/2)
        '''
        
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
        '''
        Chooses which arm to pull

        Returns:
            _ (vector): the vector corresponding to the pulled arm
            _ (int): index of the pulled arm
        '''
        estimates = np.zeros(self.n_arms)
        thetahat = self.estimate_theta().flatten()
        mat = self.compute_reverse_matrix(self.design_matrix)

        estimates = np.matmul(self.arms, thetahat.reshape(-1,1))
        upper_bounds = self.beta[self.t]*np.matmul(self.arms, mat.T)
        a = np.sum(upper_bounds**2, axis = 1)**0.5
        a = a.reshape(-1,1)
        estimates += a

        self.upper_bound = np.max(estimates)

        self.t += 1

        return self.arms[np.argmax(estimates)], np.argmax(estimates)



    def estimate_theta(self):
        '''
        Estimate the unknown vector theta with least squares method

        Returns:
            _ (vector): estimate of theta        
        '''
        return np.linalg.solve(self.design_matrix, self.load)



    


import numpy as np

def test_algorithm(policy, env, T, seeds, first_seed=1):
    '''
    Test a given policy on an environment and returns the regret estimated over some different random seeds

    Parameters:
        policy (specific class): policy to be tested
        env (class environment): environment over which to test the policy
        T (int): time horizon
        seeds (int): how many random seeds to use
        first seed (int): first seed to use

    Returns:
        regret_matrix (array): matrix having as rows the value of the cumulative regret for one random seed
    '''

    regret_matrix = np.zeros((seeds, T))
    np.random.seed(first_seed)

    for seed in range(seeds):

        policy.reset()
        for t in range(1,T):
            arm = policy.pull_arm()
            reward, expected_regret = env.pull_arm(arm)
            regret_matrix[seed, t] = regret_matrix[seed, t-1] + expected_regret
            policy.update(reward)
    
    return regret_matrix


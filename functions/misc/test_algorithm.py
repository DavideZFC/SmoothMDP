import numpy as np

def test_algorithm(policy, env, T, seeds, first_seed=1):

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


import numpy as np
from copy import deepcopy

def test_algorithm(agent0, env, K=30):
    '''
    Test a given policy on an environment and returns the regret estimated over some different random seeds

    Parameters:
        agent (specific class): policy to be tested
        env (class environment): environment over which to test the policy
        T (int): time horizon
        seeds (int): how many random seeds to use
        first seed (int): first seed to use

    Returns:
        regret_matrix (array): matrix having as rows the value of the cumulative regret for one random seed
    '''
    H = env.time_horizon
    T = H*K

    agent = deepcopy(agent0)

    returns = np.zeros(K)
    for k in range(K):

        state = env.reset()[0]
        done = False
        h = 0
        episodic_return = 0

        while not done:
            # action = env.action_space.sample()
            action = agent.choose_action(state, h)

            next_state, reward, terminated, truncated, _ = env.step(action)

            episodic_return += reward

            done = terminated or truncated
            state = next_state
            h += 1
        
        returns[k] = episodic_return
        print('episodic return {}'.format(episodic_return))
            
    return returns


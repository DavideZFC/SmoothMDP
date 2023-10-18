import numpy as np

def test_algorithm(agent, env, seeds, K=30, first_seed=1):
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
    reward_matrix = np.zeros((seeds, T))
    np.random.seed(first_seed)

    for seed in range(seeds):
        rew_index = 0

        for k in range(K):

            state = env.reset()[0]
            done = False
            h = 0

            while not done:
                # action = env.action_space.sample()
                action = agent.choose_action(state, h)

                next_state, reward, terminated, truncated, _ = env.step(action)

                reward_matrix[seed, rew_index] = reward
                rew_index += 1

                done = terminated or truncated

                agent.replay_buffer.memorize(state, action, next_state, reward)
                state = next_state
                h += 1

            agent.compute_q_values()

        agent.reset()
    
    return reward_matrix



from classes.agents.LSVI import LSVI
from classes.agents.Dummy import Dummy
from classes.environments.PQR import PQR
from functions.misc.make_experiment import test_algorithm
import numpy as np

K = 100
numel = 10000
discretize = 20
iterations = 100
env = PQR()
H = env.time_horizon

agent0 = Dummy(basis='legendre', approx_degree=4, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize, v_vector=None)

best = -10000

results = np.zeros(iterations)
for i in range(iterations):
    agent0.refresh(i)
    reward_matrix = test_algorithm(agent0, env, seeds=1, K=K, first_seed=1)
    curr = np.mean(reward_matrix)
    if curr > best:
        best = curr
    results[i] = curr
print(best)

import matplotlib.pyplot as plt
plt.hist(results)
plt.show()
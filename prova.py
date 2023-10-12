from functions.misc.merge_feature_maps import *
from functions.misc.test_algorithm import test_algorithm
from functions.orthogonal.bases import *
from classes.environments.CMAB import CMAB
from classes.agents.OBlinUCB import OBlinUCB

import numpy as np
import matplotlib.pyplot as plt


env = CMAB()
policy = OBlinUCB(basis = 'cosin', N=3)

regret_matrix = test_algorithm(policy, env, T=5000, seeds=3)

for i in range(3):
    plt.plot(regret_matrix[i,:])

plt.show()

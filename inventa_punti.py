from classes.environments.PendulumSimple import PendulumSimple
from classes.agents.FD_LSVI import FD_LSVI
from functions.misc.test_algorithm_after_learning import test_algorithm
from functions.orthogonal.bases import *

env = PendulumSimple()
agent = FD_LSVI(env)
state_disc = 40
action_disc = 20
approx_degree = 5
agent.get_datasets(disc_numbers=[state_disc, state_disc, action_disc], approx_degree=approx_degree, feature_map=sincos_features)
agent.compute_q_values()

returns = test_algorithm(agent, env)
import numpy as np
print('average return {} std {}'.format(np.mean(returns), np.std(returns)))
# problema: le q vengono sempre massimizzate da 1 o -1 a parte nel last step






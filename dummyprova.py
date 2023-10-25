
from classes.agents.LSVI import LSVI
from classes.agents.Dummy import Dummy
from classes.environments.PQR import PQR
from functions.misc.make_experiment import test_algorithm

K = 10
numel = 10000
discretize = 10
env = PQR()
H = env.time_horizon

agent0 = Dummy(basis='legendre', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize, v_vector=None)


reward_matrix = test_algorithm(agent0, env, seeds=1, K=1, first_seed=1)
print(reward_matrix)
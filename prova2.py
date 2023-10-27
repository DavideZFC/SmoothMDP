
from classes.agents.LSVI import LSVI
from classes.environments.Continuous_MountainCarEnv import Continuous_MountainCarEnv
from classes.environments.PQR import PQR
from functions.misc.make_experiment import make_experiment

K = 10
numel = 10000
discretize = 10
env = Continuous_MountainCarEnv()
H = env.time_horizon

agent0 = LSVI(basis='legendre', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize, time_horizon=H)
agent1 = LSVI(basis='legendre', approx_degree=3, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize, time_horizon=H)
agent1.compute_optimal_beta(K=K)
agent2 = LSVI(basis='legendre', approx_degree=4, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize, time_horizon=H)
agent3 = LSVI(basis='legendre', approx_degree=4, state_space_dim=env.observation_space.shape[0], action_space=env.action_space, numel=numel, discretize=discretize, time_horizon=H)
agent3.compute_optimal_beta(K=K)

labels = ['LSVI_b1_d3', 'LSVI_bB_d3', 'LSVI_b1_d4', 'LSVI_bB_d4']
agent_list = [agent0, agent1, agent2, agent3]

make_experiment(agent_list, env, seeds=5, K=K, labels=labels, exp_name='mountain car')
from functions.misc.make_experiment_SAC import *
from classes.environments.PQR_stable import PQR

env = PQR()
K = 100
H = env.time_horizon

labels = ['SAC']

make_experiment_SAC(env, seeds=5, K=K, labels=labels, exp_name='sac VS pqr')

from functions.misc.merge_feature_maps import *
from functions.misc.make_experiment import make_experiment
from functions.orthogonal.bases import *
from classes.environments.CMAB import CMAB
from classes.agents.OBlinUCB import OBlinUCB

import numpy as np
import matplotlib.pyplot as plt

deg = 5
exp_name = 'poly_cosin_sincos_legendre_{}'.format(deg)
seeds = 3
T = 5000

env = CMAB()
policies = [OBlinUCB(basis = 'poly', N=deg), OBlinUCB(basis = 'cosin', N=deg), OBlinUCB(basis = 'sincos', N=deg), OBlinUCB(basis = 'legendre', N=deg)]
labels = ['poly', 'cosin', 'sincos', 'legendre']

make_experiment(policies, env, T, seeds, labels, exp_name=exp_name)

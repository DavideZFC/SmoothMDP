from functions.misc.merge_feature_maps import *
from functions.misc.make_experiment import make_experiment
from functions.orthogonal.bases import *
from classes.environments.CMAB import CMAB
from classes.agents.OBlinUCB import OBlinUCB

import numpy as np
import matplotlib.pyplot as plt

deg = 3
exp_name = 'PE_vs_linUCB_{}'.format(deg)
seeds = 3
T = 5000

env = CMAB()
policies = [OBlinUCB(basis = 'sincos', N=deg, pe=True), OBlinUCB(basis = 'legendre', N=deg, pe=True), OBlinUCB(basis = 'sincos', N=deg), OBlinUCB(basis = 'legendre', N=deg)]
labels = ['PE-sincos', 'PE-legendre', 'sincos', 'legendre']

make_experiment(policies, env, T, seeds, labels, exp_name=exp_name)

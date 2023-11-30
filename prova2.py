from functions.misc.merge_feature_maps import *
from functions.misc.make_experiment import make_experiment
from functions.orthogonal.bases import *
from classes.environments.CMAB import CMAB
from classes.agents.OBlinUCB import OBlinUCB
from classes.agents.UMA import UMA

import numpy as np
import matplotlib.pyplot as plt

deg = 3
curve = 'faglia'
exp_name = 'deg_{}'.format(deg)+curve
seeds = 2
T = 1000

env = CMAB(curve=curve)
policies = [UMA(N=deg, T=T), OBlinUCB(basis = 'legendre', N=deg, T=T, pe=True), OBlinUCB(basis = 'legendre', N=deg, T=T)]
labels = ['UMA', 'OB-PE', 'OB-LinUCB']

make_experiment(policies, env, T, seeds, labels, exp_name=exp_name)

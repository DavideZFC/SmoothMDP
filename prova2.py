from functions.misc.merge_feature_maps import *
from functions.misc.make_experiment import make_experiment
from functions.orthogonal.bases import *
from classes.environments.CMAB import CMAB
from classes.agents.OBlinUCB import OBlinUCB

import numpy as np
import matplotlib.pyplot as plt

deg = 3
curve = 'glass'
exp_name = 'inutilePE_vs_linUCB_{}'.format(deg)+curve
seeds = 10
T = 10000

env = CMAB(curve=curve)
policies = [OBlinUCB(basis = 'legendre', N=deg, pe=True), OBlinUCB(basis = 'legendre', N=deg)]
labels = ['OB-PE', 'OB-LinUCB']

make_experiment(policies, env, T, seeds, labels, exp_name=exp_name)

from functions.misc.merge_feature_maps import *
from functions.misc.make_experiment import make_experiment
from functions.orthogonal.bases import *
from classes.environments.CMAB import CMAB
from classes.agents.OBlinUCB import OBlinUCB
from classes.agents.UMA import UMA
from classes.agents.advanced_learners import IGP_UCB
from classes.agents.zooming import ZOOM

import numpy as np
import matplotlib.pyplot as plt

deg = 3
curves = ['gaussian', 'glass', 'cardsin', 'cardioide', 'faglia']

for curve in curves:
    exp_name = 'deg_{}'.format(deg)+curve
    seeds = 5
    T = 1000

    env = CMAB(curve=curve)
    policies = [ZOOM(T=T), IGP_UCB(T=T), UMA(N=deg, T=T), OBlinUCB(basis = 'legendre', N=deg, T=T, pe=True), OBlinUCB(basis = 'legendre', N=deg, T=T)]
    labels = ['ZOOM', 'IGP_UCB', 'UMA', 'OB-PE', 'OB-LinUCB']

    make_experiment(policies, env, T, seeds, labels, exp_name=exp_name)

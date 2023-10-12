from functions.misc.merge_feature_maps import *
from functions.orthogonal.bases import *
from classes.environments.CMAB import CMAB
from classes.agents.OBlinUCB import OBlinUCB

import numpy as np


env = CMAB()
policy = OBlinUCB(basis = 'poly', N=3)

print(policy.pull_arm())


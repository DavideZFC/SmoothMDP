from functions.misc.merge_feature_maps import *
from functions.orthogonal.bases import *

import numpy as np


x = np.linspace(-1.,1.,10)
y = np.copy(x)

X, Y = np.meshgrid(x, y)
X = np.concatenate(X)
Y = np.concatenate(Y)

feature_map_x = sincos_features(X, d=3)
feature_map_y = sincos_features(Y, d=3)

global_feature = merge_feature_maps([feature_map_x, feature_map_y])

print(global_feature)
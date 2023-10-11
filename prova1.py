from functions.misc.merge_feature_maps import *
from functions.orthogonal.bases import *

import numpy as np


x = np.linspace(-1.,1.,10)


feature_map_x = poly_features(x, d=3)
print(feature_map_x)
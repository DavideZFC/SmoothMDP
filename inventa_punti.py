import numpy as np
from functions.misc.merge_feature_maps import merge_feature_maps
from functions.misc.build_mesh import build_mesh, generalized_build_mesh
from functions.orthogonal.bases import *
from functions.orthogonal.find_optimal_design import find_optimal_design

numel = 20
x = np.linspace(-1,1,numel)
y = np.linspace(1.5,2,numel)
z = np.linspace(-2,-1.5,numel)
points_list = generalized_build_mesh([x,y,z])

approx_degree = 5

feature_map = legendre_features

feature_maps = []
for i in range(3):
    feature_maps.append(feature_map(points_list[i], d=approx_degree))

full_feature_map = merge_feature_maps(feature_maps)
full_feature_map /= (full_feature_map.shape[1]**0.5)

print(np.sum(find_optimal_design(full_feature_map)>0))
import numpy as np
from functions.misc.merge_feature_maps import merge_feature_maps
from functions.misc.build_mesh import build_mesh, generalized_build_mesh
from functions.orthogonal.bases import *
from functions.orthogonal.find_optimal_design import find_optimal_design
from classes.environments.PendulumSimple import PendulumSimple


env = PendulumSimple()

state_action = np.zeros((4,3))
state_action[0,2] = 1.
next_state, reward = env.query_generator(state_action=state_action)
print(next_state)
print(reward)


numel = 10
x = np.linspace(-1,1,numel)
y = np.linspace(1.5,2,numel)
z = np.linspace(-2,-1.5,numel)
points_list = generalized_build_mesh([x,y,z])
points_vector = np.array(points_list).T
print(points_vector.shape)

approx_degree = 4

feature_map = legendre_features

feature_maps = []
for i in range(3):
    feature_maps.append(feature_map(points_list[i], d=approx_degree))

full_feature_map = merge_feature_maps(feature_maps)
full_feature_map /= (full_feature_map.shape[1]**0.5)

pi = find_optimal_design(full_feature_map)
nonzero = np.sum(pi > 0)

big_constant = 100
upper_bound = 2*nonzero + big_constant
fixed_design = np.zeros(int(upper_bound), dtype=int)
current_index = 0
for i in range(full_feature_map.shape[0]):
    if pi[i] > 0:
        times_to_pull = 1 + int(pi[i]*big_constant)
        fixed_design[current_index:current_index+times_to_pull] = i
        current_index += times_to_pull
fixed_design = fixed_design[:current_index]

query_features = np.zeros((current_index, full_feature_map.shape[1]))
query_points = np.zeros((current_index, 3))
for i in range(current_index):
    query_features[i, :] = full_feature_map[fixed_design[i], :]
    query_points[i, :] = points_vector[fixed_design[i], :]

# print(query_features.shape)
new_states, rewards = env.query_generator(query_points)
print(new_states.shape)
print(rewards.shape)



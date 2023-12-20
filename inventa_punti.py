from classes.environments.PendulumSimple import PendulumSimple
from classes.agents.FD_LSVI import FD_LSVI
import numpy as np

env = PendulumSimple()
agent = FD_LSVI(env)
agent.get_datasets(disc_numbers=[10,10,10])
next_w = agent.compute_w_step(next_w=0,last_step=True)
print(agent.get_best_future_q(state=-np.ones(2), next_w=next_w))





from classes.environments.PendulumSimple import PendulumSimple
from classes.agents.FD_LSVI import FD_LSVI

env = PendulumSimple()
agent = FD_LSVI(env)
agent.get_datasets(disc_numbers=[3,3,3])
print(agent.compute_w_step(next_w=0,last_step=True))





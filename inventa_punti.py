from classes.environments.PendulumSimple import PendulumSimple
from classes.agents.FD_LSVI import FD_LSVI
from functions.misc.test_algorithm_after_learning import test_algorithm

env = PendulumSimple()
agent = FD_LSVI(env)
state_disc = 40
action_disc = 20
agent.get_datasets(disc_numbers=[state_disc, state_disc, action_disc])
agent.compute_q_values()

test_algorithm(agent, env)






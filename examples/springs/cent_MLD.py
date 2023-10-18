import numpy as np
from env import SpringNetwork
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.envs import MonitorEpisodes
from plotting import SpringVizualizer

from dmpcpwa.agents.no_control_agent import NoControlAgent

np.random.seed(1)

PLOT = False
SAVE = True

n = 1  # num springs

sim_len = 50
env = MonitorEpisodes(TimeLimit(SpringNetwork(n), max_episode_steps=sim_len))

agent = NoControlAgent(
    n
)  # an agent who evaluates in the environment without any control action
agent.evaluate(env=env, episodes=1, seed=1)

if len(env.observations) > 0:
    X = env.observations[0].squeeze()
    U = env.actions[0].squeeze()
    R = env.rewards[0]
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)

print(f"Return = {sum(R.squeeze())}")

viz = SpringVizualizer()
viz.spring_sys_viz(X.T, U.reshape((n, sim_len)), rep=True)

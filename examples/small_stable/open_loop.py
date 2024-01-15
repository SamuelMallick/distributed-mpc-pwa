import numpy as np
from env import Network
from gymnasium.wrappers import TimeLimit
from model import get_inv_set
from mpcrl.wrappers.envs import MonitorEpisodes
from plotting import plot_system

from dmpcpwa.agents.mld_agent import MldAgent

N = 4  # controller horizon
n = 3

ep_len = N

u = np.array(
    [
        [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 0.3, -1, 0.2, -1, -0.1],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)


class OpenLoopAgent(MldAgent):
    counter = 0

    def get_control(self, state):
        u_c = u[:, [self.counter]]
        self.counter += 1
        return u_c


mpc = None
# env
env = MonitorEpisodes(
    TimeLimit(
        Network(),
        max_episode_steps=ep_len,
    )
)
agent = OpenLoopAgent(mpc)
# agent = NoControlAgent(3)
agent.evaluate(env=env, episodes=1, seed=1)

if len(env.observations) > 0:
    X = env.observations[0].squeeze()
    U = env.actions[0].squeeze()
    R = env.rewards[0]
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)

A, b = get_inv_set()
print(A @ X[[-1], 0:2].T - b)
plot_system(X, U)

import numpy as np
from env import Network
from gymnasium.wrappers import TimeLimit
from model import get_cent_system
from mpcrl.wrappers.envs import MonitorEpisodes
from scipy.linalg import block_diag

from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld import MpcMld

from plotting import plot_system

N = 9  # controller horizon
n = 3

ep_len = 20


class Cent_MPC(MpcMld):
    Q_x_l = np.array([[1, 0], [0, 1]])
    Q_u_l = 1 * np.array([[1]])
    Q_x = block_diag(*[Q_x_l] * n)
    Q_u = block_diag(*[Q_u_l] * n)

    def __init__(self, system: dict, N: int) -> None:
        super().__init__(system, N)

        obj = 0
        for k in range(N - 1):
            obj += (
                self.x[:, k] @ self.Q_x @ self.x[:, [k]]
                + self.u[:, k] @ self.Q_u @ self.u[:, [k]]
            )
        obj += self.x[:, N] @ self.Q_x @ self.x[:, [N]]

        self.mpc_model.addConstrs(self.x[i, [N]] <= 0.1 for i in range(0, 2*n, 2))


mpc = Cent_MPC(get_cent_system(), N)
# env
env = MonitorEpisodes(
    TimeLimit(
        Network(),
        max_episode_steps=ep_len,
    )
)
agent = MldAgent(mpc)
agent.evaluate(env=env, episodes=1, seed=1)

if len(env.observations) > 0:
    X = env.observations[0].squeeze()
    U = env.actions[0].squeeze()
    R = env.rewards[0]
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)

plot_system(X, U)
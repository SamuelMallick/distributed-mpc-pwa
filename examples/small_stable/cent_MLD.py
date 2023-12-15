import gurobipy as gp
import numpy as np
from env import Network
from gymnasium.wrappers import TimeLimit
from model import get_cent_system
from mpcrl.wrappers.envs import MonitorEpisodes
from plotting import plot_system
from scipy.linalg import block_diag

from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.agents.no_control_agent import NoControlAgent
from dmpcpwa.mpc.mpc_mld import MpcMld

N = 3  # controller horizon
n = 3

ep_len = 20


class Cent_MPC(MpcMld):
    Q_x_l = np.array([[1, 0], [0, 1]])
    Q_u_l = 1 * np.array([[1]])
    Q_x = block_diag(*[Q_x_l] * n)
    Q_u = block_diag(*[Q_u_l] * n)

    def __init__(self, system: dict, N: int) -> None:
        super().__init__(system, N, verbose=True)

        obj = 0
        for k in range(N):
            obj += (
                self.x[:, k] @ self.Q_x @ self.x[:, [k]]
                + self.u[:, k] @ self.Q_u @ self.u[:, [k]]
            )
        obj += self.x[:, N] @ self.Q_x @ self.x[:, [N]]

        self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)

        # A = np.array([[0.5796, -0.6926], [-0.4799, 0.8491], [-0.6981, 0.6399], [-0.2161, 1.1071], [0.3850, -1.7887], [-0.4644, 1.5762], [0.7095, -2.3776], [0.7176, -2.2292], [0.6583, -1.8403]])
        # b = np.array([[0.4294, 0.2208, 0.3211, 0.4192, 0.3104, 0.4086, 0.2930, 0.2683, 0.2448]]).T
        # self.mpc_model.addConstrs(A@self.x[i:i+2, [N]] <= b for i in range(0, 2*n, 2))


mpc = Cent_MPC(get_cent_system(), N)
# env
env = MonitorEpisodes(
    TimeLimit(
        Network(),
        max_episode_steps=ep_len,
    )
)
agent = MldAgent(mpc)
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

plot_system(X, U)

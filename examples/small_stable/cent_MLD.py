import gurobipy as gp
import numpy as np
from env import Network
from gymnasium.wrappers import TimeLimit
from model_3 import get_cent_system, get_cost_matrices, get_inv_set
from mpcrl.wrappers.envs import MonitorEpisodes
from plotting import plot_system
from scipy.linalg import block_diag

from dmpcpwa.agents.no_control_agent import NoControlAgent
from dmpcpwa.mpc.mpc_mld import MpcMld

N = 2  # controller horizon
n = 3

ep_len = 2


class Cent_MPC(MpcMld):
    Q_x_l, Q_u_l = get_cost_matrices()
    Q_x = block_diag(*[Q_x_l] * n)
    Q_u = block_diag(*[Q_u_l] * n)

    A, b = get_inv_set()

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

        self.mpc_model.addConstrs(
            self.A @ self.x[i : i + 2, [N]] <= self.b for i in range(0, 2 * n, 2)
        )


mpc = Cent_MPC(get_cent_system(), N)
# env
env = MonitorEpisodes(
    TimeLimit(
        Network(),
        max_episode_steps=ep_len,
    )
)
# agent = MldAgent(mpc)
agent = NoControlAgent(3)
agent.evaluate(env=env, episodes=1, seed=1)

if len(env.observations) > 0:
    X = env.observations[0].squeeze()
    U = env.actions[0].squeeze()
    R = env.rewards[0]
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)

print(f"cost = {sum(R)}")
plot_system(X, U)

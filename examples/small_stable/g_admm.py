import logging
import pickle
import sys
from typing import Literal

import casadi as cs
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from csnlp import Nlp
from dmpcrl.core.admm import g_map
from dmpcrl.mpc.mpc_admm import MpcAdmm
from env import Network
from gymnasium.wrappers import TimeLimit
from model import (
    get_A_c1,
    get_A_c2,
    get_adj,
    get_cent_system,
    get_inv_set,
    get_local_system,
)
from mpcrl.wrappers.agents import Log
from mpcrl.wrappers.envs import MonitorEpisodes
from plotting import plot_system
from scipy.linalg import block_diag

from dmpcpwa.agents.g_admm_coordinator import GAdmmCoordinator
from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.agents.no_control_agent import NoControlAgent
from dmpcpwa.mpc.mpc_mld import MpcMld
from dmpcpwa.mpc.mpc_switching import MpcSwitching

SAVE_WARM_START = False
if len(sys.argv) > 1:
    SAVE_WARM_START = int(sys.argv[1])

N = 7  # controller horizon
n = 3
nx_l = 2
nu_l = 1
Q_x_l = np.array([[1, 0], [0, 1]])
Q_u_l = 1 * np.array([[1]])

ep_len = 50

Adj = get_adj()
A_c1 = get_A_c1()
A_c2 = get_A_c2()
G_map = g_map(Adj)

# manually construct system descriptions and coupling
system = get_local_system()
systems = []  # list of systems, 1 for each agent
systems.append(system.copy())
Ac_i = [A_c1]
systems[0]["Ac"] = []
for i in range(len(system["S"])):
    systems[0]["Ac"] = systems[0]["Ac"] + [Ac_i]

systems.append(system.copy())
Ac_i = [A_c2, A_c2]
systems[1]["Ac"] = []
for i in range(len(system["S"])):
    systems[1]["Ac"] = systems[1]["Ac"] + [Ac_i]

systems.append(system.copy())
Ac_i = [A_c1]
systems[2]["Ac"] = []
for i in range(len(system["S"])):
    systems[2]["Ac"] = systems[2]["Ac"] + [Ac_i]

# terminal set
# A, b = get_inv_set()
A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
b = 0.3 * np.ones((4, 1))

w = 1000 * np.ones((1, b.shape[0]))  # penalty on slack vars for term set


class LocalMpc(MpcSwitching):
    rho = 0.5
    horizon = N

    def __init__(self, num_neighbours, my_index) -> None:
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)
        self.nx_l = nx_l
        self.nu_l = nu_l

        x, x_c = self.augmented_state(num_neighbours, my_index, size=nx_l)

        u, _ = self.action(
            "u",
            nu_l,
        )

        x_c_list = (
            []
        )  # store the bits of x that are couplings in a list for ease of access
        for i in range(num_neighbours):
            x_c_list.append(x_c[nx_l * i : nx_l * (i + 1), :])

        r = system["T"][0].shape[0]  # number of conditions when constraining a region
        self.set_dynamics(nx_l, nu_l, r, x, u, x_c_list)

        # normal constraints
        for k in range(N + 1):
            self.constraint(f"state_{k}", system["D"] @ x[:, [k]], "<=", system["E"])
            for x_n in x_c_list:
                self.constraint(
                    f"state_{x_n}_{k}", system["D"] @ x_n[:, [k]], "<=", system["E"]
                )
        for k in range(N):
            self.constraint(f"control_{k}", system["F"] @ u[:, [k]], "<=", system["G"])

        s, _, _ = self.variable(
            "s",
            (b.shape[0], 1),
            lb=0,
        )  # slack var for distance constraint
        self.constraint(f"terminal", A @ x[:, [N]], "<=", b + s)
        # for x_n in x_c_list:
        # self.constraint(f"terminal{x_n}", A @ x_n[:, [N]], "<=", b)

        self.set_local_cost(
            sum(
                x[:, k].T @ Q_x_l @ x[:, k] + u[:, k].T @ Q_u_l @ u[:, k]
                for k in range(N)
            )
            + x[:, N].T @ Q_x_l @ x[:, N]
            + w @ s
        )

        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": False,
            # "jit": True,
            # "jit_cleanup": True,
            "ipopt": {
                # "linear_solver": "ma97",
                # "linear_system_scaling": "mc19",
                # "nlp_scaling_method": "equilibration-based",
                "max_iter": 2000,
                "sb": "yes",
                "print_level": 0,
            },
        }
        self.init_solver(opts, solver="ipopt")


class Cent_MPC(MpcMld):
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

        self.mpc_model.addConstrs(
            A @ self.x[i : i + 2, [N]] <= b for i in range(0, 2 * n, 2)
        )

        # don't set the cost so we just find a feasible solution
        # self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)


class StableGAdmmCoordinator(GAdmmCoordinator):
    # terminal controllers for the four PWA regions touching origin
    K = [
        np.array([[-1.67, 2.39]]),
        np.array([[-2.14, 3.81]]),
        np.array([[-1.78, 4.44]]),
        np.array([[-1.43, 1.60]]),
    ]
    prev_x = [np.zeros((nx_l, N)) for i in range(n)]
    first_step = True

    def __init__(
        self,
        local_mpcs: list[MpcAdmm],
        local_fixed_parameters: list[dict],
        systems: list[dict],
        G: list[list[int]],
        Adj,
        rho: float,
        cent_mpc,
        warmstart: Literal["last", "last-successful"] = "last-successful",
        name: str = None,
    ) -> None:
        super().__init__(
            local_mpcs, local_fixed_parameters, systems, G, Adj, rho, warmstart, name
        )
        self.cent_mpc = cent_mpc

    def g_admm_control(self, state, warm_start=None):
        if self.first_step or self.prev_sol is None:
            u, info = self.cent_mpc.solve_mpc(state)
            warm_start = [info["u"][[i], :] for i in range(n)]
            self.first_step = False
            if SAVE_WARM_START:
                with open("examples/small_stable/u.pkl", "wb") as file:
                    # with open(f"examples\small_stable\u.pkl","wb") as file:
                    pickle.dump(warm_start, file)
        else:
            warm_start = [
                np.hstack((self.prev_sol[i][:, 1:], self.prev_sol[i][:, [-1]]))
                for i in range(n)
            ]

        # check if agents are in terminal set - if they are set terminal controller
        # break global state into local pieces
        x = [state[self.nx_l * i : self.nx_l * (i + 1), :] for i in range(self.n)]
        for i in range(n):
            if all(A @ x[i] <= b):
                pass
        return super().g_admm_control(state, warm_start)


# env
env = MonitorEpisodes(
    TimeLimit(
        Network(),
        max_episode_steps=ep_len,
    )
)
# cent mpc for initial feasible
cent_mpc = Cent_MPC(get_cent_system(), N)

# distributed mpcs and params
local_mpcs: list[LocalMpc] = []
local_fixed_dist_parameters: list[dict] = []
for i in range(n):
    local_mpcs.append(
        LocalMpc(
            num_neighbours=len(G_map[i]) - 1,
            my_index=G_map[i].index(i),
        )
    )
    local_fixed_dist_parameters.append(local_mpcs[i].fixed_pars_init)
# coordinator
agent = Log(
    StableGAdmmCoordinator(
        local_mpcs,
        local_fixed_dist_parameters,
        systems,
        G_map,
        Adj,
        local_mpcs[0].rho,
        cent_mpc,
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
)

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

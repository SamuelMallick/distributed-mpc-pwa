import logging

import casadi as cs
import numpy as np
from ACC_env import CarFleet
from ACC_model import ACC
from csnlp import Nlp
from dmpcrl.core.admm import g_map
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.agents import Log
from mpcrl.wrappers.envs import MonitorEpisodes
from plot_fleet import plot_fleet

from dmpcpwa.agents.g_admm_coordinator import GAdmmCoordinator
from dmpcpwa.mpc.mpc_switching import MpcSwitching

np.random.seed(1)

n = 2  # num cars
N = 5  # controller horizon
w = 1e4  # slack variable penalty

ep_len = 20  # length of episode (sim len)
Adj = np.zeros((n, n))  # adjacency matrix
if n > 1:
    for i in range(n):  # make it chain coupling
        if i == 0:
            Adj[i, i + 1] = 1
        elif i == n - 1:
            Adj[i, i - 1] = 1
        else:
            Adj[i, i + 1] = 1
            Adj[i, i - 1] = 1
else:
    Adj = np.zeros((1, 1))
G_map = g_map(Adj)

acc = ACC(ep_len, N)
nx_l = acc.nx_l
nu_l = acc.nu_l
system = acc.get_pwa_system()
Q_x_l = acc.Q_x_l
Q_u_l = acc.Q_u_l
sep = acc.sep
d_safe = acc.d_safe
leader_state = acc.get_leader_state()

# no state coupling here so all zeros
Ac = np.zeros((nx_l, nx_l))
systems = []  # list of systems, 1 for each agent
for i in range(n):
    systems.append(system.copy())
    # add the coupling part of the system
    Ac_i = []
    for j in range(n):
        if Adj[i, j] == 1:
            Ac_i.append(Ac)
    systems[i]["Ac"] = []
    for j in range(
        len(system["S"])
    ):  # duplicate it for each PWA region, as for this PWA system the coupling matrices do not change
        systems[i]["Ac"] = systems[i]["Ac"] + [Ac_i]


class LocalMpc(MpcSwitching):
    rho = 0.5
    horizon = N

    def __init__(self, num_neighbours, my_index, leader=False) -> None:
        """Instantiate inner switching MPC for admm for car fleet. If leader is true the cost uses the reference traj
        My index is used to pick out own state from the grouped coupling states.
        It should be passed in via the mapping G (G[i].index(i))"""

        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)
        self.nx_l = nx_l
        self.nu_l = nu_l

        x, x_c = self.augmented_state(num_neighbours, my_index, size=nx_l)

        u, _ = self.action(
            "u",
            nu_l,
        )

        s, _, _ = self.variable(
            "s", (1, N + 1), lb=0
        )  # slack var for distance constraint

        x_c_list = (
            []
        )  # store the bits of x that are couplings in a list for ease of access
        for i in range(num_neighbours):
            x_c_list.append(x_c[nx_l * i : nx_l * (i + 1), :])

        r = system["T"][0].shape[0]  # number of conditions when constraining a region
        self.set_dynamics(nx_l, nu_l, r, x, u, x_c_list)

        # normal constraints
        for k in range(N):
            self.constraint(f"state_{k}", system["D"] @ x[:, [k]], "<=", system["E"])
            self.constraint(f"control_{k}", system["F"] @ u[:, [k]], "<=", system["G"])

            # acceleration limits

            self.constraint(
                f"acc_{k}", x[1, [k + 1]] - x[1, [k]], "<=", acc.a_acc * acc.ts
            )
            self.constraint(
                f"de_acc_{k}", x[1, [k + 1]] - x[1, [k]], ">=", acc.a_dec * acc.ts
            )

            # safety constraints - if leader they are done later with parameters
            if not leader:
                self.constraint(
                    f"safety_{k}", x[0, [k]], "<=", x_c[0, [k]] - d_safe + s[:, [k]]
                )
        if not leader:
            self.constraint(
                f"safety_{N}", x[0, [N]], "<=", x_c[0, [N]] - d_safe + s[:, [N]]
            )

        # objective
        if leader:
            self.leader_traj = []
            for k in range(N + 1):
                temp = self.parameter(f"x_ref_{k}", (nx_l, 1))
                self.leader_traj.append(temp)
                self.fixed_pars_init[f"x_ref_{k}"] = np.zeros((nx_l, 1))
            self.set_local_cost(
                sum(
                    (x[:, [k]] - self.leader_traj[k]).T
                    @ Q_x_l
                    @ (x[:, [k]] - self.leader_traj[k])
                    + u[:, [k]].T @ Q_u_l @ u[:, [k]]
                    for k in range(N)
                )
                + (x[:, [N]] - self.leader_traj[N]).T
                @ Q_x_l
                @ (x[:, [N]] - self.leader_traj[N])
            )
        else:
            # following the agent ahead - therefore the index of the local state copy to track
            # is always the FIRST one in the local copies x_c
            self.set_local_cost(
                sum(
                    (x[:, [k]] - x_c[0:nx_l, [k]] - sep).T
                    @ Q_x_l
                    @ (x[:, [k]] - x_c[0:nx_l, [k]] - sep)
                    + u[:, [k]].T @ Q_u_l @ u[:, [k]]
                    # + w * s[:, [k]]
                    for k in range(N)
                )
                + (x[:, [N]] - x_c[0:nx_l, [N]] - sep).T
                @ Q_x_l
                @ (x[:, [N]] - x_c[0:nx_l, [N]] - sep)
                # + w * s[:, [N]]
            )

        # solver

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


class TrackingGAdmmCoordinator(GAdmmCoordinator):
    def on_timestep_end(self, env, episode: int, timestep: int) -> None:
        # time step starts from 1, so this will set the cost accurately for the next time-step
        self.set_leader_traj(leader_state[:, timestep : (timestep + N + 1)])
        return super().on_timestep_end(env, episode, timestep)

    def set_leader_traj(self, leader_traj):
        for k in range(N):  # we assume first agent is leader!
            self.agents[0].fixed_parameters[f"x_ref_{k}"] = leader_traj[:, [k]]

    def g_admm_control(self, state, warm_start=None):
        # set warm start for fleet: constant velocity
        warm_start = [
            acc.get_u_for_constant_vel(env.x[2 * i + 1, :]) * np.ones((nu_l, N))
            for i in range(n)
        ]
        return super().g_admm_control(state, warm_start)


# env
env = MonitorEpisodes(TimeLimit(CarFleet(acc, n, ep_len), max_episode_steps=ep_len))
# distributed mpcs and params
local_mpcs: list[LocalMpc] = []
local_fixed_dist_parameters: list[dict] = []
for i in range(n):
    if i == 0:
        local_mpcs.append(
            LocalMpc(
                num_neighbours=len(G_map[i]) - 1,
                my_index=G_map[i].index(i),
                leader=True,
            )
        )
    else:
        local_mpcs.append(
            LocalMpc(num_neighbours=len(G_map[i]) - 1, my_index=G_map[i].index(i))
        )
    local_fixed_dist_parameters.append(local_mpcs[i].fixed_pars_init)
# coordinator
agent = Log(
    TrackingGAdmmCoordinator(
        local_mpcs,
        local_fixed_dist_parameters,
        systems,
        G_map,
        Adj,
        local_mpcs[0].rho,
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 200},
)


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

plot_fleet(n, X, U, R, leader_state)

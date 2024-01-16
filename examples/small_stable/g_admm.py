import logging
import pickle
import sys
from typing import Literal

import casadi as cs
import gurobipy as gp
import numpy as np
from csnlp import Nlp
from dmpcrl.core.admm import g_map
from dmpcrl.mpc.mpc_admm import MpcAdmm
from env import Network
from gymnasium.wrappers import TimeLimit
from model_2 import (
    get_adj,
    get_cent_system,
    get_cost_matrices,
    get_inv_set,
    get_local_coupled_systems,
    get_local_system,
    get_terminal_K,
    get_warm_start,
)
from mpcrl.wrappers.agents import Log
from mpcrl.wrappers.envs import MonitorEpisodes
from plotting import plot_system
from scipy.linalg import block_diag

from dmpcpwa.agents.g_admm_coordinator import GAdmmCoordinator, PwaAgent
from dmpcpwa.mpc.mpc_mld import MpcMld
from dmpcpwa.mpc.mpc_switching import MpcSwitching

SAVE_WARM_START = False
CENT_WARM_START = True
MODEL_WARM_START = False
USE_TERM_CONTROLLER = True
if len(sys.argv) > 1:
    SAVE_WARM_START = int(sys.argv[1])
if len(sys.argv) > 2:
    CENT_WARM_START = int(sys.argv[2])

N = 5  # controller horizon
n = 3
nx_l = 2
nu_l = 1
Q_x_l, Q_u_l = get_cost_matrices()

ep_len = 30

Adj = get_adj()
G_map = g_map(Adj)

system = get_local_system()
systems = get_local_coupled_systems()

# terminal set
A, b = get_inv_set()

w = 100 * np.ones((1, b.shape[0]))  # penalty on slack vars for term set


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
            "s", (b.shape[0], 1), lb=0, ub=0
        )  # slack var for terminal constraint
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

        # store these to class to be reference later when u=Kx constraints are added
        self.K = [self.parameter(f"K_{k}", (nu_l, nx_l)) for k in range(N)]
        for k in range(N):
            self.fixed_pars_init[f"K_{k}"] = np.zeros((nu_l, nx_l))
        self.u = u
        self.x = x

        solver = "qrqp"
        if solver == "ipopt":
            opts = {
                "expand": True,
                "show_eval_warnings": True,
                "warn_initial_bounds": True,
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
                    "max_iter": 500,
                    "sb": "yes",
                    "print_level": 0,
                },
            }
        else:
            opts = {
                "expand": True,
                "print_time": False,
                "record_time": True,
                "error_on_fail": True,
                "print_info": False,
                "print_iter": False,
                "print_header": False,
                "max_iter": 2000,
            }
        
        self.init_solver(opts, solver=solver)


class Cent_MPC(MpcMld):
    Q_x = block_diag(*[Q_x_l] * n)
    Q_u = block_diag(*[Q_u_l] * n)

    def __init__(self, system: dict, N: int) -> None:
        super().__init__(system, N, verbose=True)

        obj = 0
        # for k in range(N):
        #    obj += (
        #        self.x[:, k] @ self.Q_x @ self.x[:, [k]]
        #        + self.u[:, k] @ self.Q_u @ self.u[:, [k]]
        #    )
        obj += self.x[:, N] @ self.Q_x @ self.x[:, [N]]

        self.mpc_model.addConstrs(
            A @ self.x[i : i + 2, [N]] <= b for i in range(0, 2 * n, 2)
        )
        self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)

        # so that Gurobi only searches for a feasbile sol
        self.mpc_model.setParam("SolutionLimit", 1)

        # limit threads to use cause the problem might be huuuuuuggggeeee
        self.mpc_model.setParam("Threads", 4)


class StableGAdmmCoordinator(GAdmmCoordinator):
    # terminal controllers for the four PWA regions touching origin
    K = get_terminal_K()
    term_flags = [False for i in range(n)]
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
        agent_class = PwaAgent,
        warmstart: Literal["last", "last-successful"] = "last-successful",
        name: str = None,
    ) -> None:
        super().__init__(
            local_mpcs, local_fixed_parameters, systems, G, Adj, rho, agent_class, warmstart, name
        )
        self.cent_mpc = cent_mpc
        for i in range(n):
            self.agents[i].set_K(self.K)

    def on_timestep_end(self, env, episode: int, timestep: int) -> None:
        return super().on_timestep_end(env, episode, timestep)

    def g_admm_control(self, state, warm_start=None):

        # check if agents are in terminal set - if they are set terminal controller
        # break global state into local pieces
        if USE_TERM_CONTROLLER:
            x = [state[self.nx_l * i : self.nx_l * (i + 1), :] for i in range(self.n)]
            for i in range(n):
                if all(A @ x[i] <= b):
                    if not self.term_flags[i]:
                        for k in range(N):
                            # set linear control constraint
                            self.agents[i].V.constraint(
                                f"term_cntrl_{k}",
                                self.agents[i].V.u[:, [k]],
                                "==",
                                self.agents[i].V.K[k] @ self.agents[i].V.x[:, [k]],
                            )
                        
                        self.term_flags[i] = True

            if all(self.term_flags):
                action_list = []
                x = [state[self.nx_l * i : self.nx_l * (i + 1), :] for i in range(self.n)]
                for i in range(n):
                    regions = self.agents[i].identify_regions(x[i], self.prev_sol[i][:, [-1]])
                    action_list.append(self.K[regions[0]]@x[i])
                return cs.DM(action_list), None, None, None
            

        # generate warm start for initial state
        if self.first_step or self.prev_sol is None:   
            if CENT_WARM_START:
                u, info = self.cent_mpc.solve_mpc(state)
                warm_start = [info["u"][[i], :] for i in range(n)]
                self.first_step = False
                if SAVE_WARM_START:
                    with open("examples/small_stable/u.pkl", "wb") as file:
                        # with open(f"examples\small_stable\u.pkl","wb") as file:
                        pickle.dump(warm_start, file)
            elif MODEL_WARM_START:
                warm_start = get_warm_start(N)
            else:
                warm_start = None
                self.first_step = False

        # generate warm start from either shifted solution or terminal controller
        else:
            prev_final_x = [self.prev_traj[i][:, [-1]] for i in range(n)]
            prev_final_u = [self.prev_sol[i][:, [-1]] for i in range(n)]
            warm_start = []
            for i in range(n):
                regions = self.agents[i].identify_regions(prev_final_x[i], prev_final_u[i])
                warm_start.append(np.hstack((self.prev_sol[i][:, 1:], self.K[regions[0]] @ prev_final_x[i])))

            if USE_TERM_CONTROLLER:
                # if any of the systems are now using terminal controllers, we generate a warm start that is u = Kx.
                # to do this we have to rollout the states using the other systems' warm starts
                if any(self.term_flags):
                    x_temp = [np.zeros((self.nx_l, self.N)) for i in range(self.n)]
                    for i in range(self.n):
                        x_temp[i][:, [0]] = state[self.nx_l * i : self.nx_l * (i + 1), [0]]  # add the first known states to the temp
                        if self.term_flags[i]:
                            regions = self.agents[i].identify_regions(x_temp[i][:, [0]], self.prev_sol[i][:, [0]])  # note here that the second argument does nothing
                            warm_start[i][:, [0]] = self.K[regions[0]] @ x_temp[i][:, [0]]

                    for k in range(1, self.N):
                        for i in range(self.n):
                            xc_temp = []
                            for j in range(self.n):
                                if self.Adj[i, j] == 1:
                                    xc_temp.append(x_temp[j][:, [k - 1]])
                            x_temp[i][:, [k]] = self.agents[i].next_state(
                                x_temp[i][:, [k - 1]], warm_start[i][:, [k - 1]], xc_temp
                            )
                            if self.term_flags[i]:
                                regions = self.agents[i].identify_regions(x_temp[i][:, [k]], self.prev_sol[i][:, [k]])  # note here that the second argument does nothing
                                warm_start[i][:, [k]] = self.K[regions[0]] @ x_temp[i][:, [k]]
                    
                    pass
        
        return super().g_admm_control(state, warm_start)

class PwaAgentTerminal(PwaAgent):
    K: list[np.ndarray] = []   # terminal controllers

    def set_K(self, K):
        self.K = K
    
    def set_sequence(self, s: list[int]):
        if len(self.K) == 0:
            raise RuntimeError("Linear controller must be set before sequence is set.")
        for i in range(len(s)):
            self.fixed_parameters[f"K_{i}"] = self.K[s[i]]
        return super().set_sequence(s)

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
        agent_class=PwaAgentTerminal,
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

print(f"cost = {sum(R)}")
plot_system(X, U)

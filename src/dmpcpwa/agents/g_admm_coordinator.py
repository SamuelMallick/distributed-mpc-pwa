import logging
from typing import Any, Literal

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from dmpcrl.core.admm import AdmmCoordinator
from dmpcrl.mpc.mpc_admm import MpcAdmm
from gymnasium import Env
from mpcrl import Agent
from mpcrl.agents.agent import ActType, ObsType

from dmpcpwa.agents.pwa_agent import PwaAgent

ADMM_DEBUG_PLOT = False
DEBUG_PRINT = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class GAdmmCoordinator(Agent):
    """Coordinates the greedy ADMM algorithm for PWA agents"""

    def __init__(
        self,
        local_mpcs: list[MpcAdmm],
        local_fixed_parameters: list[dict],
        systems: list[dict],
        G: list[list[int]],
        Adj: np.ndarray,
        rho: float,
        admm_iters: int = 50,
        switching_iters=float("inf"),
        agent_class=PwaAgent,
        warmstart: Literal["last", "last-successful"] = "last-successful",
        name: str = None,
    ) -> None:
        """Instantiates the coordinator, creating n PWA agents.

        Parameters
        ----------
        local_mpcs: list[MpcAdmm[SymType]]
            List of local MPCs for agents.
        local_fixed_parameters: List[dict]
            List of dictionaries for fixed parameters for agents.
        systems : dict
            PWA model for each agent.
        G :  List[List[int]]
            Map of local to global vars in ADMM.
        Adj: np.ndarray
            Adjacency matrix for agent coupling
        rho: float
            Augmented lagrangian penalty term."""

        # to the super class we pass the first local mpc just to satisfy the constructor
        # we copy it so the parameters don't double up etc.
        super().__init__(
            local_mpcs[0].copy(), local_fixed_parameters[0].copy(), warmstart, name
        )

        self.admm_iters = admm_iters
        self.switching_iters = switching_iters

        # construct the agents
        self.n = len(local_mpcs)
        self.agents: list[PwaAgent] = []
        for i in range(self.n):
            self.agents.append(
                agent_class(local_mpcs[i], local_fixed_parameters[i], systems[i])
            )

        # create ADMM coordinator
        self.N = local_mpcs[0].horizon
        self.Adj = Adj
        self.nx_l = local_mpcs[0].nx_l
        self.nu_l = local_mpcs[0].nu_l

        self.prev_sol = None
        self.prev_traj = None
        self.prev_sol_time = None

        # coordinator of ADMM using 1 iteration as g_admm coordinator checks sequences every ADMM iter
        self.admm_coordinator = AdmmCoordinator(
            self.agents,
            G,
            self.N,
            self.nx_l,
            self.nu_l,
            rho,
            iters=1,
        )

    def evaluate(
        self,
        env: Env[ObsType, ActType],
        episodes: int,
        deterministic: bool = True,
        seed: int = None,
        raises: bool = True,
        env_reset_options: dict[str, Any] = None,
    ):
        returns = np.zeros(episodes)
        self.on_validation_start(env)
        seeds = map(int, np.random.SeedSequence(seed).generate_state(episodes))

        for episode, current_seed in zip(range(episodes), seeds):
            self.reset(current_seed)
            for agent in self.agents:
                agent.reset(current_seed)
            state, _ = env.reset(seed=current_seed, options=env_reset_options)
            truncated, terminated, timestep = False, False, 0

            self.on_episode_start(env, episode, state)
            for agent in self.agents:
                agent.on_episode_start(env, episode, state)

            while not (truncated or terminated):
                action, sol_list, infeas_guess_flag, error_flag = self.g_admm_control(
                    state
                )

                if infeas_guess_flag or error_flag:
                    raise RuntimeError("G_admm infeasible or error.")

                state, r, truncated, terminated, _ = env.step(action)

                self.on_env_step(env, episode, timestep)
                for agent in self.agents:
                    agent.on_env_step(env, episode, timestep)

                returns[episode] += r
                timestep += 1

                self.on_timestep_end(env, episode, timestep)
                for agent in self.agents:
                    agent.on_timestep_end(env, episode, timestep)

            self.on_episode_end(env, episode, returns[episode])
            for agent in self.agents:
                agent.on_episode_end(env, episode, returns[episode])

        self.on_validation_end(env, returns)
        for agent in self.agents:
            agent.on_validation_end(env, returns)
        return returns

    def g_admm_control(self, state, warm_start=None):
        """Get the control for the given state. Warm start parameter is an initial guess for control actions."""
        seqs = [[0] * self.N for i in range(self.n)]  # switching seqs for agents

        xc = [None] * self.n
        x_pred = [None] * self.n
        action_list = None
        sol_list = None
        error_flag = False
        sol_time = 0.0

        # break global state into local pieces
        x = [state[self.nx_l * i : self.nx_l * (i + 1), :] for i in range(self.n)]

        if warm_start is None:
            warm_start = [np.zeros((self.nu_l, self.N)) for i in range(self.n)]

        infeas_flag = False
        u = warm_start

        # generate initial feasible coupling via dynamics rollout
        x_rout = self.dynamics_rollout(x, u)
        if x_rout is None:
            logger.debug(f"Rollout of initial control guess {u} was infeasible.")
            infeas_flag = True

        if ADMM_DEBUG_PLOT:  # store control at each iter to plot ADMM convergence
            u_plot_list = [
                [np.zeros((self.nu_l, self.N)) for k in range(self.admm_iters)]
                for i in range(self.n)
            ]
            switch_plot_list = [[] for i in range(self.n)]
            z_plot_list = [
                [np.zeros((self.nx_l, self.N)) for k in range(self.admm_iters)]
                for i in range(self.n)
            ]
            x_plot_list = [
                [np.zeros((self.nx_l, self.N)) for k in range(self.admm_iters)]
                for i in range(self.n)
            ]
            x_back_plot_list = [
                [np.zeros((self.nx_l, self.N)) for k in range(self.admm_iters)]
                for i in range(self.n)
            ]

        for iter in range(self.admm_iters):
            if infeas_flag:
                break
            logger.debug(f"Greedy admm iter {iter}")
            # generate local sequences and choose one  - CHOICE: this can be done with vars from local output of ADMM
            # which may not have converged to consensus - therefore adding exploration OR a cooperative
            # dynamics rollout as before the loop
            if iter < self.switching_iters:
                for i in range(self.n):
                    if iter == 0:  # first iter we must used rolled out state
                        new_seqs = self.agents[i].eval_sequences(
                            x[i],
                            u[i],
                            [x_rout[j] for j in range(self.n) if self.Adj[i, j] == 1],
                        )
                        seqs[i] = new_seqs[0]  # use first by default for first iter
                        logger.debug(f"Agent {i} initial sez: {seqs[i]}")
                    else:
                        new_seqs = self.agents[i].eval_sequences(
                            x[i],
                            u[i],
                            xc[i],  # use local ADMM vars if not first iter
                        )

                        if seqs[i] in new_seqs:
                            new_seqs.remove(seqs[i])
                        if len(new_seqs) > 0:
                            logger.debug(
                                f"Agent {i} switched: {seqs[i]} to {new_seqs[0]}"
                            )
                            seqs[i] = new_seqs[0]  # for now choosing arbritrarily first

                            if ADMM_DEBUG_PLOT:
                                switch_plot_list[i].append(iter)
                    # set sequences
                    self.agents[i].set_sequence(seqs[i])

            # perform ADMM step
            action_list, sol_list, error_flag = self.admm_coordinator.solve_admm(state)

            if not error_flag and all(
                ["t_wall_total" in sol_list[i].stats for i in range(self.n)]
            ):
                sol_time += max(
                    [sol_list[i].stats["t_wall_total"] for i in range(self.n)]
                )

            if not error_flag:
                if ADMM_DEBUG_PLOT:
                    for i in range(self.n):
                        u_plot_list[i][iter] = np.asarray(sol_list[i].vals["u"])
                        z_plot_list[i][iter] = self.admm_coordinator.z[
                            i * self.nx_l : (i + 1) * self.nx_l, :
                        ]
                        x_plot_list[i][iter] = np.asarray(sol_list[i].vals["x"])
                        if i != self.n - 1:
                            x_back_plot_list[i][iter] = np.asarray(
                                sol_list[i].vals["x_c"][-2:, :]
                            )

                # extract the vars across the horizon from the ADMM sol for each agent
                for i in range(self.n):
                    u[i] = np.asarray(sol_list[i].vals["u"])
                    x_pred[i] = np.asarray(sol_list[i].vals["x"])
                    xc_out = np.asarray(sol_list[i].vals["x_c"])
                    xc_temp = []
                    for j in range(self.agents[i].num_neighbours):
                        xc_temp.append(xc_out[self.nx_l * j : self.nx_l * (j + 1), :])
                    xc[i] = xc_temp
            else:
                break

        if ADMM_DEBUG_PLOT:
            self.plot_admm_iters(
                u_plot_list,
                z_plot_list,
                x_plot_list,
                x_back_plot_list,
                switch_plot_list,
            )

        if not error_flag and not infeas_flag:
            self.prev_sol = u
            self.prev_traj = x_pred
            self.prev_sol_time = sol_time

        return (
            cs.DM(action_list) if not infeas_flag and not error_flag else warm_start,
            sol_list,
            error_flag,
            infeas_flag,
        )

    def dynamics_rollout(self, x: list[np.ndarray], u: list[np.ndarray]):
        """For a given state and u, rollout the agents' dynamics step by step. Return None if the u is infeasible."""
        x_temp = [np.zeros((self.nx_l, self.N)) for i in range(self.n)]

        for i in range(self.n):
            x_temp[i][:, [0]] = x[i]  # add the first known states to the temp
        for k in range(1, self.N):
            for i in range(self.n):
                xc_temp = []
                for j in range(self.n):
                    if self.Adj[i, j] == 1:
                        xc_temp.append(x_temp[j][:, [k - 1]])
                next_state = self.agents[i].next_state(
                    x_temp[i][:, [k - 1]], u[i][:, [k - 1]], xc_temp
                )
                if next_state is None:
                    return None
                else:
                    x_temp[i][:, [k]] = next_state
        return x_temp

    def plot_admm_iters(self, u_list, z_list, x_list, x_back_list, switch_list):
        t = 1

        _, u_axs = plt.subplots(self.n, 1, constrained_layout=True, sharex=True)
        _, res_axs = plt.subplots(self.n, 1, constrained_layout=True, sharex=True)
        for i in range(len(u_list)):
            u_axs[i].plot([u_list[i][k][:, t] for k in range(len(u_list[i]))])
            u_axs[i].plot(
                switch_list[i], [u_list[i][k][0, t] for k in switch_list[i]], "o"
            )
            u_axs[i].set_ylabel(f"u {i}")
            u_axs[i].set_xlabel("admm iter")
            # self.u_axs[i].set_ylim((-1, 1))

        t = 1
        _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
        axs.plot([x_list[2][k][0, t] for k in range(len(u_list[i]))])
        axs.plot([x_back_list[1][k][0, t] for k in range(len(u_list[i]))])
        # axs.plot([z_list[2][k][0, t] for k in range(len(u_list[i]))])
        # for i in range(len(z_list)):
        #   axs[i].plot([x_list[i][k][1, t] for k in range(len(u_list[i]))])
        #   if i != 0:
        #       axs[i].plot(
        #           [x_back_list[i - 1][k][1, t] for k in range(len(u_list[i]))]
        #       )

        for i in range(len(z_list)):
            # res_axs[i].plot(
            #    [x_list[i][k][1, t] for k in range(len(u_list[i]))]
            # )
            res_axs[i].plot(
                [x_list[i][k][1, t] - z_list[i][k][1, t] for k in range(len(u_list[i]))]
            )
            # res_axs[i].plot(
            #    switch_list[i],
            #    [x_list[i][k][1, t] - z_list[i][k][1, t] for k in switch_list[i]],
            #    "o",
            # )
            res_axs[i].set_ylabel(f"residual {i}")
            res_axs[i].set_xlabel("admm iter")
            # self.res_axs[i].set_ylim((-1, 1))
        plt.show()

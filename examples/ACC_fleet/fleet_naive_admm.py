import sys

import gurobipy as gp
import numpy as np
from ACC_env import CarFleet
from ACC_model import ACC
from dmpcrl.core.admm import g_map
from gymnasium import Env
from gymnasium.wrappers import TimeLimit
from mpc_gear import MpcGear
from mpcrl.core.exploration import ExplorationStrategy, NoExploration
from mpcrl.wrappers.envs import MonitorEpisodes
from plot_fleet import plot_fleet

from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld import MpcMld

np.random.seed(3)

n = 2  # num cars
N = 7  # controller horizon
COST_2_NORM = True
DISCRETE_GEARS = False

if len(sys.argv) > 1:
    n = int(sys.argv[1])
if len(sys.argv) > 2:
    N = int(sys.argv[2])
if len(sys.argv) > 3:
    COST_2_NORM = bool(int(sys.argv[3]))
if len(sys.argv) > 4:
    DISCRETE_GEARS = bool(int(sys.argv[4]))

w = 1e4  # slack variable penalty
ep_len = 50  # length of episode (sim len)
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
Q_x_l = acc.Q_x_l
Q_u_l = acc.Q_u_l
sep = acc.sep
d_safe = acc.d_safe
leader_state = acc.get_leader_state()

large_num = 100000  # large number for dumby bounds on vars
rho = 0.5  # admm penalty
admm_iters = 10  # fixed number of iterations for ADMM routine


class LocalMpcADMM(MpcMld):
    def __init__(
        self, system: dict, N: int, leader: bool = False, trailer: bool = False
    ) -> None:
        super().__init__(system, N)
        self.setup_cost_and_constraints(self.u, leader, trailer)

    def setup_cost_and_constraints(self, u, leader=False, trailer=False):
        if COST_2_NORM:
            cost_func = self.min_2_norm
        else:
            cost_func = self.min_1_norm

        # vars for front and back car
        self.x_front = self.mpc_model.addMVar(
            (nx_l, N + 1), lb=-large_num, ub=large_num, name="x_front"
        )
        self.x_back = self.mpc_model.addMVar(
            (nx_l, N + 1), lb=-large_num, ub=-large_num, name="x_back"
        )

        self.s_front = self.mpc_model.addMVar(
            (1, N + 1), lb=0, ub=float("inf"), name="s_front"
        )

        self.s_back = self.mpc_model.addMVar(
            (1, N + 1), lb=0, ub=float("inf"), name="s_back"
        )

        # admm vars
        self.y_front = self.mpc_model.addMVar((nx_l, N + 1), lb=0, ub=0, name="y_front")

        self.y_back = self.mpc_model.addMVar((nx_l, N + 1), lb=0, ub=0, name="y_back")

        self.z_front = self.mpc_model.addMVar((nx_l, N + 1), lb=0, ub=0, name="z_front")

        self.z_back = self.mpc_model.addMVar((nx_l, N + 1), lb=0, ub=0, name="z_back")

        # setting these bounds to zero removes the slack var, as leader and trailer
        # dont have cars in front or behind respectively
        if leader:
            self.s_front.ub = 0
        if trailer:
            self.s_back.ub = 0

        obj = 0
        if leader:
            temp_sep = np.zeros((2, 1))
        else:
            temp_sep = sep

        for k in range(N):
            obj += cost_func(self.x[:, [k]] - self.x_front[:, [k]] - temp_sep, Q_x_l)
            obj += (
                cost_func(u[:, [k]], Q_u_l)
                + w * self.s_front[:, k]
                + w * self.s_back[:, k]
            )

            # accel cnstrs
            self.mpc_model.addConstr(
                self.x[1, [k + 1]] - self.x[1, [k]] <= acc.a_acc * acc.ts,
                name=f"acc_{k}",
            )
            self.mpc_model.addConstr(
                self.x[1, [k + 1]] - self.x[1, [k]] >= acc.a_dec * acc.ts,
                name=f"dec_{k}",
            )
            # safe distance constraints and admm terms
            if not leader:
                self.mpc_model.addConstr(
                    self.x[0, [k]] - self.s_front[:, [k]]
                    <= self.x_front[0, [k]] - d_safe,
                    name=f"safety_ahead_{k}",
                )

                obj += self.y_front[:, k] @ (
                    self.x_front[:, [k]] - self.z_front[:, [k]]
                )
                obj += (
                    (rho / 2)
                    * (self.x_front[:, k] - self.z_front[:, k])
                    @ np.eye(nx_l)
                    @ (self.x_front[:, [k]] - self.z_front[:, [k]])
                )

            if not trailer:
                self.mpc_model.addConstr(
                    self.x[0, [k]] + self.s_back[:, [k]]
                    >= self.x_back[0, [k]] + d_safe,
                    name=f"safety_behind_{k}",
                )

                obj += self.y_back[:, k] @ (self.x_back[:, [k]] - self.z_back[:, [k]])
                obj += (
                    (rho / 2)
                    * (self.x_back[:, k] - self.z_back[:, k])
                    @ np.eye(nx_l)
                    @ (self.x_back[:, [k]] - self.z_back[:, [k]])
                )

        obj += cost_func(self.x[:, [N]] - self.x_front[:, [N]] - temp_sep, Q_x_l)
        obj += +w * self.s_front[:, N] + w * self.s_back[:, N]

        if not leader:
            self.mpc_model.addConstr(
                self.x[0, [N]] - self.s_front[:, [N]] <= self.x_front[0, [N]] - d_safe,
                name=f"safety_ahead_{N}",
            )

            obj += self.y_front[:, N] @ (self.x_front[:, [N]] - self.z_front[:, [N]])
            obj += (
                (rho / 2)
                * (self.x_front[:, N] - self.z_front[:, N])
                @ np.eye(nx_l)
                @ (self.x_front[:, [N]] - self.z_front[:, [N]])
            )

        if not trailer:
            self.mpc_model.addConstr(
                self.x[0, [N]] + self.s_back[:, [N]] >= self.x_back[0, [N]] + d_safe,
                name=f"safety_behind_{N}",
            )

            obj += self.y_back[:, N] @ (self.x_back[:, [N]] - self.z_back[:, [N]])
            obj += (
                (rho / 2)
                * (self.x_back[:, N] - self.z_back[:, N])
                @ np.eye(nx_l)
                @ (self.x_back[:, [N]] - self.z_back[:, [N]])
            )

        self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)

    def set_front_vars(self, y_front, z_front):
        for k in range(N + 1):
            self.y_front[:, [k]].lb = y_front[:, [k]]
            self.y_front[:, [k]].ub = y_front[:, [k]]

            self.z_front[:, [k]].lb = z_front[:, [k]]
            self.z_front[:, [k]].ub = z_front[:, [k]]

    def set_back_vars(self, y_back, z_back):
        for k in range(N + 1):
            self.y_back[:, [k]].lb = y_back[:, [k]]
            self.y_back[:, [k]].ub = y_back[:, [k]]

            self.z_back[:, [k]].lb = z_back[:, [k]]
            self.z_back[:, [k]].ub = z_back[:, [k]]

    def set_x_front(self, x_front):
        for k in range(N + 1):
            self.x_front[:, [k]].lb = x_front[:, [k]]
            self.x_front[:, [k]].ub = x_front[:, [k]]


class ADMMCoordinator(MldAgent):
    def __init__(
        self,
        local_mpcs: list[MpcMld],
    ) -> None:
        """Initialise the coordinator.

        Parameters
        ----------
        local_mpcs: List[MpcMld]
            List of local MLD based MPCs - one for each agent.
        """
        self._exploration: ExplorationStrategy = (
            NoExploration()
        )  # to keep compatable with Agent class
        self.n = len(local_mpcs)
        self.agents: list[MldAgent] = []
        for i in range(self.n):
            self.agents.append(MldAgent(local_mpcs[i]))

        # admm vars
        self.y_front_list = [np.zeros((nx_l, N + 1)) for i in range(self.n)]
        self.y_back_list = [np.zeros((nx_l, N + 1)) for i in range(self.n)]
        self.z_list = [np.zeros((nx_l, N + 1)) for i in range(self.n)]

    def get_control(self, state):
        u = [None] * self.n

        # initial guess for coupling vars in admm comes from previous solutions #TODO timeshift with constant vel
        for i in range(self.n):
            if i != 0:  # first car has no car in front
                x_pred_ahead = self.agents[i - 1].get_predicted_state(shifted=True)
                if x_pred_ahead is not None:  # will be none on first time-step
                    self.agents[i].mpc.set_front_vars(
                        self.y_front_list[i], x_pred_ahead
                    )
            if i != n - 1:  # last car has no car behind
                x_pred_behind = self.agents[i + 1].get_predicted_state(shifted=True)
                if x_pred_behind is not None:  # will be none on first time-step
                    self.agents[i].mpc.set_back_vars(self.y_back_list[i], x_pred_behind)

        for t in range(admm_iters):
            # admm x-update
            for i in range(self.n):
                xl = state[nx_l * i : nx_l * (i + 1), :]  # pull out local part of state
                u[i] = self.agents[i].get_control(xl)

            # admm z-update and y-update together
            for i in range(self.n):
                if i == 0:
                    self.z_list[i] = (1.0 / 2.0) * (
                        self.agents[i].mpc.x.X + self.agents[i + 1].mpc.x_front.X
                    )
                    self.y_front_list[i + 1] += rho * (
                        self.agents[i + 1].mpc.x_front.X - self.z_list[i]
                    )
                elif i == n - 1:
                    self.z_list[i] = (1.0 / 2.0) * (
                        self.agents[i].mpc.x.X + self.agents[i - 1].mpc.x_back.X
                    )
                    self.y_back_list[i - 1] += rho * (
                        self.agents[i - 1].mpc.x_back.X - self.z_list[i]
                    )
                else:
                    self.z_list[i] = (1.0 / 3.0) * (
                        self.agents[i].mpc.x.X
                        + self.agents[i + 1].mpc.x_front.X
                        + self.agents[i - 1].mpc.x_back.X
                    )
                    self.y_front_list[i + 1] += rho * (
                        self.agents[i + 1].mpc.x_front.X - self.z_list[i]
                    )
                    self.y_back_list[i - 1] += rho * (
                        self.agents[i - 1].mpc.x_back.X - self.z_list[i]
                    )

            # update z and y for local agents
            for i in range(n):
                if i == 0:
                    self.agents[i].mpc.set_back_vars(
                        self.y_back_list[i], self.z_list[i + 1]
                    )
                elif i == n - 1:
                    self.agents[i].mpc.set_front_vars(
                        self.y_front_list[i], self.z_list[i - 1]
                    )
                else:
                    self.agents[i].mpc.set_front_vars(
                        self.y_front_list[i], self.z_list[i - 1]
                    )
                    self.agents[i].mpc.set_back_vars(
                        self.y_back_list[i], self.z_list[i + 1]
                    )

        return np.vstack(u)

    # here we set the leader cost because it is independent of other vehicles' states
    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        x_goal = leader_state[:, timestep : timestep + N + 1]
        self.agents[0].mpc.set_x_front(x_goal)
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: Env, episode: int) -> None:
        x_goal = leader_state[:, 0 : N + 1]
        self.agents[0].mpc.set_x_front(x_goal)
        return super().on_episode_start(env, episode)

# env
env = MonitorEpisodes(TimeLimit(CarFleet(acc, n, ep_len), max_episode_steps=ep_len))

if DISCRETE_GEARS:
    raise RuntimeError("Discrete gears not implemented")
else:
    mpc_class = LocalMpcADMM
    systems = [acc.get_pwa_system(i) for i in range(n)]
# coordinator
local_mpcs: list[MpcMld] = []
for i in range(n):
    # passing local system
    if i == 0:
        local_mpcs.append(mpc_class(systems[i], N, leader=True))
    elif i == n - 1:
        local_mpcs.append(mpc_class(systems[i], N, trailer=True))
    else:
        local_mpcs.append(mpc_class(systems[i], N))
agent = ADMMCoordinator(local_mpcs)

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
print(f"Violations = {env.unwrapped.viol_counter}")

plot_fleet(n, X, U, R, leader_state, violations=env.unwrapped.viol_counter[0])
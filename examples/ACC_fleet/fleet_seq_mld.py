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

np.random.seed(1)

n = 3  # num cars
N = 5  # controller horizon
COST_2_NORM = True
DISCRETE_GEARS = True

if len(sys.argv) > 1:
    n = int(sys.argv[1])
if len(sys.argv) > 2:
    N = int(sys.argv[2])
if len(sys.argv) > 3:
    COST_2_NORM = bool(sys.argv[3])

w = 1e4  # slack variable penalty
ep_len = 100  # length of episode (sim len)
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
full_system = acc.get_pwa_system()
friction_system = acc.get_friction_pwa_system()
Q_x_l = acc.Q_x_l
Q_u_l = acc.Q_u_l
sep = acc.sep
d_safe = acc.d_safe
leader_state = acc.get_leader_state()

large_num = 10000  # large number for dumby bounds on vars


class LocalMpcMld(MpcMld):
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
            (nx_l, N + 1), lb=large_num, ub=large_num, name="x_front"
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
            # safe distance constraints
            if not leader:
                self.mpc_model.addConstr(
                    self.x[0, [k]] - self.s_front[:, [k]]
                    <= self.x_front[0, [k]] - d_safe,
                    name=f"safety_ahead_{k}",
                )

            if not trailer:
                self.mpc_model.addConstr(
                    self.x[0, [k]] + self.s_back[:, [k]]
                    >= self.x_back[0, [k]] + d_safe,
                    name=f"safety_behind_{k}",
                )

        obj += cost_func(self.x[:, [N]] - self.x_front[:, [N]] - temp_sep, Q_x_l)
        obj += +w * self.s_front[:, N] + w * self.s_back[:, N]

        if not leader:
            self.mpc_model.addConstr(
                self.x[0, [N]] - self.s_front[:, [N]] <= self.x_front[0, [N]] - d_safe,
                name=f"safety_ahead_{N}",
            )

        if not trailer:
            self.mpc_model.addConstr(
                self.x[0, [N]] + self.s_back[:, [N]] >= self.x_back[0, [N]] + d_safe,
                name=f"safety_behind_{N}",
            )

        self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)

    def set_x_front(self, x_front):
        for k in range(N + 1):
            self.x_front[:, [k]].lb = x_front[:, [k]]
            self.x_front[:, [k]].ub = x_front[:, [k]]

    def set_x_back(self, x_back):
        for k in range(N + 1):
            self.x_back[:, [k]].lb = x_back[:, [k]]
            self.x_back[:, [k]].ub = x_back[:, [k]]


class LocalMpcGear(LocalMpcMld, MpcGear):
    def __init__(
        self, system: dict, N: int, leader: bool = False, trailer: bool = False
    ) -> None:
        MpcGear.__init__(self, system, N)
        self.setup_gears(N, acc, system["F"], system["G"])
        self.setup_cost_and_constraints(self.u_g, leader, trailer)


class TrackingSequentialMldCoordinator(MldAgent):
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

    def get_control(self, state):
        u = [None] * self.n
        for i in range(self.n):
            xl = state[nx_l * i : nx_l * (i + 1), :]  # pull out local part of state

            # get predicted state of car in front
            if i != 0:  # first car has no car in front
                x_pred_ahead = self.agents[i - 1].get_predicted_state(shifted=False)
                self.agents[i].mpc.set_x_front(x_pred_ahead)

            # get predicted state of car behind
            if i != n - 1:  # last car has no car behind
                x_pred_behind = self.agents[i + 1].get_predicted_state(shifted=True)
                if (
                    x_pred_behind is not None
                ):  # it will be None if first iteration and car behind has no saved solution
                    # apply a smart shifting, where the final position assumed a constant velocity
                    x_pred_behind[0, -1] = (
                        x_pred_behind[0, -2] + acc.ts * x_pred_behind[1, -1]
                    )

                    self.agents[i].mpc.set_x_back(x_pred_behind)

            u[i] = self.agents[i].get_control(xl)
        if DISCRETE_GEARS:
            # stack the continuous conttrol at the front and the discrete at the back
            return np.vstack(
                (
                    np.vstack([u[i][:nu_l, :] for i in range(n)]),
                    np.vstack([u[i][nu_l:, :] for i in range(n)]),
                )
            )
        else:
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
    mpc_class = LocalMpcGear
    system = friction_system
else:
    mpc_class = LocalMpcMld
    system = full_system
# coordinator
local_mpcs: list[MpcMld] = []
for i in range(n):
    # passing local system
    if i == 0:
        local_mpcs.append(mpc_class(system, N, leader=True))
    elif i == n - 1:
        local_mpcs.append(mpc_class(system, N, trailer=True))
    else:
        local_mpcs.append(mpc_class(system, N))
agent = TrackingSequentialMldCoordinator(local_mpcs)

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

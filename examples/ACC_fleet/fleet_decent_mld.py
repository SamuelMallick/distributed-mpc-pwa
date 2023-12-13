import pickle
import sys

import gurobipy as gp
import numpy as np
from ACC_env import CarFleet
from ACC_model import ACC
from dmpcrl.core.admm import g_map
from gymnasium import Env
from gymnasium.wrappers import TimeLimit
from mpcrl.core.exploration import ExplorationStrategy, NoExploration
from mpcrl.wrappers.envs import MonitorEpisodes
from mpcs.mpc_gear import MpcGear
from plot_fleet import plot_fleet

from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld import MpcMld

np.random.seed(2)

PLOT = False
SAVE = True

n = 5  # num cars
N = 5  # controller horizon
COST_2_NORM = True
DISCRETE_GEARS = False
HOMOGENOUS = True
LEADER_TRAJ = 2  # "1" - constant velocity leader traj. Vehicles start from random ICs. "2" - accelerating leader traj. Vehicles start in perfect platoon.

if len(sys.argv) > 1:
    n = int(sys.argv[1])
if len(sys.argv) > 2:
    N = int(sys.argv[2])
if len(sys.argv) > 3:
    COST_2_NORM = bool(int(sys.argv[3]))
if len(sys.argv) > 4:
    DISCRETE_GEARS = bool(int(sys.argv[4]))
if len(sys.argv) > 5:
    HOMOGENOUS = bool(int(sys.argv[5]))
if len(sys.argv) > 6:
    LEADER_TRAJ = int(sys.argv[6])

random_ICs = False
if LEADER_TRAJ == 1:
    random_ICs = True

CO_OP = False

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

acc = ACC(ep_len, N, leader_traj=LEADER_TRAJ)
nx_l = acc.nx_l
nu_l = acc.nu_l
Q_x_l = acc.Q_x_l
Q_u_l = acc.Q_u_l
Q_du_l = acc.Q_du_l
sep = acc.sep
d_safe = acc.d_safe
w = acc.w  # slack variable penalty
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
            if CO_OP and not trailer:
                obj += cost_func(self.x_back[:, [k]] - self.x[:, [k]] - temp_sep, Q_x_l)
            obj += (
                cost_func(u[:, [k]], Q_u_l)
                + w * self.s_front[:, k]
                + w * self.s_back[:, k]
            )
            if k < N - 1:
                obj += cost_func(u[:, [k + 1]] - u[:, [k]], Q_du_l)

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
        if CO_OP and not trailer:
            obj += cost_func(self.x_back[:, [N]] - self.x[:, [N]] - temp_sep, Q_x_l)
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


class TrackingDecentMldCoordinator(MldAgent):
    def __init__(self, local_mpcs: list[MpcMld], nx_l: int, nu_l: int) -> None:
        self._exploration: ExplorationStrategy = NoExploration()  # to keep compatable
        self.n = len(local_mpcs)
        self.nx_l = nx_l
        self.nu_l = nu_l
        self.agents: list[MldAgent] = []
        for i in range(self.n):
            self.agents.append(MldAgent(local_mpcs[i]))

        self.solve_times = np.zeros((ep_len, 1))
        self.node_counts = np.zeros((ep_len, 1))

    def get_control(self, state):
        u = [None] * self.n
        for i in range(self.n):
            xl = state[self.nx_l * i : self.nx_l * (i + 1), :]
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

    # current state of car to be tracked is observed and propogated forward
    # to be the prediction
    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        self.observe_states(env, timestep)

        # take the maximum solve time and maximum node count, as all agents are assumed to
        # have operated in parallel
        agent_solve_times = [self.agents[i].run_time for i in range(n)]
        agent_node_counts = [self.agents[i].node_count for i in range(n)]
        self.solve_times[env.step_counter - 1, :] = max(agent_solve_times)
        self.node_counts[env.step_counter - 1, :] = max(agent_node_counts)
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: Env, episode: int) -> None:
        self.observe_states(env, timestep=0)
        return super().on_episode_start(env, episode)

    def observe_states(self, env, timestep):
        for i in range(n):
            if i == 0:  # lead car
                x_pred_front = self.extrapolate_position(
                    leader_state[0, [timestep]], leader_state[1, [timestep]]
                )
                x_pred_back = self.extrapolate_position(
                    env.x[nx_l * (i + 1), :], env.x[nx_l * (i + 1) + 1, :]
                )
                self.agents[i].mpc.set_x_front(x_pred_front)
                self.agents[i].mpc.set_x_back(x_pred_back)
            elif i == n - 1:  # last car
                x_pred_front = self.extrapolate_position(
                    env.x[nx_l * (i - 1), :], env.x[nx_l * (i - 1) + 1, :]
                )
                self.agents[i].mpc.set_x_front(x_pred_front)
            else:
                x_pred_front = self.extrapolate_position(
                    env.x[nx_l * (i - 1), :], env.x[nx_l * (i - 1) + 1, :]
                )
                x_pred_back = self.extrapolate_position(
                    env.x[nx_l * (i + 1), :], env.x[nx_l * (i + 1) + 1, :]
                )
                self.agents[i].mpc.set_x_front(x_pred_front)
                self.agents[i].mpc.set_x_back(x_pred_back)

    def extrapolate_position(self, initial_pos, initial_vel):
        x_pred = np.zeros((nx_l, N + 1))
        x_pred[0, [0]] = initial_pos
        x_pred[1, [0]] = initial_vel
        for k in range(N):
            x_pred[0, [k + 1]] = x_pred[0, [k]] + acc.ts * x_pred[1, [k]]
            x_pred[1, [k + 1]] = x_pred[1, [k]]
        return x_pred


# env
env = MonitorEpisodes(
    TimeLimit(
        CarFleet(
            acc,
            n,
            ep_len,
            L2_norm_cost=COST_2_NORM,
            homogenous=HOMOGENOUS,
            random_ICs=random_ICs,
        ),
        max_episode_steps=ep_len,
    )
)

if DISCRETE_GEARS:
    mpc_class = LocalMpcGear
    if HOMOGENOUS:  # by not passing the index all systems are the same
        systems = [acc.get_friction_pwa_system() for i in range(n)]
    else:
        systems = [acc.get_friction_pwa_system(i) for i in range(n)]
else:
    mpc_class = LocalMpcMld
    if HOMOGENOUS:  # by not passing the index all systems are the same
        systems = [acc.get_pwa_system() for i in range(n)]
    else:
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
agent = TrackingDecentMldCoordinator(local_mpcs, nx_l, nu_l)

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
print(f"Run_times_sum: {sum(agent.solve_times)}")

if PLOT:
    plot_fleet(n, X, U, R, leader_state, violations=env.unwrapped.viol_counter[0])

if SAVE:
    with open(
        f"decent_n_{n}_N_{N}_Q_{COST_2_NORM}_DG_{DISCRETE_GEARS}_HOM_{HOMOGENOUS}_LT_{LEADER_TRAJ}"
        # + datetime.datetime.now().strftime("%d%H%M%S%f")
        + ".pkl",
        "wb",
    ) as file:
        pickle.dump(X, file)
        pickle.dump(U, file)
        pickle.dump(R, file)
        pickle.dump(agent.solve_times, file)
        pickle.dump(agent.node_counts, file)
        pickle.dump(env.unwrapped.viol_counter[0], file)
        pickle.dump(leader_state, file)

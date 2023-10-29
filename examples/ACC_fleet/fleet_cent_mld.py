import datetime
import pickle
import sys

import gurobipy as gp
import numpy as np
from ACC_env import CarFleet
from ACC_model import ACC
from gymnasium import Env
from gymnasium.wrappers import TimeLimit
from mpc_gear import MpcGear
from mpcrl.wrappers.envs import MonitorEpisodes
from plot_fleet import plot_fleet
from scipy.linalg import block_diag

from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld import MpcMld
from dmpcpwa.mpc.mpc_mld_cent_decup import MpcMldCentDecup

np.random.seed(1)

PLOT = True
SAVE = True

n = 3  # num cars
N = 10  # controller horizon
COST_2_NORM = False
DISCRETE_GEARS = False
HOMOGENOUS = True
LEADER_TRAJ = 1  # "1" - constant velocity leader traj. Vehicles start from random ICs. "2" - accelerating leader traj. Vehicles start in perfect platoon.

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

w = 1e4  # slack variable penalty
ep_len = 100  # length of episode (sim len)

acc = ACC(ep_len, N, leader_traj=LEADER_TRAJ)
nx_l = acc.nx_l
nu_l = acc.nu_l
Q_x_l = acc.Q_x_l
Q_u_l = acc.Q_u_l
sep = acc.sep
d_safe = acc.d_safe
leader_state = acc.get_leader_state()


class MPCMldCent(MpcMldCentDecup):
    def __init__(self, systems: list[dict], n, N: int) -> None:
        super().__init__(systems, n, N)
        self.setup_cost_and_constraints(self.u)

    def setup_cost_and_constraints(self, u):
        """Sets up the cost and constraints. Penalises the u passed in."""
        if COST_2_NORM:
            cost_func = self.min_2_norm
        else:
            cost_func = self.min_1_norm

        # slack vars for soft constraints
        self.s = self.mpc_model.addMVar((n, N + 1), lb=0, ub=float("inf"), name="s")

        # formulate cost
        # leader_traj gets changed and fixed by setting its bounds
        self.leader_traj = self.mpc_model.addMVar(
            (nx_l, N + 1), lb=0, ub=0, name="leader_traj"
        )
        obj = 0
        for i in range(n):
            local_state = self.x[nx_l * i : nx_l * (i + 1), :]
            local_control = u[nu_l * i : nu_l * (i + 1), :]
            if i == 0:
                # first car follows traj with no sep
                follow_state = self.leader_traj
                temp_sep = np.zeros((2, 1))
            else:
                # otherwise follow car infront (i-1)
                follow_state = self.x[nx_l * (i - 1) : nx_l * (i), :]
                temp_sep = sep
            for k in range(N):
                obj += cost_func(
                    local_state[:, [k]] - follow_state[:, [k]] - temp_sep, Q_x_l
                )
                obj += cost_func(local_control[:, [k]], Q_u_l) + w * self.s[i, [k]]
            obj += (
                cost_func(local_state[:, [N]] - follow_state[:, [N]] - temp_sep, Q_x_l)
                + w * self.s[i, [N]]
            )
        self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)

        # add extra constraints
        # acceleration constraints
        for i in range(n):
            for k in range(N):
                self.mpc_model.addConstr(
                    acc.a_dec * acc.ts
                    <= self.x[nx_l * i + 1, [k + 1]] - self.x[nx_l * i + 1, [k]],
                    name=f"dec_car_{i}_step{k}",
                )
                self.mpc_model.addConstr(
                    self.x[nx_l * i + 1, [k + 1]] - self.x[nx_l * i + 1, [k]]
                    <= acc.a_acc * acc.ts,
                    name=f"acc_car_{i}_step{k}",
                )

        # safe distance behind follower vehicle

        for i in range(n):
            local_state = self.x[nx_l * i : nx_l * (i + 1), :]
            if i != 0:  # leader isn't following another car
                follow_state = self.x[nx_l * (i - 1) : nx_l * (i), :]
                for k in range(N + 1):
                    self.mpc_model.addConstr(
                        local_state[0, [k]]
                        <= follow_state[0, [k]] - d_safe + self.s[i, [k]],
                        name=f"safe_dis_car_{i}_step{k}",
                    )

    def set_leader_traj(self, leader_traj):
        for k in range(N + 1):
            self.leader_traj[:, [k]].ub = leader_traj[:, [k]]
            self.leader_traj[:, [k]].lb = leader_traj[:, [k]]


class MpcGearCent(MPCMldCent, MpcMldCentDecup, MpcGear):
    def __init__(self, systems: list[dict], n: int, N: int) -> None:
        MpcMldCentDecup.__init__(self, systems, n, N)  # use the MpcMld constructor
        F = block_diag(
            *([systems[0]["F"]] * n)
        )  # here we are assuming that the F and G are the same for all systems
        G = np.vstack([systems[0]["G"]] * n)
        self.setup_gears(N, acc, F, G)
        self.setup_cost_and_constraints(self.u_g)


class TrackingMldAgent(MldAgent):
    def __init__(self, mpc: MpcMld) -> None:
        self.solve_times = np.zeros((ep_len, 1))
        self.node_counts = np.zeros((ep_len, 1))
        super().__init__(mpc)

    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        # time step starts from 1, so this will set the cost accurately for the next time-step
        self.mpc.set_leader_traj(leader_state[:, timestep : (timestep + N + 1)])
        self.solve_times[env.step_counter - 1, :] = self.run_time
        self.node_counts[env.step_counter - 1, :] = self.node_count
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: Env, episode: int) -> None:
        self.mpc.set_leader_traj(leader_state[:, 0 : N + 1])
        return super().on_episode_start(env, episode)


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
    if HOMOGENOUS:  # by not passing the index all systems are the same
        systems = [acc.get_friction_pwa_system() for i in range(n)]
    else:
        systems = [acc.get_friction_pwa_system(i) for i in range(n)]
    mpc = MpcGearCent(systems, n, N)
    mpc.set_leader_traj(leader_state[:, 0 : N + 1])
    agent = TrackingMldAgent(mpc)
else:
    # mld mpc
    if HOMOGENOUS:  # by not passing the index all systems are the same
        systems = [acc.get_pwa_system() for i in range(n)]
    else:
        systems = [acc.get_pwa_system(i) for i in range(n)]
    mld_mpc = MPCMldCent(systems, n, N)
    # initialise the cost with the first tracking point
    mld_mpc.set_leader_traj(leader_state[:, 0 : N + 1])
    agent = TrackingMldAgent(mld_mpc)

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
        f"cent_n_{n}_N_{N}_Q_{COST_2_NORM}_DG_{DISCRETE_GEARS}_HOM_{HOMOGENOUS}_LT_{LEADER_TRAJ}"
        + datetime.datetime.now().strftime("%d%H%M%S%f")
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

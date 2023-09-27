import gurobipy as gp
import numpy as np
from ACC_env import CarFleet
from ACC_model import ACC
from dmpcrl.core.admm import g_map
from gymnasium import Env
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.envs import MonitorEpisodes
from plot_fleet import plot_fleet

from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld import MpcMld
from dmpcpwa.utils.pwa_models import cent_from_dist

np.random.seed(1)

n = 2  # num cars
N = 3  # controller horizon
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
system = acc.get_pwa_system()
Q_x_l = acc.Q_x_l
Q_u_l = acc.Q_u_l
sep = acc.sep
d_safe = acc.d_safe
leader_state = acc.get_leader_state()

# construct centralised system
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

cent_system = cent_from_dist(systems, Adj)


class MPCMldCent(MpcMld):
    def __init__(self, system: dict, N: int) -> None:
        super().__init__(system, N)

        # slack vars for soft constraints
        self.s = self.mpc_model.addMVar((n, N + 1), lb=0, ub=float("inf"), name="s")

        # formulate cost
        # leader_traj gets changed and fixed by setting its bounds
        self.leader_traj = self.mpc_model.addMVar(
            (n, N + 1), lb=0, ub=0, name="leader_traj"
        ) 
        obj = 0
        for i in range(n):
            local_state = self.x[nx_l * i : nx_l * (i + 1), :]
            local_control = self.u[nu_l * i : nu_l * (i + 1), :]
            if i == 0:
                # first car follows traj with no sep
                follow_state = self.leader_traj
                for k in range(N):
                    obj += (
                        local_state[:, k] - follow_state[:, k] - np.zeros((1, 2))
                    ) @ Q_x_l @ (
                        local_state[:, [k]] - follow_state[:, [k]] - np.zeros((2, 1))
                    ) + local_control[
                        :, k
                    ] @ Q_u_l @ local_control[
                        :, [k]
                    ]
                obj += (
                    (local_state[:, N] - follow_state[:, N] - np.zeros((1, 2)))
                    @ Q_x_l
                    @ (local_state[:, [N]] - follow_state[:, [N]] - np.zeros((2, 1)))
                )
            else:
                # otherwise follow car infront (i-1)
                follow_state = self.x[nx_l * (i - 1) : nx_l * (i), :]
                for k in range(N):
                    obj += (
                        (local_state[:, k] - follow_state[:, k] - sep.T)
                        @ Q_x_l
                        @ (local_state[:, [k]] - follow_state[:, [k]] - sep)
                        + local_control[:, k] @ Q_u_l @ local_control[:, [k]]
                        + w * self.s[i, [k]]
                    )
                obj += (local_state[:, N] - follow_state[:, N] - sep.T) @ Q_x_l @ (
                    local_state[:, [N]] - follow_state[:, [N]] - sep
                ) + w * self.s[i, [N]]
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


class TrackingMldAgent(MldAgent):
    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        # time step starts from 1, so this will set the cost accurately for the next time-step
        self.mpc.set_leader_traj(leader_state[:, timestep : (timestep + N + 1)])
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: Env, episode: int) -> None:
        self.mpc.set_leader_traj(leader_state[:, 0 : N + 1])
        return super().on_episode_start(env, episode)


# env
env = MonitorEpisodes(TimeLimit(CarFleet(acc, n, ep_len), max_episode_steps=ep_len))
# mld mpc
mld_mpc = MPCMldCent(cent_system, N)
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

plot_fleet(n, X, U, R, leader_state, violations=env.unwrapped.viol_counter[0])

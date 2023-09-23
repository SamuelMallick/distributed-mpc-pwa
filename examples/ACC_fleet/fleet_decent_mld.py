import gurobipy as gp
import numpy as np
from ACC_env import CarFleet
from ACC_model import ACC
from dmpcrl.core.admm import g_map
from gymnasium import Env
from gymnasium.wrappers import TimeLimit
from mpcrl.core.exploration import ExplorationStrategy, NoExploration
from mpcrl.wrappers.envs import MonitorEpisodes
from plot_fleet import plot_fleet

from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld import MpcMld

np.random.seed(1)

n = 5  # num cars
N = 3  # controller horizon
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
system = acc.get_pwa_system()
Q_x_l = acc.Q_x_l
Q_u_l = acc.Q_u_l
sep = acc.sep
d_safe = acc.d_safe
leader_state = acc.get_leader_state()


class LocalMpcMld(MpcMld):
    def __init__(self, system: dict, N: int) -> None:
        super().__init__(system, N)

        # extra constraints
        self.s_ahead = self.mpc_model.addMVar(
            (1, N + 1), lb=0, ub=float("inf"), name="s_ahead"
        )
        self.s_behind = self.mpc_model.addMVar(
            (1, N + 1), lb=0, ub=float("inf"), name="s_behind"
        )
        self.safety_constraints_ahead = []
        self.safety_constraints_behind = []
        for k in range(N):
            # accel cnstrs
            self.mpc_model.addConstr(
                self.x[1, [k + 1]] - self.x[1, [k]] <= acc.a_acc * acc.ts,
                name=f"acc_{k}",
            )
            self.mpc_model.addConstr(
                self.x[1, [k + 1]] - self.x[1, [k]] >= acc.a_dec * acc.ts,
                name=f"dec_{k}",
            )
            # safe distance constraints - RHS updated each timestep by coordinator
            self.safety_constraints_ahead.append(  # for car in front
                self.mpc_model.addConstr(
                    self.x[0, [k]] - self.s_ahead[:, [k]] <= float("inf"),
                    name=f"safety_ahead_{k}",
                )
            )

            self.safety_constraints_behind.append(  # for car behind
                self.mpc_model.addConstr(
                    self.x[0, [k]] + self.s_behind[:, [k]] >= -float("inf"),
                    name=f"safety_behind_{k}",
                )
            )
        self.safety_constraints_ahead.append(
            self.mpc_model.addConstr(
                self.x[0, [N]] - self.s_ahead[:, [N]] <= float("inf"),
                name=f"safety_ahead_{N}",
            )
        )
        self.safety_constraints_behind.append(  # for car behind
            self.mpc_model.addConstr(
                self.x[0, [N]] + self.s_behind[:, [N]] >= -float("inf"),
                name=f"safety_behind_{k}",
            )
        )


class TrackingDecentMldCoordinator(MldAgent):
    def __init__(self, local_mpcs: list[MpcMld], nx_l: int, nu_l: int) -> None:
        self._exploration: ExplorationStrategy = NoExploration()  # to keep compatable
        self.n = len(local_mpcs)
        self.nx_l = nx_l
        self.nu_l = nu_l
        self.agents: list[MldAgent] = []
        for i in range(self.n):
            self.agents.append(MldAgent(local_mpcs[i]))

    def get_control(self, state):
        u = [None] * self.n
        for i in range(self.n):
            xl = state[self.nx_l * i : self.nx_l * (i + 1), :]
            u[i] = self.agents[i].get_control(xl)
        return np.vstack(u)

    # current state of car to be tracked is observed and propogated forward
    # to be the prediction
    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        self.observe_states(env, timestep)
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: Env, episode: int) -> None:
        self.observe_states(env, timestep=0)
        return super().on_episode_start(env, episode)

    def observe_states(self, env, timestep):
        for i in range(n):
            if i == 0:  # lead car
                x_pred_ahead = self.extrapolate_position(
                    leader_state[0, [timestep]], leader_state[1, [timestep]]
                )
                x_pred_behind = self.extrapolate_position(
                    env.x[nx_l * (i + 1), :], env.x[nx_l * (i + 1) + 1, :]
                )
            elif i == n - 1:  # last car
                x_pred_ahead = self.extrapolate_position(
                    env.x[nx_l * (i - 1), :], env.x[nx_l * (i - 1) + 1, :]
                )
                x_pred_behind = None, None
            else:
                x_pred_ahead = self.extrapolate_position(
                    env.x[nx_l * (i - 1), :], env.x[nx_l * (i - 1) + 1, :]
                )
                x_pred_behind = self.extrapolate_position(
                    env.x[nx_l * (i + 1), :], env.x[nx_l * (i + 1) + 1, :]
                )

            if i != 0:  # add front distance constraint if not front car
                for k in range(N + 1):
                    self.agents[i].mpc.safety_constraints_ahead[k].RHS = (
                        x_pred_ahead[0, [k]] - d_safe
                    )
            if i != n - 1:  # add beack distance constraint if not rear car
                for k in range(N + 1):
                    self.agents[i].mpc.safety_constraints_behind[k].RHS = (
                        x_pred_behind[0, [k]] + d_safe
                    )

            # set costs
            if i == 0:
                desired_sep = np.zeros((nx_l, 1))
            else:
                desired_sep = sep

            obj = 0
            for k in range(N):
                obj += (
                    (self.agents[i].mpc.x[:, k] - x_pred_ahead[:, k] - desired_sep.T)
                    @ Q_x_l
                    @ (
                        self.agents[i].mpc.x[:, [k]]
                        - x_pred_ahead[:, [k]]
                        - desired_sep
                    )
                    + self.agents[i].mpc.u[:, k] @ Q_u_l @ self.agents[i].mpc.u[:, [k]]
                    + w * self.agents[i].mpc.s_ahead[:, [k]]
                    + w * self.agents[i].mpc.s_behind[:, [k]]
                )
            obj += (
                (self.agents[i].mpc.x[:, N] - x_pred_ahead[:, N] - desired_sep.T)
                @ Q_x_l
                @ (self.agents[i].mpc.x[:, [N]] - x_pred_ahead[:, [N]] - desired_sep)
                + w * self.agents[i].mpc.s_ahead[:, [N]]
                + w * self.agents[i].mpc.s_behind[:, [N]]
            )
            self.agents[i].mpc.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)

    def extrapolate_position(self, initial_pos, initial_vel):
        x_pred = np.zeros((nx_l, N + 1))
        x_pred[0, [0]] = initial_pos
        x_pred[1, [0]] = initial_vel
        for k in range(N):
            x_pred[0, [k + 1]] = x_pred[0, [k]] + acc.ts * x_pred[1, [k]]
            x_pred[1, [k + 1]] = x_pred[1, [k]]
        return x_pred


# env
env = MonitorEpisodes(TimeLimit(CarFleet(acc, n, ep_len), max_episode_steps=ep_len))

# coordinator
local_mpcs: list[MpcMld] = []
for i in range(n):
    # passing local system
    local_mpcs.append(LocalMpcMld(system, N))
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

plot_fleet(n, X, U, R, leader_state, violations=env.unwrapped.viol_counter[0])

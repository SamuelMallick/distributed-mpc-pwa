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

n = 3  # num cars
N = 10  # controller horizon
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


class LocalMpcMld(MpcMld):
    def __init__(self, system: dict, N: int) -> None:
        super().__init__(system, N)

        # extra constraints
        self.s = self.mpc_model.addMVar((1, N + 1), lb=0, ub=float("inf"), name="s")
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
                    self.x[0, [k]] - self.s[:, [k]] <= float("inf"),
                    name=f"safety_ahead_{k}",
                )
            )

            self.safety_constraints_behind.append(  # for car behind
                self.mpc_model.addConstr(
                    self.x[0, [k]] + self.s[:, [k]] >= -float("inf"),
                    name=f"safety_behind_{k}",
                )
            )
        self.safety_constraints_ahead.append(
            self.mpc_model.addConstr(
                self.x[0, [N]] - self.s[:, [N]] <= float("inf"),
                name=f"safety_ahead_{N}",
            )
        )
        self.safety_constraints_behind.append(  # for car behind
            self.mpc_model.addConstr(
                self.x[0, [N]] + self.s[:, [N]] >= -float("inf"),
                name=f"safety_behind_{k}",
            )
        )


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
        nx_l: int
            Dimension of local state.
        nu_l: int
            Dimension of local control.
        Q_x_l: np.ndarray
            Quadratic penalty matrix for state tracking.
        Q_u_l: np.ndarray
            Quadratic penalty matrix for control effort.
        sep: np.ndarray
            Desired state seperation between tracked vehicles.
        d_safe: float
            Safe distance between vehicles.
        w: float
            Penalty on slack var s in cost.
        N: int
            Prediction horizon.
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

                # set distance constraint for car ahead
                for k in range(N + 1):
                    self.agents[i].mpc.safety_constraints_ahead[k].RHS = (
                        x_pred_ahead[0, [k]] - d_safe
                    )

                # set cost of tracking car ahead
                obj = 0
                for k in range(N):
                    obj += (
                        (self.agents[i].mpc.x[:, k] - x_pred_ahead[:, k] - sep.T)
                        @ Q_x_l
                        @ (self.agents[i].mpc.x[:, [k]] - x_pred_ahead[:, [k]] - sep)
                        + self.agents[i].mpc.u[:, k]
                        @ Q_u_l
                        @ self.agents[i].mpc.u[:, [k]]
                        + w * self.agents[i].mpc.s[:, [k]]
                    )
                obj += (
                    self.agents[i].mpc.x[:, N] - x_pred_ahead[:, N] - sep.T
                ) @ Q_x_l @ (
                    self.agents[i].mpc.x[:, [N]] - x_pred_ahead[:, [N]] - sep
                ) + w * self.agents[
                    i
                ].mpc.s[
                    :, [N]
                ]
                self.agents[i].mpc.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)

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

                    # set distance constraint for car ahead
                    for k in range(N + 1):
                        self.agents[i].mpc.safety_constraints_behind[k].RHS = (
                            x_pred_behind[0, [k]] + d_safe
                        )

            u[i] = self.agents[i].get_control(xl)
        return np.vstack(u)

    # here we set the leader cost because it is independent of other vehicles' states
    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        x_goal = leader_state[:, timestep : timestep + N + 1]
        self.agents[0].set_cost(Q_x_l, Q_u_l, x_goal)
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: Env, episode: int) -> None:
        x_goal = leader_state[:, 0 : N + 1]
        self.agents[0].set_cost(Q_x_l, Q_u_l, x_goal)
        return super().on_episode_start(env, episode)


# env
env = MonitorEpisodes(TimeLimit(CarFleet(acc, n, ep_len), max_episode_steps=ep_len))
# coordinator
local_mpcs: list[MpcMld] = []
for i in range(n):
    # passing local system
    local_mpcs.append(LocalMpcMld(system, N))
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

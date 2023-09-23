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
from dmpcpwa.utils.pwa_models import cent_from_dist

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

# construct semi-centralised syystem of 2 or 3 agents for local problems
# no state coupling here so all zeros
Ac = np.zeros((nx_l, nx_l))
systems = []
for i in range(n):
    temp_systems = []
    temp_systems.append(system.copy())
    temp_systems.append(system.copy())
    if i == 0 or i == n - 1:  # first and last agent have one neighbor, others have 2
        temp_Adj = np.array([[0, 0], [0, 0]])
    else:
        temp_Adj = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        temp_systems.append(system.copy())
    for j in range(len(temp_systems)):
        temp_systems[j]["Ac"] = []
        for k in range(
            len(system["S"])
        ):  # duplicate it for each PWA region, as for this PWA system the coupling matrices do not change
            temp_systems[j]["Ac"] = temp_systems[j]["Ac"] + []

    systems.append(cent_from_dist(temp_systems, temp_Adj))


class LocalMpc(MpcMld):
    """Mpc for a vehicle with a car in front and behind. Local state has car
    is organised with x = [x_front, x_me, x_back]."""

    def __init__(
        self, system: dict, N: int, pos_in_fleet: int, num_vehicles: int
    ) -> None:
        super().__init__(system, N)

        self.pos_in_fleet = pos_in_fleet
        self.num_vehicles = num_vehicles
        # the index of the local vehicles position is different if it is the leader or trailer
        if pos_in_fleet == 1:
            my_index = 0
            b_index = 2
            self.b_index = b_index
        elif pos_in_fleet == num_vehicles:
            my_index = 2
            f_index = 0
            self.f_index = f_index
        else:
            my_index = 2
            f_index = 0
            b_index = 4
            self.f_index = f_index
            self.b_index = b_index
        self.my_index = my_index

        # constraints and slacks for cars in front
        if pos_in_fleet > 1:
            self.s_front = self.mpc_model.addMVar(
                (1, N + 1), lb=0, ub=float("inf"), name="s_front"
            )
            for k in range(N + 1):
                self.mpc_model.addConstr(
                    self.x[f_index, [k]] - self.x[my_index, [k]]
                    >= d_safe - self.s_front[:, [k]],
                    name=f"safety_ahead_{k}",
                )
            if pos_in_fleet > 2:
                self.s_front_2 = self.mpc_model.addMVar(
                    (1, N + 1), lb=0, ub=float("inf"), name="s_front_2"
                )
                self.x_front_2 = self.mpc_model.addMVar(
                    (nx_l, N + 1), lb=0, ub=0, name="x_front_2"
                )
                for k in range(N + 1):
                    self.mpc_model.addConstr(
                        self.x_front_2[0, [k]] - self.x[f_index, [k]]
                        >= d_safe - self.s_front_2[:, [k]],
                        name=f"safety_ahead_2_{k}",
                    )
        if pos_in_fleet <= 2:  # leader and its follower
            self.ref_traj = self.mpc_model.addMVar(
                (nx_l, N + 1), lb=0, ub=0, name="ref_traj"
            )

        # constraints and slacks for cars in back
        if num_vehicles - pos_in_fleet >= 1:
            self.s_back = self.mpc_model.addMVar(
                (1, N + 1), lb=0, ub=float("inf"), name="s_back"
            )
            for k in range(N + 1):
                self.mpc_model.addConstr(
                    self.x[my_index, [k]] - self.x[b_index, [k]]
                    >= d_safe - self.s_back[:, [k]],
                    name=f"safety_back_{k}",
                )

            if num_vehicles - pos_in_fleet >= 2:
                self.s_back_2 = self.mpc_model.addMVar(
                    (1, N + 1), lb=0, ub=float("inf"), name="s_back_2"
                )
                # fixed state of car 2 back
                self.x_back_2 = self.mpc_model.addMVar(
                    (nx_l, N + 1), lb=0, ub=0, name="x_back_2"
                )

                for k in range(N + 1):
                    self.mpc_model.addConstr(
                        self.x[b_index, [k]] - self.x_back_2[0, [k]]
                        >= d_safe - self.s_back_2[:, [k]],
                        name=f"safety_back_2_{k}",
                    )

        # accel cnstrs
        for k in range(N):
            for i in range(self.u.shape[0]):
                self.mpc_model.addConstr(
                    self.x[2 * i + 1, [k + 1]] - self.x[2 * i + 1, [k]]
                    <= acc.a_acc * acc.ts,
                    name=f"acc_{i}_{k}",
                )
                self.mpc_model.addConstr(
                    self.x[2 * i + 1, [k + 1]] - self.x[2 * i + 1, [k]]
                    >= acc.a_dec * acc.ts,
                    name=f"dec_{i}_{k}",
                )

        # set local cost
        obj = 0
        # front position tracking portions of cost
        if pos_in_fleet > 1:
            for k in range(N + 1):
                obj += (
                    (
                        self.x[my_index : my_index + 2, k]
                        - self.x[f_index : f_index + 2, k]
                        - sep.T
                    )
                    @ Q_x_l
                    @ (
                        self.x[my_index : my_index + 2, [k]]
                        - self.x[f_index : f_index + 2, [k]]
                        - sep
                    )
                )
                +w * self.s_front[:, [k]]
            if pos_in_fleet > 2:
                for k in range(N + 1):
                    obj += (
                        (
                            self.x[f_index : f_index + 2, k]
                            - self.x_front_2[:, k]
                            - sep.T
                        )
                        @ Q_x_l
                        @ (
                            self.x[f_index : f_index + 2, [k]]
                            - self.x_front_2[:, [k]]
                            - sep
                        )
                    ) + w * self.s_front_2[:, [k]]
        if pos_in_fleet == 1:  # leader
            for k in range(N + 1):
                obj += (
                    (
                        self.x[my_index : my_index + 2, k]
                        - self.ref_traj[:, k]
                        - np.zeros((nx_l, 1)).T
                    )
                    @ Q_x_l
                    @ (
                        self.x[my_index : my_index + 2, [k]]
                        - self.ref_traj[:, [k]]
                        - np.zeros((nx_l, 1))
                    )
                )
        if pos_in_fleet == 2:  # follower of leader
            for k in range(N + 1):
                obj += (
                    (
                        self.x[f_index : f_index + 2, k]
                        - self.ref_traj[:, k]
                        - np.zeros((nx_l, 1)).T
                    )
                    @ Q_x_l
                    @ (
                        self.x[f_index : f_index + 2, [k]]
                        - self.ref_traj[:, [k]]
                        - np.zeros((nx_l, 1))
                    )
                )

        # slacks for vehicles behind
        if num_vehicles - pos_in_fleet >= 1:
            for k in range(N + 1):
                obj += (
                    (
                        self.x[b_index : b_index + 2, k]
                        - self.x[my_index : my_index + 2, k]
                        - sep.T
                    )
                    @ Q_x_l
                    @ (
                        self.x[b_index : b_index + 2, [k]]
                        - self.x[my_index : my_index + 2, [k]]
                        - sep
                    )
                ) + w * self.s_back[:, [k]]
            if num_vehicles - pos_in_fleet >= 2:
                for k in range(N + 1):
                    obj += (
                        (self.x_back_2[:, k] - self.x[b_index : b_index + 2, k] - sep.T)
                        @ Q_x_l
                        @ (
                            self.x_back_2[:, [k]]
                            - self.x[b_index : b_index + 2, [k]]
                            - sep
                        )
                    ) + w * self.s_back_2[:, [k]]

        # control penalty in cost
        for i in range(self.u.shape[0]):
            for k in range(N):
                obj += self.u[[i], k] @ Q_u_l @ self.u[i, [k]]

        self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)

    def set_leader_traj(self, ref_traj):
        for k in range(N + 1):
            self.ref_traj[:, [k]].lb = ref_traj[:, [k]]
            self.ref_traj[:, [k]].ub = ref_traj[:, [k]]

    def set_x_front_2(self, x_front_2):
        for k in range(N + 1):
            self.x_front_2[:, [k]].lb = x_front_2[:, [k]]
            self.x_front_2[:, [k]].ub = x_front_2[:, [k]]

    def set_x_back_2(self, x_back_2):
        for k in range(N + 1):
            self.x_back_2[:, [k]].lb = x_back_2[:, [k]]
            self.x_back_2[:, [k]].ub = x_back_2[:, [k]]

    def eval_cost(self, x, u):
        # set the bounds of the vars in the model to fix the vals
        for k in range(
            N
        ):  # we dont constain the N+1th state, as it is defined by shifted control
            self.x[:, [k]].ub = x[:, [k]]
            self.x[:, [k]].lb = x[:, [k]]
        self.u.ub = u
        self.u.lb = u
        self.IC.RHS = x[:, [0]]
        self.mpc_model.optimize()
        if self.mpc_model.Status == 2:  # check for successful solve
            cost = self.mpc_model.objVal
        else:
            cost = float("inf")  # infinite cost if infeasible
        return cost

    def solve_mpc(self, state):
        # the solve method is overridden so that the bounds on the vars are set back to normal before solving.
        self.x.ub = float("inf")
        self.x.lb = -float("inf")
        self.u.ub = float("inf")
        self.u.lb = -float("inf")
        return super().solve_mpc(state)


class TrackingEventBasedCoordinator(MldAgent):
    def __init__(
        self,
        local_mpcs: list[LocalMpc],
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

        # store control and state guesses
        self.state_guesses = [np.zeros((nx_l, N + 1)) for i in range(n)]
        self.control_guesses = [np.zeros((nu_l, N)) for i in range(n)]

    def get_control(self, state):
        [None] * self.n

        temp_costs = [None] * self.n
        for iter in range(10):
            best_cost_dec = -float("inf")
            best_idx = -1  # gets set to an agent index if there is a cost improvement
            for i in range(self.n):
                # get local initial condition and local initial guesses
                if i == 0:
                    x_l = state[nx_l * i : nx_l * (i + 2), :]
                    x_guess = np.vstack(
                        (self.state_guesses[i], self.state_guesses[i + 1])
                    )
                    u_guess = np.vstack(
                        (self.control_guesses[i], self.control_guesses[i + 1])
                    )
                elif i == n - 1:
                    x_l = state[nx_l * (i - 1) : nx_l * (i + 1), :]
                    x_guess = np.vstack(
                        (self.state_guesses[i - 1], self.state_guesses[i])
                    )
                    u_guess = np.vstack(
                        (self.control_guesses[i - 1], self.control_guesses[i])
                    )
                else:
                    x_l = state[nx_l * (i - 1) : nx_l * (i + 2), :]
                    x_guess = np.vstack(
                        (
                            self.state_guesses[i - 1],
                            self.state_guesses[i],
                            self.state_guesses[i + 1],
                        )
                    )
                    u_guess = np.vstack(
                        (
                            self.control_guesses[i - 1],
                            self.control_guesses[i],
                            self.control_guesses[i + 1],
                        )
                    )

                # set the constant predictions for neighbors of neighbors
                if i > 1:
                    self.agents[i].mpc.set_x_front_2(self.state_guesses[i - 2])
                if i < n - 2:
                    self.agents[i].mpc.set_x_back_2(self.state_guesses[i + 2])

                temp_costs[i] = self.agents[i].mpc.eval_cost(x_guess, u_guess)
                self.agents[i].get_control(x_l)
                new_cost = self.agents[i].get_predicted_cost()
                if temp_costs[i] - new_cost > best_cost_dec:
                    best_cost_dec = temp_costs[i] - new_cost
                    best_idx = i

            # update state and control guesses based on the winner
            if best_idx >= 0:
                best_x = self.agents[best_idx].x_pred
                best_u = self.agents[best_idx].u_pred
                if best_idx == 0:
                    self.state_guesses[0] = best_x[0:2, :]
                    self.state_guesses[1] = best_x[2:4, :]
                    self.control_guesses[0] = best_u[[0], :]
                    self.control_guesses[1] = best_u[[1], :]
                elif best_idx == n - 1:
                    self.state_guesses[n - 2] = best_x[0:2, :]
                    self.state_guesses[n - 1] = best_x[2:4, :]
                    self.control_guesses[n - 2] = best_u[[0], :]
                    self.control_guesses[n - 1] = best_u[[1], :]
                else:
                    self.state_guesses[best_idx - 1] = best_x[0:2, :]
                    self.state_guesses[best_idx] = best_x[2:4, :]
                    self.state_guesses[best_idx + 1] = best_x[4:6, :]
                    self.control_guesses[best_idx - 1] = best_u[[0], :]
                    self.control_guesses[best_idx] = best_u[[1], :]
                    self.control_guesses[best_idx + 1] = best_u[[2], :]

        return np.vstack([self.control_guesses[i][:, [0]] for i in range(n)])

    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        self.agents[0].mpc.set_leader_traj(leader_state[:, timestep : timestep + N + 1])
        self.agents[1].mpc.set_leader_traj(leader_state[:, timestep : timestep + N + 1])


        # shift previous solutions to be initial guesses at next step
        for i in range(n):
            self.state_guesses[i] = np.concatenate(
                (self.state_guesses[i][:, 1:], self.state_guesses[i][:, -1:]),
                axis=1,
            )
            self.control_guesses[i] = np.concatenate(
                (self.control_guesses[i][:, 1:], self.control_guesses[i][:, -1:]),
                axis=1,
            )
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: Env, episode: int) -> None:
        self.agents[0].mpc.set_leader_traj(leader_state[:, 0 : N + 1])
        self.agents[1].mpc.set_leader_traj(leader_state[:, 0 : N + 1])

        # initialise first step guesses with extrapolating positions
        for i in range(self.n):
            xl = env.x[
                nx_l * i : nx_l * (i + 1), :
            ]  # initial local state for vehicle i
            self.state_guesses[i] = self.extrapolate_position(xl[0, :], xl[1, :])
            self.control_guesses[i] = acc.get_u_for_constant_vel(xl[1, :]) * np.ones(
                (nu_l, N)
            )

        return super().on_episode_start(env, episode)

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
local_mpcs: list[LocalMpc] = []
for i in range(n):
    # passing local system
    local_mpcs.append(LocalMpc(systems[i], N, i + 1, n))
agent = TrackingEventBasedCoordinator(local_mpcs)

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

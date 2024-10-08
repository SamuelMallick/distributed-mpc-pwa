from typing import Any

import numpy as np
from csnlp.util.io import SupportsDeepcopyAndPickle
from gymnasium import Env
from mpcrl import Agent
from mpcrl.core.callbacks import AgentCallbackMixin
from mpcrl.core.exploration import ExplorationStrategy, NoExploration
from mpcrl.core.warmstart import WarmStartStrategy
from mpcrl.util.named import Named

from dmpcpwa.mpc.mpc_mld import MpcMld


class MldAgent(Agent):
    """A pwa agent who uses an mld controller."""

    def __init__(
        self,
        mpc: MpcMld,
    ) -> None:
        """Constructor is overriden and the super class' instructor is not called as
        this agent uses an mpc that does not inheret from the MPC baseclass."""

        # this below is just to keep this class compatible with Agent init
        Named.__init__(self, None)
        SupportsDeepcopyAndPickle.__init__(self)
        AgentCallbackMixin.__init__(self)
        self._fixed_pars = None

        # these params are just for compatibility with Agent class
        self._exploration: ExplorationStrategy = NoExploration()
        self._warmstart = WarmStartStrategy("last")

        self._store_last_successful = True
        self._last_action_on_fail = False
        self._last_solution = None
        self._last_action = None

        self.mpc = mpc
        self.x_pred = None  # stores most recent predicted state after being solved
        self.u_pred = None  # stores most recent predicted control after being solved
        self.cost_pred = None  # stores most recent predicted cost after being solved
        self.run_time = None  # stores most recent solve time of mpc
        self.node_count = None  # stores the node count from last MPC solution
        self.num_bin_vars = (
            None  # stors the number of binary variables in the model AFTER the presolve
        )

    def evaluate(
        self,
        env: Env,
        episodes: int,
        deterministic: bool = True,
        seed: int = None,
        raises: bool = True,
        env_reset_options: dict[str, Any] = None,
        open_loop: bool = False,
    ):
        """Evaluates the agent in a given environment. Overriding the function of Agent
        to use the mld_mpc instead.

        Parameters
        ----------
        env : Env[ObsType, ActType]
            A gym environment where to test the agent in.
        episodes : int
            Number of evaluation episodes.
        deterministic : bool, optional
            Whether the agent should act deterministically; by default, `True`.
        seed : None, int or sequence of ints, optional
            Agent's and each env's RNG seed.
        raises : bool, optional
            If `True`, when any of the MPC solver runs fails, or when an update fails,
            the corresponding error is raised; otherwise, only a warning is raised.
        env_reset_options : dict, optional
            Additional information to specify how the environment is reset at each
            evalution episode (optional, depending on the specific environment).

        Returns
        -------
        array of doubles
            The cumulative returns (one return per evaluation episode)."""
        returns = np.zeros(episodes)
        self.on_validation_start(env)
        seeds = map(int, np.random.SeedSequence(seed).generate_state(episodes))

        for episode, current_seed in zip(range(episodes), seeds):
            self.reset(current_seed)
            state, _ = env.reset(seed=current_seed, options=env_reset_options)
            truncated, terminated, timestep = False, False, 0
            self.on_episode_start(env, episode, state)

            if open_loop:
                _, info = self.get_control(state)
                actions = info["u"]
                counter = 0

            while not (truncated or terminated):
                # changed origonal agents evaluate here to use the mld mpc
                if not open_loop:
                    action, _ = self.get_control(state)
                else:
                    if counter > actions.shape[1]:
                        raise RuntimeError(
                            f"Open loop actions of length {actions.shape[1]} where not enough for episode."
                        )
                    action = actions[:, [counter]]
                    counter += 1

                state, r, truncated, terminated, _ = env.step(action)
                self.on_env_step(env, episode, timestep)

                returns[episode] += r
                timestep += 1
                self.on_timestep_end(env, episode, timestep)

            self.on_episode_end(env, episode, returns[episode])

        self.on_validation_end(env, returns)
        return returns

    def get_control(
        self, state, raises: bool = False, try_again_if_infeasible: bool = False
    ):
        u, info = self.mpc.solve_mpc(
            state, raises=raises, try_again_if_infeasible=try_again_if_infeasible
        )
        self.x_pred = info["x"]
        self.u_pred = info["u"]
        self.cost_pred = info["cost"]
        self.run_time = info["run_time"]
        self.node_count = info["nodes"]
        self.num_bin_vars = info["bin_vars"]
        return u, info

    def set_cost(self, Q_x, Q_u, x_goal: np.ndarray = None, u_goal: np.ndarray = None):
        """Set cost of the agents mpc-MIP as sum_k x(k)' * Q_x * x(k) + u(k)' * Q_u * u(k).
        Restricted to quadratic in the states and control.
        If x_goal or u_goal passed the cost uses (x-x_goal) and (u_goal)"""

        self.mpc.set_cost(Q_x, Q_u, x_goal, u_goal)

    def get_predicted_state(self, shifted=False) -> np.ndarray:
        """Returns the predicted state trajectory from the most recent solve of the local MPC.
        If shifted is true, the sequence is shifted by one, with the final time step duplicated.
        If no mpc has been solved, the returned value will be none."""
        if shifted and self.x_pred is not None:
            x_pred_shifted = self.x_pred.copy()
            return np.concatenate(
                (x_pred_shifted[:, 1:], x_pred_shifted[:, -1:]), axis=1
            )
        else:
            return self.x_pred

    def get_predicted_cost(self):
        """Return cost of most recent MPC solution."""
        return self.cost_pred

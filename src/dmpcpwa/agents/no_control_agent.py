from typing import Any

import casadi as cs
import numpy as np
from gymnasium import Env
from mpcrl import Agent


class NoControlAgent(Agent):
    """An agent who uses zero control input."""

    def __init__(self, nu: int, mpc) -> None:
        """Initialise the no control agent with control dimension nu."""
        self.nu = nu

        # to keep compatible
        super().__init__(mpc, {}, "last-succesful", None)

    def evaluate(
        self,
        env: Env,
        episodes: int,
        deterministic: bool = True,
        seed: int = None,
        raises: bool = True,
        env_reset_options: dict[str, Any] = None,
    ):
        """Evaluates the agent in a given environment. Overriding the evaluation to not
        use control.

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

            while not (truncated or terminated):
                # changed origonal agents evaluate here not control
                action = np.zeros((self.nu, 1))
                action = cs.DM(action)

                state, r, truncated, terminated, _ = env.step(action)
                self.on_env_step(env, episode, timestep)

                returns[episode] += r
                timestep += 1
                self.on_timestep_end(env, episode, timestep)

            self.on_episode_end(env, episode, returns[episode])

        self.on_validation_end(env, returns)
        return returns

from typing import Any

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from model import get_cent_spring_model
from scipy.linalg import block_diag


class SpringNetwork(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """A chain of PWA mass spring dampers."""

    Q_x_l = np.array([[1, 0], [0, 1]])
    Q_u_l = 0.01 * np.array([[1]])

    def __init__(self, n: int) -> None:
        self.n = n  # number of springs

        self.Q_x = block_diag(*[self.Q_x_l] * n)
        self.Q_u = block_diag(*[self.Q_u_l] * n)
        self.sys = get_cent_spring_model(n)

        super().__init__()

    def reset(
        self,
        *,
        seed: int = None,
        options: dict[str, Any] = None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Resets the state of the LTI system."""
        super().reset(seed=seed, options=options)
        self.x = np.zeros((2 * self.n, 1))
        self.x[0:2, :] = np.array([[2], [0]])
        return self.x, {}

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        """Computes the stage cost `L(s,a)`."""

        return state.T @ self.Q_x @ state + action.T @ self.Q_u @ action

    def step(
        self, action: cs.DM
    ) -> tuple[npt.NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        """Steps the fleet system."""

        action = action.full()
        r = self.get_stage_cost(self.x, action)

        x_new = None
        for i in range(len(self.sys["S"])):
            if all(
                self.sys["S"][i] @ self.x + self.sys["R"][i] @ action
                <= self.sys["T"][i]
            ):
                x_new = (
                    self.sys["A"][i] @ self.x
                    + self.sys["B"][i] @ action
                    + self.sys["c"][i]
                )
        if x_new is None:
            raise RuntimeError("No PWA region found for spring system.")
        self.x = x_new
        return x_new, r, False, False, {}

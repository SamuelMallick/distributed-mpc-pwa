from typing import Any

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from model import get_cent_system
from scipy.linalg import block_diag


class Network(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """A chain of PWA mass spring dampers."""

    Q_x_l = np.array([[1, 0], [0, 1]])
    Q_u_l = 1 * np.array([[1]])

    n = 3

    def __init__(self) -> None:
        self.Q_x = block_diag(*[self.Q_x_l] * self.n)
        self.Q_u = block_diag(*[self.Q_u_l] * self.n)
        self.sys = get_cent_system()

        super().__init__()

    def reset(
        self,
        *,
        seed: int = None,
        options: dict[str, Any] = None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Resets the state of the LTI system."""
        super().reset(seed=seed, options=options)
        #self.x = 1 * np.array([[-1.9, 1.2, 0.1, 1.6, 1.9, -1.6]]).T
        self.x = 1 * np.array([[-1.9, 0.7, -1.5, -1.3, 1.9, -1.4]]).T
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
            raise RuntimeError("No PWA region found for system.")
        self.x = x_new
        return x_new, r, False, False, {}

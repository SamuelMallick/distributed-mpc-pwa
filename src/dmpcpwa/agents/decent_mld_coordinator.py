import logging
from typing import Any, Collection, List, Literal, Optional, Sequence, Union

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from csnlp.wrappers import Mpc
from gymnasium import Env
from mpcrl import Agent
from mpcrl.agents.agent import ActType, ObsType, SymType
from mpcrl.core.exploration import ExplorationStrategy, NoExploration

from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.agents.pwa_agent import PwaAgent
from dmpcrl.core.admm import AdmmCoordinator
from dmpcrl.mpc.mpc_admm import MpcAdmm
from dmpcpwa.mpc.mpc_mld import MpcMld

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class DecentMldCoordinator(MldAgent):
    def __init__(self, local_mpcs: List[MpcMld], nx_l: int, nu_l: int) -> None:
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

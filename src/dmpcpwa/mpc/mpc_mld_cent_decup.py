import logging

import gurobipy as gp
import numpy as np

from dmpcpwa.mpc.mpc_mld import MpcMld

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.CRITICAL)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class MpcMldCentDecup(MpcMld):
    """ "An mpc that converts the networked PWA mpc problem to MLD form.
    The PWA systems are assumed decoupled in the dynamics."""

    def __init__(self, systems: list[dict], n: int, N: int, verbose=False, thread_limit: int | None = None, constrain_first_state = True) -> None:
        """Instantiate the mpc.

        Parameters
        ----------
        system: List[dict]
            List of dictionary which contain the definitions of the PWA systems {S, R, T, A, B, c, D, E, F, G}.
             When S[i]x+R[x]u <= T[i] -> x+ = A[i]x + B[i]u + c[i].
             For MLD conversion the state and input must be constrained: Dx <= E, Fu <= G.
        n: int
            The number of agents.
        N:int
            Prediction horizon length."""

        nx_l = systems[0]["A"][0].shape[0]
        nu_l = systems[0]["B"][0].shape[1]

        # build mld model

        mpc_model = gp.Model("mld_mpc")
        mpc_model.setParam("OutputFlag", verbose)
        mpc_model.setParam("Heuristics", 0)
        # mpc_model.setParam("Presolve", 0)
        # mpc_model.setParam('FeasibilityTol', 1e-3)
        if thread_limit is not None:
            mpc_model.params.threads = thread_limit

        # Uncomment if you need to differentiate between infeasbile and unbounded
        # mpc_model.setParam("DualReductions", 0)

        x = mpc_model.addMVar(
            (n * nx_l, N + 1), lb=-float("inf"), ub=float("inf"), name="x"
        )  # state
        u = mpc_model.addMVar(
            (n * nu_l, N), lb=-float("inf"), ub=float("inf"), name="u"
        )  # control

        self.delta: list[gp.MVar] = [] # store the binary region indicators for each agent
        for i in range(n):
            # pull out local parts for agent i
            x_l = x[nx_l * i : nx_l * (i + 1), :]
            u_l = u[nu_l * i : nu_l * (i + 1), :]

            self.delta.append(self.create_MLD_dynamics_and_constraints(systems[i], mpc_model, x_l, u_l, N, constrain_first_state=constrain_first_state))
        
        # IC constraint - gets updated everytime solve_mpc is called
        self.IC = mpc_model.addConstr(x[:, [0]] == np.zeros((n * nx_l, 1)), name="IC")

        # assign parts of model to be used by class later
        self.mpc_model = mpc_model
        self.x = x
        self.u = u
        self.m = n * nu_l
        self.N = N

        logger.critical("MLD MPC setup complete.")

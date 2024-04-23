import logging

import gurobipy as gp
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.CRITICAL)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class MpcMld:
    """An MPC that converts a PWA mpc problem into a MIP."""

    def __init__(self, system: dict, N: int, verbose=False, thread_limit: int | None = None) -> None:
        """Instantiate the mld based mpc. In the constructor pwa system is converted
        to mld and the associated dynamics and constraints are created, along with states
        and control variables.

        Parameters
        ----------
        system: dict
            Dictionary containing the definition of the PWA system {S, R, T, A, B, c, D, E, F, G}.
             When S[i]x+R[x]u <= T[i] -> x+ = A[i]x + B[i]u + c[i].
             For MLD conversion the state and input must be constrained: Dx <= E, Fu <= G.
        N:int
            Prediction horizon length."""

        n = system["A"][0].shape[0]
        m = system["B"][0].shape[1]

        # build mld model

        mpc_model = gp.Model("mld_mpc")
        mpc_model.setParam("OutputFlag", verbose)
        mpc_model.setParam("Heuristics", 0)
        mpc_model.setParam('FeasibilityTol', 1e-3)
        if thread_limit is not None:
            mpc_model.params.threads = thread_limit
        # mpc_model.setParam("MIPStart", 1)  # using warm-starting from previous sol

        # Uncomment if you need to differentiate between infeasbile and unbounded
        # mpc_model.setParam("DualReductions", 0)

        x = mpc_model.addMVar(
            (n, N + 1), lb=-float("inf"), ub=float("inf"), name="x"
        )  # state
        u = mpc_model.addMVar(
            (m, N), lb=-float("inf"), ub=float("inf"), name="u"
        )  # control

        # create MLD dynamics from PWA
        self.create_MLD_dynamics_and_constraints(system, mpc_model, x, u, N)

        # IC constraint - gets updated everytime solve_mpc is called
        self.IC = mpc_model.addConstr(x[:, [0]] == np.zeros((n, 1)), name="IC")

        # assign parts of model to be used by class later
        self.mpc_model = mpc_model
        self.x = x
        self.u = u
        self.n = n
        self.m = m
        self.N = N

        logger.critical("MLD MPC setup complete.")

    def create_MLD_dynamics_and_constraints(self, system, mpc_model, x, u, N):
        # extract values from system
        S = system["S"]
        R = system["R"]
        T = system["T"]
        A = system["A"]
        B = system["B"]
        c = system["c"]
        D = system["D"]
        E = system["E"]
        F = system["F"]
        G = system["G"]
        s = len(S)  # number of PWA regions
        n = A[0].shape[0]
        m = B[0].shape[1]

        M_st = [None] * s  # The upper bound for each region
        model_lin = gp.Model("linear model for mld set-up")
        model_lin.setParam("OutputFlag", 0)

        x_temp = model_lin.addMVar(
            (n, 1), lb=-float("inf"), ub=float("inf"), name="x_lin"
        )
        u_temp = model_lin.addMVar(
            (m, 1), lb=-float("inf"), ub=float("inf"), name="u_lin"
        )
        model_lin.addConstr(D @ x_temp <= E, name="state constraints")
        model_lin.addConstr(F @ u_temp <= G, name="control constraints")
        for i in range(s):
            obj = S[i] @ x_temp + R[i] @ u_temp - T[i]
            M_st[i] = np.zeros(obj.shape)
            for j in range(obj.shape[0]):
                model_lin.setObjective(obj[j, 0], gp.GRB.MAXIMIZE)
                model_lin.update()
                model_lin.optimize()
                M_st[i][j, 0] = model_lin.ObjVal
        logger.critical(
            "Solved linear model for PWA region bounds, M_star = " + str(M_st)
        )

        # bounds for state updates

        M_ub = np.zeros((n, 1))
        m_lb = np.zeros((n, 1))
        for j in range(n):
            M_regions = [None] * s
            m_regions = [None] * s
            for i in range(s):
                obj = A[i][j, :] @ x_temp + B[i][j, :] @ u_temp + c[i][j, :]
                model_lin.setObjective(obj, gp.GRB.MAXIMIZE)
                model_lin.update()
                model_lin.optimize()
                M_regions[i] = model_lin.ObjVal
                model_lin.setObjective(obj, gp.GRB.MINIMIZE)
                model_lin.update()
                model_lin.optimize()
                m_regions[i] = model_lin.ObjVal
            M_ub[j] = np.max(M_regions)
            m_lb[j] = np.min(m_regions)
        logger.critical(
            "Solved linear model for PWA state update bounds, M = "
            + str(M_ub.T)
            + "', m = "
            + str(m_lb.T)
            + "'"
        )

        # auxillary var z has 3 dimensions. (Region, state, time)
        z = mpc_model.addMVar((s, n, N), lb=-float("inf"), ub=float("inf"), name="z")
        # binary auxillary var
        delta = mpc_model.addMVar((s, N), vtype=gp.GRB.BINARY, name="delta")

        # constraint that only 1 delta can be active at each time step
        mpc_model.addConstrs(
            (gp.quicksum(delta[i, j] for i in range(s)) == 1 for j in range(N)),
            name="Delta sum constraints",
        )

        # constraints along predictions horizon for dynamics, state and control
        for k in range(N):
            # Set the branch priority of the binaries proportional to earlyness (earlier is more important)
            for i in range(s):
                # delta[i, k].setAttr('BranchPriority', s*N - s*k - i)
                # delta[i, k].setAttr('BranchPriority', N-k)
                pass

            # add state and input constraints to model, then binary and auxillary constraint, then dynamics constraints

            mpc_model.addConstr(D @ x[:, [k]] <= E, name="state constraints")
            mpc_model.addConstr(F @ u[:, [k]] <= G, name=f"control constraints_{k}")

            mpc_model.addConstrs(
                (
                    S[i] @ x[:, [k]] + R[i] @ u[:, [k]] - T[i]
                    <= M_st[i] * (1 - delta[i, [k]])
                    for i in range(s)
                ),
                name="Region constraints",
            )

            mpc_model.addConstrs(
                (z[i, :, k] <= M_ub @ (delta[i, [k]]) for i in range(s)),
                name="Z leq binary constraints",
            )

            mpc_model.addConstrs(
                (z[i, :, k] >= m_lb @ (delta[i, [k]]) for i in range(s)),
                name="Z geq binary constraints",
            )

            mpc_model.addConstrs(
                (
                    z[i, :, k].reshape(n, 1)
                    <= A[i] @ x[:, [k]]
                    + B[i] @ u[:, [k]]
                    + c[i]
                    - (m_lb * (1 - delta[i, [k]]))
                    for i in range(s)
                ),
                name="Z leq state constraints",
            )

            mpc_model.addConstrs(
                (
                    z[i, :, k].reshape(n, 1)
                    >= (A[i] @ x[:, [k]])
                    + (B[i] @ u[:, [k]])
                    + c[i]
                    - (M_ub * (1 - delta[i, [k]]))
                    for i in range(s)
                ),
                name="Z geq state constraints",
            )

            mpc_model.addConstr(
                x[:, [k + 1]]
                == gp.quicksum(z[i, :, k].reshape(n, 1) for i in range(s)),
                name="dynamics",
            )

        mpc_model.addConstr(
            D @ x[:, [N]] <= E, name="state constraints"
        )  # final state constraint

        # trivial terminal constraint condition x(N) = 0
        # mpc_model.addConstr(x[:, [N]] == np.zeros((n, 1)))

    def set_cost(self, Q_x, Q_u, x_goal: np.ndarray = None, u_goal: np.ndarray = None):
        """Set cost of the MIP as sum_k x(k)' * Q_x * x(k) + u(k)' * Q_u * u(k).
        Restricted to quadratic in the states and control.
        If x_goal or u_goal passed the cost uses (x-x_goal) and (u_goal)"""

        # construct zero goal points if not passed
        if x_goal is None:
            x_goal = np.zeros((self.x[:, [0]].shape[0], self.N + 1))
        if u_goal is None:
            u_goal = np.zeros((self.u[:, [0]].shape[0], self.N + 1))

        obj = 0
        for k in range(self.N):
            obj += (self.x[:, k] - x_goal[:, k].T) @ Q_x @ (
                self.x[:, [k]] - x_goal[:, [k]]
            ) + (self.u[:, k] - u_goal[:, k].T) @ Q_u @ (
                self.u[:, [k]] - u_goal[:, [k]]
            )
        obj += (
            (self.x[:, self.N] - x_goal[:, self.N].T)
            @ Q_x
            @ (self.x[:, [self.N]] - x_goal[:, [self.N]])
        )
        self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)

    def min_1_norm(self, x, Q=None):
        """returns a dummy variable t which should be minimised in the cost
        to substitute the minimization of |Qx| (1-norm). This functions will
        also add the required constraints on x to make the minimization of
        t equivalent to minimizing |Qx|. If Q not passed it is identity."""
        if x.shape[1] > 1:
            raise RuntimeError("x must be a columb vector.")
        if Q is None:
            Q = np.eye(x.shape[0])
        n = x.shape[0]
        t = self.mpc_model.addMVar((1, 1), lb=-float("inf"), ub=float("inf"), name="t")
        y = self.mpc_model.addMVar(
            x.shape, lb=-float("inf"), ub=float("inf"), name="y_dumb"
        )
        x = Q @ x
        for i in range(n):
            self.mpc_model.addConstr(x[i, :] <= y[i, :], name="dumb_leq")
            self.mpc_model.addConstr(-x[i, :] <= y[i, :], name="dumb_geq")
        self.mpc_model.addConstr(sum(y[i, :] for i in range(n)) == t, name="dumb_eq")
        return t

    def min_2_norm(self, x, Q):
        """return the term x'@Q@x."""
        # handle the zero matrix case individually as Gurobi is wierd with it
        if not Q.any(): 
            return 0
        
        # have to do the tranpose part manually because gurobi does not support
        # taking the transpose of an Expr
        n = x.shape[0]
        M = Q @ x
        return sum(x[i] * M[i] for i in range(n))

    def solve_mpc(self, state):
        self.IC.RHS = state
        self.mpc_model.optimize()
        if self.mpc_model.Status == 2:  # check for successful solve
            u = self.u.X
            x = self.x.X
            cost = self.mpc_model.objVal
        else:
            u = np.zeros((self.m, self.N))
            x = np.zeros(self.x.shape)
            cost = float("inf")
            logger.info("Infeasible")
            raise RuntimeError(f'Infeasible problem!')

        run_time = self.mpc_model.Runtime
        nodes = self.mpc_model.NodeCount
        try:
            bin_vars = self.mpc_model.presolve().NumBinVars
        except:
            bin_vars = 0

        self.mpc_model.NodeCount
        return u[:, [0]], {
            "x": x,
            "u": u,
            "cost": cost,
            "run_time": run_time,
            "nodes": nodes,
            "bin_vars": bin_vars,
        }

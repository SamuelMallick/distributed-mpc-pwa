import numpy as np
from csnlp.wrappers import Mpc
from mpcrl.agents.agent import Agent, SymType


class PwaAgent(Agent[SymType]):
    """An agent who has knowledge of it's own PWA dynamics and can use this to do things such as
    identify PWA regions given state and control trajectories."""

    def __init__(
        self,
        mpc: Mpc,
        fixed_parameters: dict,
        pwa_system: dict,
        use_terminal_sequence: bool = False
    ) -> None:
        """Initialise the agent.

        Parameters
        ----------
        mpc : Mpc[casadi.SX or MX]
            The MPC controller used as policy provider by this agent. The instance is
            modified in place to create the approximations of the state function `V(s)`
            and action value function `Q(s,a)`, so it is recommended not to modify it
            further after initialization of the agent. Moreover, some parameter and
            constraint names will need to be created, so an error is thrown if these
            names are already in use in the mpc. These names are under the attributes
            `perturbation_parameter`, `action_parameter` and `action_constraint`.
        fixed_parameters : dict[str, array_like] or collection of, optional
            A dict (or collection of dict, in case of `csnlp.MultistartNlp`) whose keys
            are the names of the MPC parameters and the values are their corresponding
            values. Use this to specify fixed parameters, that is, non-learnable. If
            `None`, then no fixed parameter is assumed.
        pwa_system: dict
            Contains {S, R, T, A, B, c, [Ac_j]}, where each is a list of matrices defining dynamics.
            When the inequality S[i]x + R[i]u <= T[i] is true, the dynamics are x^+ = A[i]x + B[i]u + c[i] + sum_j Ac_j[i] x_j.
        use_terminal_sequence: bool
            A flag to indicate if the local switching sequences are of length N or N+1. 
        """
        super().__init__(mpc, fixed_parameters)
        self.S = pwa_system["S"]
        self.R = pwa_system["R"]
        self.T = pwa_system["T"]
        self.A = pwa_system["A"]
        self.B = pwa_system["B"]
        self.c = pwa_system["c"]
        self.Ac = pwa_system["Ac"]

        self.num_neighbours = len(self.Ac[0])

        if use_terminal_sequence and not all(all(e == 0 for e in R) for R in self.R):
            raise RuntimeError("Cannot use a terminal sequence if the state switches based on the control input!")
        else:
            self.use_terminal_sequence = use_terminal_sequence

    def next_state(
        self,
        x: np.ndarray,
        u: np.ndarray,
        xc: list[np.ndarray] | None = None,
        eps=0,
    ):
        """Increment the dynamics as x+ = A[i]x + B[i]u + c[i] + sum_j Ac[i]_j xc_j
        if S[i]x + R[i]u <= T + eps.
        If the coupled states xc are not passed the coupling part of dynamics is ignored.
        If no PWA regions is found for the given x/u, None is returned.
        """

        next_state_options = (
            []
        )  # in case of boundary of PWA regions, there may be several next state options
        for i in range(len(self.S)):
            if all(self.S[i] @ x + self.R[i] @ u <= self.T[i] + eps):
                if xc is None:
                    return self.A[i] @ x + self.B[i] @ u + self.c[i]
                else:
                    next_state_options.append(
                        self.A[i] @ x
                        + self.B[i] @ u
                        + self.c[i]
                        + sum(self.Ac[i][j] @ xc[j] for j in range(self.num_neighbours))
                    )

        if len(next_state_options) == 0:
            # no PWA regions found for given x/u
            return None
        elif len(next_state_options) > 1:
            # given x/u is on boundary of PWA regions
            for i in range(1, len(next_state_options)):
                if np.linalg.norm(next_state_options[0] - next_state_options[i]) > 1e-5:
                    print(
                        "Warning: x/u pair provided on boundary of non-continuous PWA dynamics, arbritrarily selecting a PWA region."
                    )
            return next_state_options[0]
        else:
            return next_state_options[0]

    def next_state_with_region(
        self,
        s: int,
        x: np.ndarray,
        u: np.ndarray | None = None,
        xc: list[np.ndarray] | None = None,
    ):
        """Increment the dynamics as x+ = A[s]x + B[s]u + c[s] + sum_j Ac[s]_j xc_j
        If the coupled states xc are not passed the coupling part of dynamics is ignored.
        If control u is not passed the dynamics are assumed to switch depending only on x.
        """
        if u is None:
            if not all(all(e == 0 for e in R) for R in self.R):
                raise RuntimeError(
                    "No control input passed when PWA system switches based on control input!"
                )
            else:
                u = np.zeros((self.R[0].shape[1], 1))

        if xc is None:
            return self.A[s] @ x + self.B[s] @ u + self.c[s]
        else:
            return (
                self.A[s] @ x
                + self.B[s] @ u
                + self.c[s]
                + sum(self.Ac[s][j] @ xc[j] for j in range(self.num_neighbours))
            )

    def eval_sequences(
        self, x0: np.ndarray, u: np.ndarray, xc: list[np.ndarray] | None = None
    ):
        """Evaluate all possible switching sequences of PWA dynamics by rolling out
        dynamics from state x, applying control u, and coupled states xc.
        If the coupled states xc are not passed the coupling part of dynamics is ignored.
        """

        N = u.shape[1]  # horizon length
        s = [[0] * (N)] if not self.use_terminal_sequence else [[0] * (N+1)]  # list of sequences, start with just zeros
        x_list = [[x0]]  # list of state trajectories for each sequence

        for k in range(N):
            x_new = []  # new branches of state trajectories
            s_new = []  # new branches of switching sequences

            for i in range(len(x_list)):  # for each branched state sequence
                current_regions = self.identify_regions(x_list[i][-1], u[:, [k]])

                if (
                    len(current_regions) == 0
                ):  # if no regions found, cancel this branch, as is infeasible
                    x_list.pop(i)
                    s.pop(i)

                for j in range(len(current_regions)):
                    if j == 0:  # first one gets appended to all current branch
                        s[i][k] = current_regions[0]
                        if xc is None:
                            x_list[i].append(
                                self.next_state_with_region(
                                    current_regions[0], x_list[i][-1], u[:, [k]]
                                )
                            )
                        else:
                            x_list[i].append(
                                self.next_state_with_region(
                                    current_regions[0],
                                    x_list[i][-1],
                                    u[:, [k]],
                                    [xc[i][:, [k]] for i in range(self.num_neighbours)],
                                )
                            )
                    else:
                        # for other identified regions, they define new branches
                        s_temp = s[i].copy()
                        s_temp[k] = current_regions[j]
                        s_new.append(s_temp)

                        x_temp = x_list[i].copy()
                        if xc is None:
                            x_temp[k] = self.next_state_with_region(
                                current_regions[j], x_list[i][-1], u[:, [k]]
                            )
                        else:
                            x_temp[k] = self.next_state_with_region(
                                current_regions[j],
                                x_list[i][-1],
                                u[:, [k]],
                                [xc[i][:, [k]] for i in range(self.num_neighbours)],
                            )
                        x_new.append(x_temp)

                # combine new branches
            s = s + s_new
            x_list = x_list + x_new

        if self.use_terminal_sequence:
            for i in range(len(x_list)):    
                current_regions = self.identify_regions(x_list[i][-1])  # check region of final state
                for j in range(len(current_regions)):
                    if j == 0:
                        s[i][-1] = current_regions[0]
                    else:
                        s_temp = s[i].copy()
                        s_temp[-1] = current_regions[j]
                        s_new.append(s_temp)
            s = s + s_new

        # remove duplicates
        s_unique = []
        for seq in s:
            if seq not in s_unique:
                s_unique.append(seq)
        return s_unique

    def identify_regions(
        self, x: np.ndarray, u: np.ndarray | None = None, eps: float = 0
    ):
        """Generate the indices of the regions where Sx+Ru<=T + eps is true.
        If control u is not passed the dynamics are assumed to switch depending only on x.
        """
        if u is None:
            if not all(all(e == 0 for e in R) for R in self.R):
                raise RuntimeError(
                    "No control input passed when PWA system switches based on control input!"
                )
            else:
                u = np.zeros((self.R[0].shape[1], 1))

        regions = []
        for i in range(len(self.S)):
            if all(self.S[i] @ x + self.R[i] @ u <= self.T[i] + eps):
                regions.append(i)
        return regions

    def set_sequence(self, s: list[int]):
        """Modify the parameters in the constraints of the ADMM Mpc to
        enforce the sequence s"""

        # TODO confirm that these parameters have been named correctly in the MPC
        #
        for i in range(len(s)):
            self.fixed_parameters[f"A_{i}"] = self.A[s[i]]
            self.fixed_parameters[f"B_{i}"] = self.B[s[i]]
            self.fixed_parameters[f"c_{i}"] = self.c[s[i]]
            self.fixed_parameters[f"S_{i}"] = self.S[s[i]]
            self.fixed_parameters[f"R_{i}"] = self.R[s[i]]
            self.fixed_parameters[f"T_{i}"] = self.T[s[i]]
            for j in range(self.num_neighbours):
                self.fixed_parameters[f"Ac_{i}_{j}"] = self.Ac[s[i]][j]

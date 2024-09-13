from itertools import product

import numpy as np
from scipy.linalg import block_diag


def cent_from_dist(d_systems: list[dict], Adj: np.ndarray):
    """Creates a centralised representation of the distributed PWA system.
    PWA dynamics represented as: x+ = A[i]x + B[i]u + c[i] if S[i]x + R[i]u <= T.
    With state and control constraints Dx <= E, Fu <= G.

    Parameters
    ----------
    d_systems: List[dict]
        List of systems, each for an agent. Systems of form {S, R, T, A, B, c, D, E, F, G, [Aj]}.
    Adj: np.ndarray
        Adjacency matrix for system.
    """

    n = len(d_systems)  # num sub-systems
    nx_l = d_systems[0]["A"][0].shape[0]  # local x dim
    s_l = len(d_systems[0]["S"])  # num switching regions for each sub
    s = s_l**n  # num switching regions for centralised sys

    # each entry of the list contains a possible permutation of the PWA region indexes
    lst = [list(i) for i in product([j for j in range(s_l)], repeat=n)]

    S = []
    R = []
    T = []
    A = []
    B = []
    c = []

    # loop through every permutation
    for idxs in lst:
        S_tmp = []
        R_tmp = []
        T_tmp = []
        A_tmp = []
        B_tmp = []
        c_tmp = []
        for i in range(n):  # add the relevant region for each agent
            S_tmp.append(d_systems[i]["S"][idxs[i]])
            R_tmp.append(d_systems[i]["R"][idxs[i]])
            T_tmp.append(d_systems[i]["T"][idxs[i]])
            A_tmp.append(d_systems[i]["A"][idxs[i]])
            B_tmp.append(d_systems[i]["B"][idxs[i]])
            c_tmp.append(d_systems[i]["c"][idxs[i]])

        A_diag = block_diag(*A_tmp)
        # add coupling to A
        for i in range(n):
            coupling_idx = 0  # keep track of which Ac_i_j corresponds to Adj[i, j]
            for j in range(n):
                if Adj[i, j] == 1:
                    A_diag[nx_l * i : nx_l * (i + 1), nx_l * j : nx_l * (j + 1)] = (
                        d_systems[i]["Ac"][idxs[i]][coupling_idx]
                    )
                    coupling_idx += 1

        S.append(block_diag(*S_tmp))
        R.append(block_diag(*R_tmp))
        T.append(np.vstack(T_tmp))
        A.append(A_diag)
        B.append(block_diag(*B_tmp))
        c.append(np.vstack(c_tmp))

    # state and control constraints which don't switch by region
    D = block_diag(*[d_systems[i]["D"] for i in range(n)])
    E = np.vstack([d_systems[i]["E"] for i in range(n)])
    F = block_diag(*[d_systems[i]["F"] for i in range(n)])
    G = np.vstack([d_systems[i]["G"] for i in range(n)])

    return {
        "S": S,
        "R": R,
        "T": T,
        "A": A,
        "B": B,
        "c": c,
        "D": D,
        "E": E,
        "F": F,
        "G": G,
    }


def evalulate_cost(
    x: np.ndarray,
    u: np.ndarray,
    system: dict,
    Q_x: np.ndarray,
    Q_u: np.ndarray,
    seq: None | list[int] = None,
) -> float:
    """Evaluate the cost of a trajectory given a PWA system. The cost is assumed to be quadratic
    with x'Q_x*x + u'Q_u*u for each state and control pair plus x'Q_x*x for the final state. Initial state is propagated by the
    PWA dynamics. If a seq is provided, the switching sequence is used to determine the regions
    rather than the PWA inqeualities.

    Parameters
    ----------
    x: np.ndarray
        Initial state.
    u: np.ndarray
        Control trajectory.
    system: dict
        PWA system.
    Q_x: np.ndarray
        State cost matrix.
    Q_u: np.ndarray
        Control cost matrix.
    seq: None | list[int]
        Switching sequence.

    Returns
    -------
    float
        Cost of the trajectory.
    """
    x_traj = np.empty((x.shape[0], u.shape[1] + 1))
    cost = 0.0
    for k in range(u.shape[1]):
        x_traj[:, [k]] = x
        cost += x.T @ Q_x @ x + u[:, k].T @ Q_u @ u[:, k]
        if seq is not None:
            r = seq[k]
            if not all(
                system["S"][r] @ x + system["R"][r] @ u[:, [k]] <= system["T"][r] + 1e-6
            ):
                raise ValueError(
                    "Switching sequence provided does not satisfy PWA inequalities."
                )
        else:
            region_found = False
            for j in range(len(system["S"])):
                if all(
                    system["S"][j] @ x + system["R"][j] @ u[:, [k]] <= system["T"][j]
                ):
                    if region_found:
                        raise ValueError(
                            "Multiple regions found for the current state and input."
                        )
                    r = j
                    region_found = True
            if not region_found:
                raise ValueError("No region found for the current state and input.")
        x = system["A"][r] @ x + system["B"][r] @ u[:, [k]] + system["c"][r]
    return cost + x.T @ Q_x @ x


def evalulate_cost_distributed(
    x: np.ndarray | list[np.ndarray],
    u: np.ndarray | list[np.ndarray],
    systems: list[dict],
    adj: np.ndarray,
    Q_x: np.ndarray,
    Q_u: np.ndarray,
    seqs: None | list[list[int]] = None,
) -> float:
    """Evaluate the cost of a trajectory given a distributed PWA system. The cost is assumed to be quadratic
    with x'Q_x*x + u'Q_u*u for each state and control pair plus x'Q_x*x for the final state. Initial state is propagated by the
    PWA dynamics. If a seq is provided, the switching sequence is used to determine the regions
    rather than the PWA inqeualities.

    Parameters
    ----------
    x: np.ndarray | list[np.ndarray]
        Initial state. Either in centralized form or list of local states.
    u: np.ndarray | list[np.ndarray]
        Control trajectory. Either in centralized form or list of local controls.
    systems: list[dict]
        PWA systems.
    adj: np.ndarray
        Adjacency matrix.
    Q_x: np.ndarray
        State cost matrix, assumed to be local and the same for all agents.
    Q_u: np.ndarray
        Control cost matrix, assumed to be local and the same for all agents.
    seq: None | list[list[int]]
        Switching sequences.

    Returns
    -------
    float
        Cost of the trajectory.
    """
    n = len(systems)
    if isinstance(x, np.ndarray):
        x = np.split(x, n, axis=0)
    if isinstance(u, np.ndarray):
        u = np.split(u, n, axis=0)

    cost = 0.0
    for k in range(u[0].shape[1]):
        x_ = []
        for i in range(n):
            cost += x[i].T @ Q_x @ x[i] + u[i][:, k].T @ Q_u @ u[i][:, k]
            if seqs is not None:
                r = seqs[i][k]
                if not all(
                    systems[i]["S"][r] @ x[i] + systems[i]["R"][r] @ u[i][:, [k]]
                    <= systems[i]["T"][r] + 1e-6
                ):
                    raise ValueError(
                        "Switching sequence provided does not satisfy PWA inequalities."
                    )
            else:
                region_found = False
                for j in range(len(systems[i]["S"])):
                    if all(
                        systems[i]["S"][j] @ x[i] + systems[i]["R"][j] @ u[i][:, [k]]
                        <= systems[i]["T"][j]
                    ):
                        if region_found:
                            raise ValueError(
                                "Multiple regions found for the current state and input."
                            )
                        r = j
                        region_found = True
                if not region_found:
                    raise ValueError("No region found for the current state and input.")
            coupling = np.zeros((x[i].shape[0], 1))
            neighbor_count = 0
            for j in range(n):
                if adj[i, j] == 1:
                    coupling += systems[i]["Ac"][r][neighbor_count] @ x[j]
                    neighbor_count += 1
            x_.append(
                systems[i]["A"][r] @ x[i]
                + systems[i]["B"][r] @ u[i][:, [k]]
                + systems[i]["c"][r]
                + coupling
            )
        x = x_
    return cost + sum(x[i].T @ Q_x @ x[i] for i in range(n))

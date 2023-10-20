import itertools

import numpy as np
from scipy.linalg import block_diag

tau = 0.5  # sampling time for discretization
k1 = 10  # spring constant when one-sided spring active
k2 = 1  # spring constant when one-sided spring not active
damp = 4  # damping constant
mass = 10  # mass
l = 0  # displacement for masses to encounter 1-sided spring

A_spring_1 = np.array([[1, tau], [-((tau * 2 * k1) / mass), 1 - (tau * damp) / mass]])
A_spring_2 = np.array([[1, tau], [-((tau * 2 * k2) / mass), 1 - (tau * damp) / mass]])
B_spring = np.array([[0], [tau / mass]])
A_c = np.array([[0, 0], [(tau * k2) / mass, 0]])

x_lim_1_spring = 5
x_lim_2_spring = 5
u_lim_spring = 20


def get_spring_model():
    m = 1

    # Regions - 1 switch about l
    S = [np.array([[1, 0]]), np.array([[-1, 0]])]
    R = [np.zeros((1, m)), np.zeros((1, m))]
    T = [np.array([[l]]), np.array([[-l]])]

    # Dynamics - the spring system is connected to two springs
    A = [A_spring_1, A_spring_2]
    B = [B_spring, B_spring]
    c = [np.array([[0], [0]]), np.array([[0], [0]])]

    # Constraints
    x1_lim = x_lim_1_spring  # Displacement constraint
    x2_lim = x_lim_2_spring  # Velocity constraint
    u_lim = u_lim_spring  # Input constraint
    D = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    E = np.array([[x1_lim], [x1_lim], [x2_lim], [x2_lim]])
    F = np.array([[1], [-1]])
    G = np.array([[u_lim], [u_lim]])

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


def get_cent_spring_model(N: int = 2):
    n = 2 * N
    m = N
    s = 2**N

    l = 0  # Displacement for masses to encounter 1-sided spring

    # List contains the combinations we need to define the regions
    lst = [list(i) for i in itertools.product([0, 1], repeat=N)]

    # Regions
    S = []
    T = []

    # Dynamics
    A = []
    B = []
    c = []

    A1 = A_spring_1
    A2 = A_spring_2
    B1 = B_spring

    for i in range(s):
        S_temp = []
        T_temp = []
        A_temp = []
        for j in range(N):
            if lst[i][j] == 1:
                S_temp.append(np.array([[1, 0]]))
                T_temp.append(l)
                A_temp.append(A1)
            else:
                S_temp.append(np.array([[-1, 0]]))
                T_temp.append(-l)
                A_temp.append(A2)
        S.append(block_diag(*S_temp))
        T.append(np.asarray(T_temp).reshape(N, 1))
        A_blck = block_diag(*A_temp)  # Block diagonal each systems dynamics
        aux = np.empty((0, 2), int)  # Auxillary matrix to get offdiagonal coupling
        A_full = (
            A_blck
            + block_diag(aux.T, *[A_c] * (N - 1), aux)
            + block_diag(aux, *[A_c] * (N - 1), aux.T)
        )
        A.append(A_full)
        B.append(block_diag(*[B1] * N))

    # c is zero. This system is actually not continuos PWA unless l = 0,
    # as otehrwise the velocity will jump instantaneously
    c = [np.zeros((n, 1))] * s
    R = [np.zeros((N, m))] * s

    # Constraints
    x1_lim = x_lim_1_spring  # Displacement constraint
    x2_lim = x_lim_2_spring  # Velocity constraint
    u_lim = u_lim_spring  # Input constraint
    D = np.concatenate((np.eye(n), -np.eye(n)))
    E = np.zeros((2 * n, 1))
    for i in range(2 * n):
        if (i % 2) == 0:  # Check if even (if its position state or vel state)
            E[i, 0] = x1_lim
        else:
            E[i, 0] = x2_lim
    F = np.concatenate((np.eye(m), -np.eye(m)))
    G = u_lim * np.ones((2 * m, 1))

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

import numpy as np

from dmpcpwa.utils.pwa_models import cent_from_dist


def get_IC():
    return np.array([[-11, -18, 2, -19, 15, 19]]).T
    # return np.array([[-17,-18,18,-19,-18,19]]).T
    # return np.array([[17, 18, 18, 15, -18, 15]]).T
    # return np.array([[0, 19, 19, 0, -19, 0]]).T
    # return np.array([[0, -19, -18, -15, 18, -15]]).T
    # return np.array([[-10, 18, 10, 18, 18, -10]]).T
    # return np.array([[18, 10, -10, -18, 10, -18]]).T
    # return np.array([[-17, -18, 19, 0, 18, -15]]).T
    # return np.array([[0, -19, 18, -19, 10, -18]]).T
    # return np.array([[-18, 15, 19, 0, 10, 18]]).T


Q_x = np.array([[2, 0], [0, 2]])
Q_u = 0.2 * np.array([[1]])


def get_cost_matrices():
    return Q_x, Q_u


A1 = np.array([[0.6324, 0.2785], [0.0975, 0.5469]])
A2 = np.array([[0.6555, 0.7060], [0.1712, 0.0318]])
A3 = np.array([[0.6324, 0.2785], [0.0975, 0.5469]])
A4 = np.array([[0.6555, 0.7060], [0.1712, 0.0318]])

np.linalg.eig(A1)

B1 = np.array([[1], [0]])
B2 = np.array([[1], [0]])
B3 = np.array([[1], [0]])
B4 = np.array([[1], [0]])

c1 = np.zeros((2, 1))
c2 = np.zeros((2, 1))
c3 = np.zeros((2, 1))
c4 = np.zeros((2, 1))

A = [A1, A2, A3, A4]
B = [B1, B2, B3, B4]
c = [c1, c2, c3, c4]

D = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
E = np.array([[20], [20], [20], [20]])

u_lim = 3
F = np.array([[1], [-1]])
G = np.array([[u_lim], [u_lim]])

S1 = np.array([[1, -1], [-1, -1]])
R1 = np.zeros((2, 1))
T1 = np.zeros((2, 1))

S2 = np.array([[-1, 1], [-1, -1]])
R2 = np.zeros((2, 1))
T2 = np.zeros((2, 1))

S3 = np.array([[-1, 1], [1, 1]])
R3 = np.zeros((2, 1))
T3 = np.zeros((2, 1))

S4 = np.array([[1, -1], [1, 1]])
R4 = np.zeros((2, 1))
T4 = np.zeros((2, 1))

S = [S1, S2, S3, S4]
R = [R1, R2, R3, R4]
T = [T1, T2, T3, T4]


def get_local_system():
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


Adj = np.array([[0, 1, 0], [1, 0, 1], [1, 0, 0]])
A_c = 2e-3 * np.eye(2)


def get_cent_system():
    # system 1
    sys_1 = get_local_system()
    sys_1["Ac"] = [[A_c] for i in range(4)]

    # system 2
    sys_2 = get_local_system()
    sys_2["Ac"] = [[A_c, A_c] for i in range(4)]

    # system 3
    sys_3 = get_local_system()
    sys_3["Ac"] = [[A_c] for i in range(4)]

    return cent_from_dist([sys_1, sys_2, sys_3], Adj)


def get_local_coupled_systems():
    # manually construct system descriptions and coupling
    system = get_local_system()
    systems = []  # list of systems, 1 for each agent
    systems.append(system.copy())
    Ac_i = [A_c]
    systems[0]["Ac"] = []
    for i in range(len(system["S"])):
        systems[0]["Ac"] = systems[0]["Ac"] + [Ac_i]

    systems.append(system.copy())
    Ac_i = [A_c, A_c]
    systems[1]["Ac"] = []
    for i in range(len(system["S"])):
        systems[1]["Ac"] = systems[1]["Ac"] + [Ac_i]

    systems.append(system.copy())
    Ac_i = [A_c]
    systems[2]["Ac"] = []
    for i in range(len(system["S"])):
        systems[2]["Ac"] = systems[2]["Ac"] + [Ac_i]

    return systems


def get_adj():
    return Adj


def get_A_c():
    return A_c


# invariant set
P = np.array([[7.8514, 8.1971], [8.1957, -7.8503]])
g = 47
A_t = np.vstack((P, -P))
b_t = g * np.ones((4, 1))


def get_inv_set():
    return A_t, b_t


def get_warm_start():
    raise RuntimeError("get_warm_start not implemented for model 2")


# linear terminal controllers
K = [
    np.array([[-0.0544, -0.1398]]),
    np.array([[-0.1544, -0.0295]]),
    np.array([[-0.0544, -0.1398]]),
    np.array([[-0.1544, -0.0295]]),
]


def get_terminal_K():
    return K

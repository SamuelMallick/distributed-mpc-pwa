import numpy as np

from dmpcpwa.utils.pwa_models import cent_from_dist

A1 = np.array([[1, 0.8], [0, 1.15]])
A2 = np.array([[1, 0.9], [0, 1.2]])
A3 = np.array([[1, 0.95], [1.2, 1.15]])
A4 = np.array([[1, 0.85], [0, 1.15]])
A5 = np.array([[1, 0.8], [0, 1.1]])

B_all = np.array([[1], [0.3]])

c1 = np.zeros((2, 1))
c2 = np.zeros((2, 1))
c3 = np.zeros((2, 1))
c4 = np.zeros((2, 1))
c5 = np.array([[-0.1], [0.1]])

A = [A1, A2, A3, A4, A5]
B = [B_all for i in range(5)]
c = [c1, c2, c3, c4, c5]

D = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
E = np.array([[2], [2], [2], [2]])

u_lim = 1
F = np.array([[1], [-1]])
G = np.array([[u_lim], [u_lim]])

S1 = np.array([[-1, 0], [0, -1]])
R1 = np.zeros((2, 1))
T1 = np.zeros((2, 1))

S2 = np.array([[1, 0], [0, -1], [-1, 1]])
R2 = np.zeros((3, 1))
T2 = np.array([[0], [0], [0.5]])

S3 = np.array([[1, 0], [0, 1]])
R3 = np.zeros((2, 1))
T3 = np.zeros((2, 1))

S4 = np.array([[-1, 0], [0, 1]])
R4 = np.zeros((2, 1))
T4 = np.zeros((2, 1))

S5 = np.array([[1, 0], [0, -1], [1, -1]])
R5 = np.zeros((3, 1))
T5 = np.array([[0], [0], [-0.5]])

S = [S1, S2, S3, S4, S5]
R = [R1, R2, R3, R4, R5]
T = [T1, T2, T3, T4, T5]


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


def get_cent_system():
    A_c1 = 4e-3 * np.ones((2, 2))
    A_c2 = 2e-3 * np.ones((2, 2))

    # system 1
    sys_1 = get_local_system()
    sys_1["Ac"] = [[A_c1] for i in range(5)]

    # system 2
    sys_2 = get_local_system()
    sys_2["Ac"] = [[A_c2, A_c2] for i in range(5)]

    # system 3
    sys_3 = get_local_system()
    sys_3["Ac"] = [[A_c1] for i in range(5)]

    Adj = np.array([[0, 1, 0], [1, 0, 1], [1, 0, 0]])

    return cent_from_dist([sys_1, sys_2, sys_3], Adj)

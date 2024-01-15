import numpy as np

from dmpcpwa.utils.pwa_models import cent_from_dist


def get_IC():
    return 0.7 * np.array([[-1.9, 0.7, -1.5, -1.3, 1.9, -1.4]]).T


Q_x = np.array([[1, 0], [0, 1]])
Q_u = 1 * np.array([[1]])


def get_cost_matrices():
    return Q_x, Q_u


A1 = np.array([[1, 0.8], [0, 1.15]])
A2 = np.array([[1, 0.9], [0, 1.2]])
A3 = np.array([[1, 0.85], [0, 1.15]])

np.linalg.eig(A1)

B1 = np.array([[1], [0.3]])
B2 = np.array([[1], [0.3]])
B3 = np.array([[1], [0.3]])


c1 = np.zeros((2, 1))
c2 = np.zeros((2, 1))
c3 = np.zeros((2, 1))

A = [A1, A2, A3]
B = [B1, B2, B3]
c = [c1, c2, c3]

D = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
E = np.array([[2], [2], [2], [2]])

u_lim = 2
F = np.array([[1], [-1]])
G = np.array([[u_lim], [u_lim]])

S1 = np.array([[-1, 0], [0, -1]])
R1 = np.zeros((2, 1))
T1 = np.array([[0], [0]])

S2 = np.array([[0, 0], [0, 1]])
R2 = np.zeros((2, 1))
T2 = np.array([[0], [0]])

S3 = np.array([[1, 0], [0, -1]])
R3 = np.zeros((2, 1))
T3 = np.array([[0], [0]])

S = [S1, S2, S3]
R = [R1, R2, R3]
T = [T1, T2, T3]


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
A_c = 1e-2 * np.eye(2)


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
A_t = np.array(
    [
        [-0.453236242767145, 0.717618597190272],
        [-0.447213595499958, 0],
        [0.493668038095369, -0.651887271907789],
        [0.447213595499958, 0],
        [0.319444671265620, -1.61636409821550],
        [-0.319444671265620, 1.61636409821550],
        [0.603100563386216, -2.10125716403916],
        [-0.603100563386216, 2.10125716403916],
        [0.649860814466927, -2.05275492670565],
        [-0.649860814466927, 2.05275492670565],
        [0.591515751481720, -1.78605177958610],
        [-0.591515751481720, 1.78605177958610],
        [0.527855972779948, -1.49679660533768],
        [-0.527855972779948, 1.49679660533768],
        [0.437776033561876, -1.19569737248679],
        [-0.437776033561877, 1.19569737248679],
    ]
)
b_t = np.array(
    [
        [
            0.528772594986718,
            0.894427190999916,
            0.575616932419200,
            0.894427190999916,
            0.876538647179918,
            0.876538647179918,
            0.799106296400673,
            0.799106296400673,
            0.690931987303658,
            0.690931987303658,
            0.582827357656755,
            0.582827357656755,
            0.487724656414042,
            0.487724656414042,
            0.406738553289337,
            0.406738553289337,
        ]
    ]
).T


def get_inv_set():
    return A_t, b_t

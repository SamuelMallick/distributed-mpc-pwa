import numpy as np

from dmpcpwa.utils.pwa_models import cent_from_dist

A1 = np.array([[1, 0.8], [0, 1.15]])
A2 = np.array([[1, 0.9], [0, 1.2]])
A3 = np.array([[1, 0.95], [0, 1.2]])
A4 = np.array([[1, 0.85], [0, 1.15]])
A5 = np.array([[1, 0.8], [0, 1.1]])

np.linalg.eig(A1)

B1 = np.array([[1], [0.3]])
B2 = np.array([[1], [0.3]])
B3 = np.array([[1.2], [0.3]])
B4 = np.array([[1], [0.3]])
B5 = np.array([[1], [0.3]])

c1 = np.zeros((2, 1))
c2 = np.zeros((2, 1))
c3 = np.zeros((2, 1))
c4 = np.zeros((2, 1))
c5 = np.array([[-0.1], [0.1]])

A = [A1, A2, A3, A4, A5]
B = [B1, B2, B3, B4, B5]
c = [c1, c2, c3, c4, c5]

D = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
E = np.array([[2], [2], [2], [2]])

u_lim = 1
F = np.array([[1], [-1]])
G = np.array([[u_lim], [u_lim]])

S1 = np.array([[-1, 0], [0, -1], [0, 0]])
R1 = np.zeros((3, 1))
T1 = np.zeros((3, 1))

S2 = np.array([[1, 0], [0, -1], [-1, 1]])
R2 = np.zeros((3, 1))
T2 = np.array([[0], [0], [0.5]])

S3 = np.array([[1, 0], [0, 1], [0, 0]])
R3 = np.zeros((3, 1))
T3 = np.zeros((3, 1))

S4 = np.array([[-1, 0], [0, 1], [0, 0]])
R4 = np.zeros((3, 1))
T4 = np.zeros((3, 1))

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


Adj = np.array([[0, 1, 0], [1, 0, 1], [1, 0, 0]])
A_c1 = 4e-3 * np.ones((2, 2))
A_c2 = 2e-3 * np.ones((2, 2))


def get_cent_system():
    # system 1
    sys_1 = get_local_system()
    sys_1["Ac"] = [[A_c1] for i in range(5)]

    # system 2
    sys_2 = get_local_system()
    sys_2["Ac"] = [[A_c2, A_c2] for i in range(5)]

    # system 3
    sys_3 = get_local_system()
    sys_3["Ac"] = [[A_c1] for i in range(5)]

    return cent_from_dist([sys_1, sys_2, sys_3], Adj)


def get_adj():
    return Adj


def get_A_c1():
    return A_c1


def get_A_c2():
    return A_c2


# invariant set
A_t = np.array(
    [
        [-0.838167327304677, 2.51323092851876],
        [0.585508344938054, -1.75631719114707],
        [0.541654381033104, -0.775688909303791],
        [-0.476973740057866, 0.850220570513130],
        [-0.201101499285382, 1.43646188576863],
        [0.0878243840517243, -0.841682655544037],
        [-0.538619343812254, 2.37460628225254],
        [0.349690037777757, -1.57977900076078],
        [-0.656168180152136, 2.63105086621010],
        [0.446354103490724, -1.80464637275753],
        [-0.659574109903369, 2.54239695824888],
        [0.456622901487060, -1.76715355410731],
        [-0.880304361905936, 2.85024201553224],
        [0.613962474041644, -1.98974691126937],
        [-0.826308401398932, 2.53188349781508],
        [0.577500296627462, -1.77021376036799],
        [-0.844222949043780, 2.48914879864801],
        [0.589597561096654, -1.73907940396705],
        [-0.717764117593537, 2.04305646158912],
        [0.501691486678731, -1.42830157647150],
        [-0.730865407961130, 2.10251451801120],
        [0.511096575648060, -1.47057270266255],
        [-0.704828966531172, 1.98896631663900],
        [0.492538113030303, -1.39018540377949],
        [0.411104161913566, -1.14484689500220],
    ]
)
b_t = np.array(
    [
        [
            0.195206369510529,
            0.132640535827569,
            0.323909319857796,
            0.222757789474440,
            0.313370573535101,
            0.212140234989872,
            0.300270066454669,
            0.204704178673106,
            0.276964261446150,
            0.189268426364798,
            0.250666509075253,
            0.171260422554812,
            0.225050740530035,
            0.153470210910057,
            0.195206369510529,
            0.132640535827569,
            0.195206369510529,
            0.132640535827569,
            0.168395183463942,
            0.113905931538888,
            0.168340834316817,
            0.113858823371605,
            0.168539395528995,
            0.114011120107059,
            0.0980054691451204,
        ]
    ]
).T


def get_inv_set():
    return A_t, b_t

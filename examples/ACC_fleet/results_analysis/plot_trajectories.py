import pickle

import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

nx_l = 2
n = 3
plot_len = 50

with open(
    f"examples/ACC_fleet/data/cent/cent_n_3_N_10_Q_False_DG_False_HOM_True_LT_1.pkl",
    "rb",
) as file:
    X = pickle.load(file)
    U = pickle.load(file)
    R = pickle.load(file)
    solve_times = pickle.load(file)
    node_counts = pickle.load(file)
    violations = pickle.load(file)
    leader_state = pickle.load(file)

_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(leader_state[0, :plot_len], "--")
axs[1].plot(leader_state[1, :plot_len], "--")
for i in range(n):
    axs[0].plot(X[:plot_len, nx_l * i])
    axs[1].plot(X[:plot_len, nx_l * i + 1])
axs[0].set_ylabel(r"pos ($m$)")
axs[1].set_ylabel(r"vel ($ms^{-1}$)")
axs[1].set_xlabel(r"time step $k$")
axs[0].legend(["reference"])
plt.show()

import pickle

import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

nx_l = 2
plot_len = 50
name = "admm20"
DG = False
Q = True
HOM = True
n = 5
N = 5
LT = 1
with open(
    f"{name}_n_{n}_N_{N}_Q_{Q}_DG_{DG}_HOM_{HOM}_LT_{LT}.pkl",
    "rb",
) as file:
    X = pickle.load(file)
    U = pickle.load(file)
    R = pickle.load(file)
    solve_times = pickle.load(file)
    node_counts = pickle.load(file)
    violations = pickle.load(file)
    leader_state = pickle.load(file)

print(f"tracking const: {sum(R)}")
print(f"av comp time: {sum(solve_times)/len(solve_times)}")

_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(leader_state[0, :plot_len], "--")
axs[1].plot(leader_state[1, :plot_len], "--")
for i in range(n):
    axs[0].plot(X[:plot_len, nx_l * i])
    axs[1].plot(X[:plot_len, nx_l * i + 1])
axs[0].set_ylabel(r"pos ($m$)")
axs[1].set_ylabel(r"vel ($ms^{-1}$)")
axs[1].set_ylim(0, 40)
axs[1].set_xlabel(r"time step $k$")
axs[0].legend(["reference"])

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.plot(U)

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.plot(R.squeeze())
plt.show()

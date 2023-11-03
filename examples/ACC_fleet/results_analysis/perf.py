import pickle

import matplotlib.pyplot as plt
import numpy as np

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

nx_l = 2

types = ["cent", "decent", "seq", "event", "admm"]

LT = 1
HOM = True
DG = False
Q = False
n_sw = [i for i in range(2, 4)]
N = 7

track_costs = []
time_min = []
time_max = []
time_av = []
nodes = []
viols = []
counter = 0
for type in types:
    track_costs.append([])
    time_min.append([])
    time_max.append([])
    time_av.append([])
    nodes.append([])
    viols.append([])
    for n in n_sw:
        with open(
            f"examples/ACC_fleet/data/{type}/perf_n/{type}_n_{n}_N_{N}_Q_{Q}_DG_{DG}_HOM_{HOM}_LT_{LT}.pkl",
            "rb",
        ) as file:
            X = pickle.load(file)
            U = pickle.load(file)
            R = pickle.load(file)
            solve_times = pickle.load(file)
            node_counts = pickle.load(file)
            violations = pickle.load(file)
            leader_state = pickle.load(file)

        track_costs[counter].append(sum(R))
        time_min[counter].append(min(solve_times)[0])
        time_max[counter].append(max(solve_times)[0])
        time_av[counter].append(sum(solve_times)[0] / len(solve_times))
        nodes[counter].append(max(node_counts))
        viols[counter].append(sum(violations) / 100)
    counter += 1

leg = ["decent", "seq", "event", "admm"]
lw = 0.8

# tracking cost as percentrage performance drop from centralized
perf_drop = []
for i in range(1, len(types)):
    perf_drop.append(
        [
            100 * (track_costs[i][j] - track_costs[0][j]) / track_costs[0][j]
            for j in range(len(track_costs[0]))
        ]
    )

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
for i in range(counter - 1):
    axs.plot(n_sw, perf_drop[i], "-o", linewidth=lw)
axs.legend(leg)
axs.set_xlabel("$n$")
axs.set_ylabel(r"$\%J$")
# axs.set_ylim(-2, 500)

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
error_lower = [
    [time_av[i][j] - time_min[i][j] for j in range(len(n_sw))] for i in range(counter)
]
error_upper = [
    [time_max[i][j] - time_av[i][j] for j in range(len(n_sw))] for i in range(counter)
]
for i in range(1, counter):
    # axs.plot(n_sw, time_av[i], '-o')
    # axs.fill_between(n_sw, time_max[i], time_min[i], alpha = 0.5)#, where=(time_max[i] > time_min[i]), interpolate=True, color='gray', alpha=0.5)
    _, _, bars = axs.errorbar(
        np.asarray([n_sw[j] + 0.0 * i for j in range(len(n_sw))]),
        np.asarray(time_av[i]),
        yerr=[np.asarray(error_lower[i]), np.asarray(error_upper[i])],
        linewidth=lw,
        fmt="-o",
        capsize=4,
    )
    [bar.set_alpha(0.7) for bar in bars]
leg_temp = [x for pair in zip(leg, [""] * len(leg)) for x in pair]
axs.legend(leg)
axs.set_xlabel("$n$")
axs.set_ylabel("$t_{av}$")

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
for i in range(1, counter):
    axs.plot(n_sw, nodes[i], "-o", linewidth=lw)
axs.legend(leg)
axs.set_xlabel("$n$")
axs.set_ylabel(r"$\#nodes$")

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
for i in range(1, counter):
    axs.plot(n_sw, viols[i], "-o", linewidth=lw)
axs.legend(leg)
axs.set_xlabel("$n$")
axs.set_ylabel(r"$\#CV$")

plt.show()

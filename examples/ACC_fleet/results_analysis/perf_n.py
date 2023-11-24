import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

nx_l = 2

# types = ["cent", "decent", "seq", "event2", "event3","event4", "admm5", "admm10", "admm20"]
types = ["cent", "decent", "seq", "event3", "event4", "admm10", "admm20"]
# types = ["cent", "decent", "seq","event4", "admm20"]
num_iter_vars = 2

LT = 1
HOM = True
DG = False
Q = True
n_sw = [i for i in range(2, 11)]
N = 6

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
            f"{type}_n_{n}_N_{N}_Q_{Q}_DG_{DG}_HOM_{HOM}_LT_{LT}.pkl",
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

# leg = ["decent", "seq", "event (2)","event (3)","event (4)", "admm (5)", "admm (10)", "admm (20)"]
leg = ["decent", "seq", "event (3)", "event (4)", "admm (10)", "admm (20)"]
# leg = ["decent", "seq","event (4)", "admm (20)"]
lw = 1.5
ms = 5

# tracking cost as percentrage performance drop from centralized
perf_drop = []
for i in range(1, counter):
    perf_drop.append(
        [
            100 * (track_costs[i][j] - track_costs[0][j]) / track_costs[0][j]
            for j in range(len(track_costs[0]))
        ]
    )
gs = GridSpec(3, 4, width_ratios=[3, 1, 1, 1])

# _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs = plt.subplot(gs[:, :1])
for i in range(counter - 1):
    axs.plot(
        n_sw,
        np.asarray(perf_drop[i]).reshape(len(n_sw)),
        "-o",
        linewidth=lw,
        markersize=ms,
        color=f"C{i}",
    )
axs.legend(leg)
axs.set_xlabel("$n$")
axs.set_ylabel(r"$\%J$")
axs.set_ylim(-25, 1000)
y_lim = 300

axs = [None] * 3
# Add the three subplots on the right
axs[0] = plt.subplot(gs[0, 1:])
axs[1] = plt.subplot(gs[1, 1:])
axs[2] = plt.subplot(gs[2, 1:])

# _, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
for i in range(2):
    axs[0].plot(
        n_sw,
        np.asarray(perf_drop[i]).reshape(len(n_sw)),
        "-o",
        linewidth=lw,
        markersize=ms,
        color=f"C{i}",
    )
# axs[0].legend(leg[:2])
# axs[0].set_ylabel(r"$\%J$")
axs[0].set_ylim(-25, y_lim)

for i in range(2, 2 + num_iter_vars):
    axs[1].plot(
        n_sw,
        np.asarray(perf_drop[i]).reshape(len(n_sw)),
        "-o",
        linewidth=lw,
        markersize=ms,
        color=f"C{i}",
    )
# axs[1].legend(leg[2:5])
# axs[1].set_ylabel(r"$\%J$")
axs[1].set_ylim(-25, y_lim)

for i in range(2 + num_iter_vars, 2 + 2 * num_iter_vars):
    axs[2].plot(
        n_sw,
        np.asarray(perf_drop[i]).reshape(len(n_sw)),
        "-o",
        linewidth=lw,
        markersize=ms,
        color=f"C{i}",
    )
# axs[2].legend(leg[7:8])
axs[2].set_xlabel("$n$")
# axs[2].set_ylabel(r"$\%J$")
axs[2].set_ylim(-25, y_lim)

plt.figure()
gs = GridSpec(3, 4, width_ratios=[3, 1, 1, 1])

axs = plt.subplot(gs[:, :1])
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
        markersize=ms,
        fmt="-o",
        capsize=4,
    )
    [bar.set_alpha(0.7) for bar in bars]
axs.legend(leg)
axs.set_xlabel("$n$")
axs.set_ylabel("$t_{COMP}$")

axs = [None] * 3
# Add the three subplots on the right
axs[0] = plt.subplot(gs[0, 1:])
axs[1] = plt.subplot(gs[1, 1:])
axs[2] = plt.subplot(gs[2, 1:])
y_lim = 2
for i in range(1, 3):
    _, _, bars = axs[0].errorbar(
        np.asarray([n_sw[j] + 0.0 * i for j in range(len(n_sw))]),
        np.asarray(time_av[i]),
        yerr=[np.asarray(error_lower[i]), np.asarray(error_upper[i])],
        linewidth=lw,
        markersize=ms,
        color=f"C{i-1}",
        fmt="-o",
        capsize=4,
    )
    [bar.set_alpha(0.7) for bar in bars]
# axs[0].legend(leg[:2])
# axs[0].set_ylabel(r"$\%J$")
axs[0].set_ylim(-0.1, y_lim)

for i in range(3, 3 + num_iter_vars):
    _, _, bars = axs[1].errorbar(
        np.asarray([n_sw[j] + 0.0 * i for j in range(len(n_sw))]),
        np.asarray(time_av[i]),
        yerr=[np.asarray(error_lower[i]), np.asarray(error_upper[i])],
        linewidth=lw,
        markersize=ms,
        color=f"C{i-1}",
        fmt="-o",
        capsize=4,
    )
    [bar.set_alpha(0.7) for bar in bars]
# axs[1].legend(leg[2:5])
# axs[1].set_ylabel(r"$\%J$")
axs[1].set_ylim(-0.1, y_lim)

for i in range(3 + num_iter_vars, 3 + 2 * num_iter_vars):
    _, _, bars = axs[2].errorbar(
        np.asarray([n_sw[j] + 0.0 * i for j in range(len(n_sw))]),
        np.asarray(time_av[i]),
        yerr=[np.asarray(error_lower[i]), np.asarray(error_upper[i])],
        linewidth=lw,
        markersize=ms,
        color=f"C{i-1}",
        fmt="-o",
        capsize=4,
    )
    [bar.set_alpha(0.7) for bar in bars]
# axs[2].legend(leg[5:8])
axs[2].set_xlabel("$n$")
# axs[2].set_ylabel(r"$\%J$")
axs[2].set_ylim(-0.1, y_lim)

plt.figure()
gs = GridSpec(3, 4, width_ratios=[3, 1, 1, 1])

axs = plt.subplot(gs[:, :1])
for i in range(1, counter):
    axs.plot(
        n_sw,
        nodes[i],
        "-o",
        linewidth=lw,
        markersize=ms,
    )
axs.legend(leg)
axs.set_xlabel("$n$")
axs.set_ylabel(r"$\#nodes$")

axs = [None] * 3
# Add the three subplots on the right
axs[0] = plt.subplot(gs[0, 1:])
axs[1] = plt.subplot(gs[1, 1:])
axs[2] = plt.subplot(gs[2, 1:])
y_lim = 1000
for i in range(1, 3):
    axs[0].plot(n_sw, nodes[i], "-o", linewidth=lw, markersize=ms, color=f"C{i-1}")
axs[0].set_ylim(-0.1, y_lim)

for i in range(3, 3 + num_iter_vars):
    axs[1].plot(n_sw, nodes[i], "-o", linewidth=lw, markersize=ms, color=f"C{i-1}")
axs[1].set_ylim(-0.1, y_lim)

for i in range(3 + num_iter_vars, 3 + 2 * num_iter_vars):
    axs[2].plot(n_sw, nodes[i], "-o", linewidth=lw, markersize=ms, color=f"C{i-1}")
axs[2].set_xlabel("$n$")
axs[2].set_ylim(-0.1, y_lim)

_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
for i in range(1, counter):
    axs.plot(
        n_sw,
        viols[i],
        "-o",
        linewidth=lw,
        markersize=ms,
    )
axs.legend(leg)
axs.set_xlabel("$n$")
axs.set_ylabel(r"$\#CV$")


plt.show()

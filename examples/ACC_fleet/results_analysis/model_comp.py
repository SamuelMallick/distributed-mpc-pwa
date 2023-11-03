import pickle

nx_l = 2
n = 4
plot_len = 100
name = "event"
DG = True
with open(
    f"examples/ACC_fleet/data/{name}/model_comp/{name}_n_4_N_6_Q_False_DG_{DG}_HOM_True_LT_1.pkl",
    "rb",
) as file:
    X = pickle.load(file)
    U = pickle.load(file)
    R = pickle.load(file)
    solve_times = pickle.load(file)
    node_counts = pickle.load(file)
    violations = pickle.load(file)
    leader_state = pickle.load(file)

dg_cost = sum(R)
dg_time = sum(solve_times) / len(solve_times)

DG = False
with open(
    f"examples/ACC_fleet/data/{name}/model_comp/{name}_n_4_N_6_Q_False_DG_{DG}_HOM_True_LT_1.pkl",
    "rb",
) as file:
    X = pickle.load(file)
    U = pickle.load(file)
    R = pickle.load(file)
    solve_times = pickle.load(file)
    node_counts = pickle.load(file)
    violations = pickle.load(file)
    leader_state = pickle.load(file)

pwa_cost = sum(R)
pwa_time = sum(solve_times) / len(solve_times)

print(f"J percent inc {100*(dg_cost-pwa_cost)/pwa_cost}")
print(f"t percent inc {100*(dg_time-pwa_time)/pwa_time}")

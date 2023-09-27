import pickle

with open(
    "examples/ACC_fleet/test.pkl",
    "rb",
) as file:
    X = pickle.load(file)
    U = pickle.load(file)
    R = pickle.load(file)
    run_times = pickle.load(file)

print(
    f"mean = {sum(run_times)/len(run_times)}, max = {max(run_times)}, min = {min(run_times)}"
)

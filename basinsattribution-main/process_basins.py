import utils
import numpy as np
import os
import json
import analyze_basins
import tqdm


utils.good_plt_config()
os.makedirs("outputs", exist_ok=True)

data = []
with open("/home/neocortex/repos/LLaMEA-ELA/LLaMEA/exp-09-25_072937-LLaMEA-gpt-5-nano-ELA-Basins_Separable-sharing/log.jsonl", "r") as f:
    for line in f:
        if line.strip():  # skip empty lines
            data.append(json.loads(line))

for row in tqdm.tqdm(data):

    if row["fitness"] < 0.5:
        continue
    ns = {}
    exec(row["code"], ns)                      # defines class with same name as row["name"]
    F   = getattr(ns[row["name"]](dim=2), "f")      # instantiate & grab its .f method

    n = 10
    x1 = np.linspace(-5, 5, n)
    x2 = np.linspace(-5, 5, n)
    X1, X2 = np.meshgrid(x1, x2)
    X_data = np.column_stack([X1.ravel(), X2.ravel()])

    y_data = np.array([F(x) for x in X_data])
    #utils.plot_3D_surface((-5, 5), (-5, 5), F, X_data)
        
    bloc = analyze_basins.BasinsLoc()
    nr_of_optima = bloc.alg_closest_points(F, X=X_data, y=y_data)

    n_init = len(X_data)

    def find_root(i, to, cache):
        # recursive path compression
        if to[i] != i:
            if i not in cache:
                cache[i] = find_root(to[i], to, cache)
            to[i] = cache[i]
        return to[i]

    def collapse_to(to):
        cache = {}
        for i in range(len(to)):
            to[i] = find_root(i, to, cache)
        return to

    # After running alg_closest_points
    to = np.array(bloc.to).copy()
    roots = collapse_to(to)
    roots_init = roots[:n_init]

    unique_basins, counts = np.unique(roots_init, return_counts=True)

    basin_info = []

    #print(f"Number of basins: {len(unique_basins)}")
    for basin_id, size in zip(unique_basins, counts):
        #print(f"Basin {basin_id}: size={size}, f(x)={F(bloc.X[basin_id]):.4f}")
        basin_info.append((size, F(bloc.X[basin_id])))
    row["basin_info"] = basin_info
    row["nr_of_optima"] = nr_of_optima

    #fig, ax = bloc.plot_attraction_basins(F, X=X_data)
    

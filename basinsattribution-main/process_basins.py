import utils
import numpy as np
import os
import json
import analyze_basins
import tqdm


utils.good_plt_config()
os.makedirs("outputs", exist_ok=True)


base_dir = "/home/neocortex/repos/LLaMEA-ELA/LLaMEA"
experiment_dirs = [
    "exp-09-25_072937-LLaMEA-gpt-5-nano-ELA-Basins_Separable-sharing",
    "exp-09-25_073301-LLaMEA-gpt-5-nano-ELA-Basins_Homogeneous-sharing",
    "exp-09-25_074532-LLaMEA-gpt-5-nano-ELA-Basins_GlobalLocal-sharing",
    "exp-09-25_091027-LLaMEA-gpt-5-nano-ELA-Basins_Multimodality-sharing",
    "exp-09-25_105741-LLaMEA-gpt-5-nano-ELA-Basins-sharing",
    "exp-09-25_114848-LLaMEA-gpt-5-nano-ELA-Separable_GlobalLocal-sharing",
    "exp-09-25_115914-LLaMEA-gpt-5-nano-ELA-Separable_Multimodality-sharing",
    "exp-09-25_121517-LLaMEA-gpt-5-nano-ELA-Separable_Homogeneous-sharing",
    "exp-09-25_122101-LLaMEA-gpt-5-nano-ELA-Separable-sharing",
    "exp-09-25_122848-LLaMEA-gpt-5-nano-ELA-GlobalLocal_Multimodality-sharing",
    "exp-09-25_132136-LLaMEA-gpt-5-nano-ELA-GlobalLocal_Homogeneous-sharing",
    "exp-09-25_144046-LLaMEA-gpt-5-nano-ELA-GlobalLocal-sharing",
    "exp-09-25_145409-LLaMEA-gpt-5-nano-ELA-Multimodality_Homogeneous-sharing",
    "exp-09-25_165258-LLaMEA-gpt-5-nano-ELA-Multimodality-sharing",
    "exp-09-25_165843-LLaMEA-gpt-5-nano-ELA-Homogeneous-sharing",
    "exp-09-26_120435-LLaMEA-gpt-5-nano-ELA-NOT Homogeneous_Separable-sharing",
    "exp-09-26_121734-LLaMEA-gpt-5-nano-ELA-NOT Homogeneous_GlobalLocal-sharing",
    "exp-09-26_122528-LLaMEA-gpt-5-nano-ELA-NOT Homogeneous_Multimodality-sharing",
    "exp-09-26_123417-LLaMEA-gpt-5-nano-ELA-NOT Homogeneous-sharing",
    "exp-09-26_124130-LLaMEA-gpt-5-nano-ELA-NOT Basins_Separable-sharing",
    "exp-09-26_125536-LLaMEA-gpt-5-nano-ELA-NOT Basins_GlobalLocal-sharing",
    "exp-09-26_130053-LLaMEA-gpt-5-nano-ELA-NOT Basins_Multimodality-sharing",
    "exp-09-26_130619-LLaMEA-gpt-5-nano-ELA-NOT Basins-sharing",
]

for exp_dir in experiment_dirs:
    datadir = "{}/{}".format(base_dir, exp_dir)

    data = []
    with open(f"{datadir}/log.jsonl", "r") as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))

    for row in tqdm.tqdm(data):

        if row["fitness"] < 0.5:
            continue
        ns = {}
        exec(row["code"], ns)                      # defines class with same name as row["name"]
        F   = getattr(ns[row["name"]](dim=2), "f")      # instantiate & grab its .f method

        n = 15
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
            basin_info.append((size, bloc.X[basin_id], F(bloc.X[basin_id])))
        row["basin_info"] = basin_info
        row["nr_of_basins"] = nr_of_optima
        

        #fig, ax = bloc.plot_attraction_basins(F, X=X_data)
    with open(f"outputs/{exp_dir}.jsonl", "w") as f_out:
        for row in data:
            json.dump(row, f_out)
            f_out.write("\n")

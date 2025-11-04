import utils
import numpy as np
import os
import json
import analyze_basins
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import ioh

import sys


def process_bbob(fid=1):
    output_file = os.path.join("outputs", f"bbob.jsonl")
    n = 10
    x1 = np.linspace(-5, 5, n)
    x2 = np.linspace(-5, 5, n)
    X1, X2 = np.meshgrid(x1, x2)
    X_data = np.column_stack([X1.ravel(), X2.ravel()])
    iid = 4
    #for iid in tqdm.tqdm(range(5), desc="instances", leave=False):

    row = {"function_id": fid, "instance_id": iid}
    F = ioh.get_problem(fid, iid, 2)
    
    y_data = np.array([F(x) for x in X_data])
    bloc = analyze_basins.BasinsLoc()
    nr_of_optima = bloc.alg_closest_points(F, X=X_data, y=y_data)

    def find_root(i, to):
        while to[i] != i:
            to[i] = to[to[i]]
            i = to[i]
        return i

    def collapse_to(to):
        for i in range(len(to)):
            to[i] = find_root(i, to)
        return to

    to = np.array(bloc.to).copy()
    roots = collapse_to(to)
    roots_init = roots[:len(X_data)]

    unique_basins, counts = np.unique(roots_init, return_counts=True)
    basin_info = [
        (int(size), bloc.X[basin_id].tolist(), float(F(bloc.X[basin_id])))
        for basin_id, size in zip(unique_basins, counts)
    ]

    row["basin_info"] = basin_info
    row["nr_of_basins"] = int(nr_of_optima)

    with open(output_file, "a") as f_out:
        json.dump(row, f_out)
        f_out.write("\n")

    return f"Done bbob f{fid}"


def process_experiment(exp_dir, base_dir):
    datadir = os.path.join(base_dir, exp_dir)
    output_file = os.path.join("outputs", f"{exp_dir}.jsonl")

    # Skip if already processed (optional)
    if os.path.exists(output_file):
        return f"Skipped {exp_dir} (exists)"

    with open(f"{datadir}/log.jsonl", "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    for row in tqdm.tqdm(data, desc=exp_dir, leave=False):
        if row["fitness"] < 0.5:
            continue
        ns = {}
        exec(row["code"], ns)
        F = getattr(ns[row["name"]](dim=2), "f")

        n = 10
        x1 = np.linspace(-5, 5, n)
        x2 = np.linspace(-5, 5, n)
        X1, X2 = np.meshgrid(x1, x2)
        X_data = np.column_stack([X1.ravel(), X2.ravel()])
        y_data = np.array([F(x) for x in X_data])

        bloc = analyze_basins.BasinsLoc()
        nr_of_optima = bloc.alg_closest_points(F, X=X_data, y=y_data)

        def find_root(i, to):
            while to[i] != i:
                to[i] = to[to[i]]
                i = to[i]
            return i

        def collapse_to(to):
            for i in range(len(to)):
                to[i] = find_root(i, to)
            return to

        to = np.array(bloc.to).copy()
        roots = collapse_to(to)
        roots_init = roots[:len(X_data)]

        unique_basins, counts = np.unique(roots_init, return_counts=True)
        basin_info = [
            (int(size), bloc.X[basin_id].tolist(), float(F(bloc.X[basin_id])))
            for basin_id, size in zip(unique_basins, counts)
        ]

        row["basin_info"] = basin_info
        row["nr_of_basins"] = int(nr_of_optima)

        with open(output_file, "a") as f_out:
            json.dump(row, f_out)
            f_out.write("\n")

    return f"Done {exp_dir}"

if __name__ == "__main__":

    sys.setrecursionlimit(10000)

    #utils.good_plt_config()
    os.makedirs("outputs", exist_ok=True)

    base_dir = "./"
    experiment_dirs = [
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
        "exp-10-30_094616-LLaMEA-gpt-5-nano-ELA-NOT Basins_Separable-sharing",
        "exp-10-30_103416-LLaMEA-gpt-5-nano-ELA-NOT Basins_GlobalLocal-sharing",
        "exp-10-30_115015-LLaMEA-gpt-5-nano-ELA-NOT Basins_Multimodality-sharing", 
        "exp-10-30_120126-LLaMEA-gpt-5-nano-ELA-NOT Basins-sharing" ,
        "exp-10-30_121257-LLaMEA-gpt-5-nano-ELA-Basins_Separable-sharing" ,
        "exp-10-30_123650-LLaMEA-gpt-5-nano-ELA-Basins_GlobalLocal-sharing", 
        "exp-10-30_150153-LLaMEA-gpt-5-nano-ELA-Basins_Multimodality-sharing", 
        "exp-10-30_182432-LLaMEA-gpt-5-nano-ELA-Basins-sharing"
    ]


    if True:
        # one clean global progress bar
        with tqdm.tqdm(total=len(experiment_dirs), desc="Experiments", ncols=90) as pbar:
            with ProcessPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(process_experiment, exp, base_dir): exp for exp in experiment_dirs}
                for fut in as_completed(futures):
                    msg = fut.result()
                    tqdm.tqdm.write(msg)
                    pbar.update(1)

    # Process BBOB functions
    #process_bbob(15)
    # with tqdm.tqdm(total=24, desc="BBOB Functions", ncols=90) as pbar:
    #     with ProcessPoolExecutor(max_workers=24) as executor:
    #         futures = {executor.submit(process_bbob, fid): fid for fid in range(1, 24 + 1)}
    #         for fut in as_completed(futures):
    #             msg = fut.result()
    #             tqdm.tqdm.write(msg)
    #             pbar.update(1)

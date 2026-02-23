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

    base_dir = "/home/neocortex/repos/LLaMEA-ELA/LLaMEA/"
    experiment_dirs = [
        "exp-11-07_095145-LLaMEA-gpt-5-nano-ELA-Separable_GlobalLocal-sharing",
        "exp-11-07_100349-LLaMEA-gpt-5-nano-ELA-Separable_Multimodality-sharing",
        "exp-11-07_101329-LLaMEA-gpt-5-nano-ELA-Separable_Basins-sharing",
        "exp-11-07_102426-LLaMEA-gpt-5-nano-ELA-Separable_Homogeneous-sharing",
        "exp-11-07_103559-LLaMEA-gpt-5-nano-ELA-Separable-sharing",
        "exp-11-07_104417-LLaMEA-gpt-5-nano-ELA-GlobalLocal_Multimodality-sharing",
        "exp-11-07_105439-LLaMEA-gpt-5-nano-ELA-GlobalLocal_Basins-sharing",
        "exp-11-07_131500-LLaMEA-gpt-5-nano-ELA-GlobalLocal_Homogeneous-sharing",
        "exp-11-07_151340-LLaMEA-gpt-5-nano-ELA-GlobalLocal-sharing",
        "exp-11-07_153338-LLaMEA-gpt-5-nano-ELA-Multimodality_Basins-sharing",
        "exp-11-07_191623-LLaMEA-gpt-5-nano-ELA-Multimodality_Homogeneous-sharing",
        "exp-11-07_192657-LLaMEA-gpt-5-nano-ELA-Multimodality-sharing",
        "exp-11-08_121302-LLaMEA-gpt-5-nano-ELA-Basins_Homogeneous-sharing",
        "exp-11-08_135452-LLaMEA-gpt-5-nano-ELA-Basins-sharing",
        "exp-11-08_175315-LLaMEA-gpt-5-nano-ELA-Homogeneous-sharing",
        "exp-11-08_182445-LLaMEA-gpt-5-nano-ELA-NOT Basins_Separable-sharing",
        "exp-11-08_191508-LLaMEA-gpt-5-nano-ELA-NOT Basins_GlobalLocal-sharing",
        "exp-11-08_195321-LLaMEA-gpt-5-nano-ELA-NOT Basins_Multimodality-sharing",
        "exp-11-08_222921-LLaMEA-gpt-5-nano-ELA-NOT Basins-sharing",
        "exp-11-08_230434-LLaMEA-gpt-5-nano-ELA-NOT Homogeneous_Separable-sharing",
        "exp-11-08_234910-LLaMEA-gpt-5-nano-ELA-NOT Homogeneous_GlobalLocal-sharing",
        "exp-11-09_003403-LLaMEA-gpt-5-nano-ELA-NOT Homogeneous_Multimodality-sharing",
        "exp-11-09_010643-LLaMEA-gpt-5-nano-ELA-NOT Homogeneous-sharing",
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

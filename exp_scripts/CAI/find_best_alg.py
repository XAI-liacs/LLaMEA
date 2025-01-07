import os
import json
import numpy as np


def find_best_alg(list_of_dirs):
    max_aucs = []
    max_aucs_idx = []
    for dir_name in list_of_dirs:
        log_file = f"exp_data/CAI/{dir_name}/log.jsonl"
        aucs = []
        with open(log_file, "r") as reader:
            for line in reader:
                obj = json.loads(line)
                aucs += [obj["fitness"]]
        max_auc = max(aucs)
        max_auc_idx = aucs.index(max_auc)
        max_aucs.append(max_auc)
        max_aucs_idx.append(max_auc_idx)
    best_alg_auc = max(max_aucs)
    best_alg_idx = max_aucs.index(best_alg_auc)
    return list_of_dirs[best_alg_idx], max_aucs_idx[best_alg_idx]


exp_names = ["Photonic01-extend", "Photonic02", "Photonic03-extend"]  # , "Photonic03-extend"
dirs = os.listdir("exp_data/CAI")
exp_dirs = [[] for _ in exp_names]
for dir_name in dirs:
    if any([dir_name.endswith(f"{i}") for i in exp_names]):
        for i, exp_name in enumerate(exp_names):
            if dir_name.endswith(f"{exp_name}"):
                exp_dirs[i].append(dir_name)
for i, exp_name in enumerate(exp_names):
    best_alg_dir, idx = find_best_alg(exp_dirs[i])
    print(f"Best algorithm for {exp_name} is {best_alg_dir} with index {idx}")

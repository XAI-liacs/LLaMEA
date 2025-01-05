import os
import jsonlines
import numpy as np
import matplotlib.pyplot as plt


def extract_auc(exp_folder):
    log_file = f"{exp_folder}/log.jsonl"
    aucs = []
    with jsonlines.open(log_file) as reader:
        for obj in reader:
            aucs += [obj["fitness"]]
    # check all the aucs are not NaN
    if all([np.isinf(auc) for auc in aucs]):
        return None
    if all([np.isnan(auc) for auc in aucs]):
        return None
    return aucs

exps = [f"0{i}{info}" for i in range(1, 4) for info in ["", "-default"]]
labels = [f"problem{i}{info}" for i in range(1, 4) for info in [
    " with dynamic mutation rate", " with default LLaMEA settings"]]

auc_data = []
for exp in exps:
    aucs = []
    exp_folders = os.listdir(f"exp_data/CAI/real/{exp}")
    for exp_folder in exp_folders:
        auc = extract_auc(f"exp_data/CAI/real/{exp}/{exp_folder}")
        if auc == None:
            continue
        aucs += [auc]
    max_len = max([len(auc) for auc in aucs])
    aucs = [auc + [auc[-1]]*(max_len - len(auc)) for auc in aucs]
    auc_data += [np.array(aucs)]
current_best_aucs = [np.maximum.accumulate(aucs, axis=1) for aucs in auc_data]
# print(result.shape)
for i, label in enumerate(labels):
    plt.plot(np.mean(current_best_aucs[i], axis=0), label=label)
plt.legend()
plt.savefig("results/CAI/real.png")

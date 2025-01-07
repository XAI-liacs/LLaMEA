import os
import jsonlines
import numpy as np
import pandas as pd
import seaborn as sns
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

exps = ["Photonic01", "Photonic02", "Photonic03"]
labels = ["Bragg Mirror", "Ellipsometry", "Photovoltaics"]
cud = ["#e69f00", "#56b4e9", "#009e73", "#f0e442",
        "#0072b2", "#d55e00", "#cc79a7", "#000000"]
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
keys = ["generation", "auc", "run", "problem"]
values = []
for k in range(len(current_best_aucs)):
    array = current_best_aucs[k]
    for i in range(5):
        for j in range(100):
            values += [[j, array[i, j], i, labels[k]]]
df = pd.DataFrame(values, columns=keys)
for i, label in enumerate(labels):
    df_label = df[df["problem"] == label]
    sns.lineplot(x="generation", y="auc", data=df_label, label=label, palette=[cud[i]])
plt.ylabel("AOCC")
plt.legend()
plt.savefig("results/CAI/real.png")

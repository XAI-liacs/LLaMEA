import os
import jsonlines
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def extract_auc(exp_folder):
    log_file = f"{exp_folder}/log.jsonl"
    aucs = []
    final_y = []
    with jsonlines.open(log_file) as reader:
        current_best = None
        for obj in reader:
            aucs += [obj["fitness"]]
            if obj["metadata"] != {}:
                if current_best == None:
                    current_best = np.mean(obj["metadata"]["final_y"])
                else:
                    current_best = min(
                        np.mean(obj["metadata"]["final_y"]), current_best)
            final_y += [current_best]
    # check all the aucs are not NaN
    if all([np.isinf(auc) for auc in aucs]):
        return None, None
    if all([np.isnan(auc) for auc in aucs]):
        return None, None
    if len(aucs) < 100:
        return None, None
    return aucs, final_y

problems = [
    "bragg",
    # "ellipsometry",
    "photovoltaic"
]
exps = []
labels = []

for prob in problems:
    exps += [f"{prob}_(1 + 1)",
                f"{prob}_with_description_(1 + 1)",
                f"{prob}_with_description_insight_(1 + 1)"]
    # if prob == "photovoltaic":
    #     exps += [f"{prob}_(1 + 1)",
    #              f"{prob}_with_description_(1 + 1)",
    #              f"{prob}_with_description_insight_(1 + 1)"]
    # else:
    #     exps += [f"{prob}",
    #             f"{prob}_with_description",
    #             f"{prob}_with_description_insight"]
    labels += [f"{prob}",
               f"{prob} with description",
               f"{prob} with description\nand algorithmic insights"]

# exps = []
# labels = []
# exp_classes = ["(1, 5)", "(1 + 5)", "(2, 10)", "(2 + 10)", "(5 + 5)"]
# for exp_class in exp_classes:
#     exps += [
#         f"bragg_with_description_insight_{exp_class}",
#         # f"ellipsometry_with_description_insight_{exp_class}"
#     ]
#     labels += [
#         f"mini-bragg {exp_class}",
#         # f"ellipsometry {exp_class}"
#     ]

# exps = ["bragg",
#         "bragg_with_description",
#         "bragg_with_description_insight",
#         "ellipsometry",
#         "ellipsometry_with_description",
# #         "ellipsometry_with_description_insight"]
# labels = ["mini-bragg",
#           "mini-bragg with description",
#           "mini-bragg with description\nand algorithmic insights",
#           "ellipsometry",
#           "ellipsometry with description",
#           "ellipsometry with description\nand algorithmic insights"]
cud = ["#e69f00", "#56b4e9", "#009e73", "#f0e442",
       "#0072b2", "#d55e00", "#cc79a7", "#000000"]
linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
auc_data = []
y_data = []
root_path = "exp_data/CAI/descriptions_insights/"
exp_folders = os.listdir(root_path)

for exp in exps:
    aucs = []
    ys = []
    for exp_folder in exp_folders:
        contents = exp_folder.split("-")
        if contents[-1] != exp:
            continue
        auc, y = extract_auc(f"{root_path}{exp_folder}")
        print(f"Extracting {exp_folder}")
        if auc == None or len(auc) < 100:
            print(f"Skipping {exp_folder}")
            continue
        aucs += [auc]
        ys += [y]
    print([len(auc) for auc in aucs])
    max_len = max([len(auc) for auc in aucs])
    aucs = [auc + [auc[-1]]*(max_len - len(auc)) for auc in aucs]
    auc_data += [np.array(aucs)]
    y_data += [np.array(ys)]
current_best_aucs = [np.maximum.accumulate(aucs, axis=1) for aucs in auc_data]
# print(auc_data[0][0])
# print(current_best_aucs[0][0])
keys = ["generation", "auc", "final_y", "run", "problem"]
values = []
for k in range(len(current_best_aucs)):
    array = current_best_aucs[k]
    y = y_data[k]
    for i in range(len(y)):
        for j in range(100):
            values += [[j, array[i, j], y[i, j], i, labels[k]]]
# print(np.array(values)[:100, 1])
df = pd.DataFrame(values, columns=keys)
df.to_csv("exp_data/CAI/real/conv_plot_description_insight_1.csv")

df_baseline = pd.read_csv("exp_data/CAI/real/baselines.csv")
df_baseline_label = df_baseline[df_baseline["problem"] == "mini-bragg (1 + 1)"]

# sns.lineplot(x="generation", y="auc", data=df_baseline_label,
#              color=cud[5], label="mini-bragg (1 + 1)")
df = pd.read_csv("exp_data/CAI/real/conv_plot_description_insight_1.csv")
for problem in problems:
    for i, label in enumerate(labels):
        if problem not in label:
            continue
        df_label = df[df["problem"] == label]
        print(df_label)
        sns.lineplot(x="generation", y="auc", data=df_label,
                    label=label, linestyle=linestyles[int(i/3)], color=cud[i%3])
    # plt.yscale("log")
    plt.ylabel("AOCC")
    plt.legend()
    plt.savefig(f"results/CAI/auc_description_insight_{problem}.png")
    plt.cla()

    for i, label in enumerate(labels):
        if problem not in label:
            continue
        df_label = df[df["problem"] == label]
        sns.lineplot(x="generation", y="final_y", data=df_label,
                    label=label, linestyle=linestyles[int(i/3)], color=cud[i%3])
    plt.yscale("log")
    plt.ylabel(r"$y^*$")
    plt.legend()
    plt.savefig(f"results/CAI/y_description_insight_{problem}.png")
    plt.cla()

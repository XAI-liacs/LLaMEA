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



def plot(root_folder_path, exp_names, labels, plot_name):
    cud = ["#e69f00", "#56b4e9", "#009e73", "#f0e442",
        "#0072b2", "#d55e00", "#cc79a7", "#000000"]
    linestyles = ["-", "--", ":", "-.", "-", "--", "-.", ":"]
    auc_data = []
    y_data = []
    exp_folders = os.listdir(root_folder_path)
    for exp in exp_names:
        aucs = []
        ys = []
        for exp_folder in exp_folders:
            contents = exp_folder.split("-")
            if contents[-1] != exp:
                continue
            auc, y = extract_auc(f"{root_folder_path}{exp_folder}")
            print(f"Extracting {exp_folder}")
            if auc == None or len(auc) < 100:
                print(f"Skipping {exp_folder}")
                continue
            aucs += [auc]
            ys += [y]
        max_len = max([len(auc) for auc in aucs])
        aucs = [auc + [auc[-1]]*(max_len - len(auc)) for auc in aucs]
        auc_data += [np.array(aucs)]
        y_data += [np.array(ys)]
    current_best_aucs = [np.maximum.accumulate(aucs, axis=1) for aucs in auc_data]
    keys = ["generation", "auc", "final_y", "run", "problem"]
    values = []
    for k in range(len(current_best_aucs)):
        array = current_best_aucs[k]
        y = y_data[k]
        for i in range(len(y)):
            for j in range(100):
                values += [[j, array[i, j], y[i, j], i, labels[k]]]
    df = pd.DataFrame(values, columns=keys)
    df.to_csv("exp_data/CAI/real/temp.csv")
    df_baseline = pd.read_csv("exp_data/CAI/real/baselines.csv")
    if "bragg" in plot_name:
        df_baseline_label = df_baseline[df_baseline["problem"] == "mini-bragg (1 + 1)"]
    elif "ellipsometry" in plot_name:
        df_baseline_label = df_baseline[df_baseline["problem"] == "ellipsometry (1 + 1)"]
    else:
        df_baseline_label = df_baseline[df_baseline["problem"] == "photovoltaic (1 + 1)"]

    sns.lineplot(x="generation", y="auc", data=df_baseline_label,
                 color=cud[5], label="ES (1 + 1)", linestyle="--")
    df = pd.read_csv("exp_data/CAI/real/temp.csv")
    for i, label in enumerate(labels):
        df_label = df[df["problem"] == label]
        sns.lineplot(x="generation", y="auc", data=df_label,
                    label=label, linestyle=linestyles[i], color=cud[i])
    # plt.yscale("log")
    plt.ylabel("AOCC")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"results/CAI/auc_description_insight_{plot_name}.png")
    plt.cla()

    for i, label in enumerate(labels):
        df_label = df[df["problem"] == label]
        sns.lineplot(x="generation", y="final_y", data=df_label,
                    label=label, linestyle=linestyles[i], color=cud[i])
    plt.yscale("log")
    plt.ylabel(r"$y^*$")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"results/CAI/y_description_insight_{plot_name}.png")
    plt.cla()

def plot_description_insights():
    problems = [
        "bragg",
        "ellipsometry",
        "photovoltaic"
    ]
    for prob in problems:
        exp_names = [
            f"{prob}_(1 + 1)",
            f"{prob}_with_description_(1 + 1)",
            f"{prob}_with_description_insight_(1 + 1)"]
        labels = [
            f"{prob}",
            f"{prob} with description",
            f"{prob} with description\nand algorithmic insights"]
        plot("exp_data/CAI/descriptions_insights/", exp_names, labels, prob)
        if prob == "photovoltaic":
            plot("exp_data/CAI/descriptions_insights/start_with_same_solution/",
                 exp_names, labels, "photovoltaic_start_with_same_solution")


def plot_population():
    problems = [
        "bragg",
        "ellipsometry",
        "photovoltaic"
    ]
    for prob in problems:
        if prob == "photovoltaic":
            exp_names = [
                f"{prob}_(1, 5)",
                f"{prob}_(1 + 5)",
                f"{prob}_(2, 10)",
                f"{prob}_(2 + 10)",
                f"{prob}_(5 + 5)",
            ]
        else:
            exp_names = [
                f"{prob}_with_description_insight_(1, 5)",
                f"{prob}_with_description_insight_(1 + 5)",
                f"{prob}_with_description_insight_(2, 10)",
                f"{prob}_with_description_insight_(2 + 10)",
                f"{prob}_with_description_insight_(5 + 5)",
            ]
        labels = [
            f"ES (1 , 5)",
            f"ES (1 + 5)",
            f"ES (2 , 10)",
            f"ES (2 + 10)",
            f"ES (5 + 5)",
        ]
        plot("exp_data/CAI/populations/", exp_names, labels, f"{prob}_population")
        # if prob == "photovoltaic":
        #     plot("exp_data/CAI/descriptions_insights/start_with_same_solution/",
        #          exp_names, labels, "photovoltaic_start_with_same_solution")


# plot_description_insights()
plot_population()

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
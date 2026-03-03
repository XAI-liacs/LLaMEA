# Simple helper file to plot generated auc files (used for the LLaMEA paper).
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import tqdm
import ast
import difflib
import jellyfish


def code_compare(code1, code2):
    # Parse the Python code into ASTs
    # Use difflib to find differences
    diff = difflib.ndiff(code1.splitlines(), code2.splitlines())
    # Count the number of differing lines
    diffs = sum(1 for x in diff if x.startswith("- ") or x.startswith("+ "))
    # Calculate total lines for the ratio
    total_lines = max(len(code1.splitlines()), len(code2.splitlines()))
    similarity_ratio = (total_lines - diffs) / total_lines if total_lines else 1
    return 1 - similarity_ratio


exp_dir = "exp-05-07_14:52:37 baseline"
budget = 100

# GPT 4
exp_dirs = [
    [
        "exp-05-07_145237-gpt-4-turbo-plain",
        "exp-05-14_123400-gpt-4-turbo-plain",
        "exp-05-14_123425-gpt-4-turbo-plain",
        "exp-05-15_064710-gpt-4-turbo-plain",
        "exp-05-15_064730-gpt-4-turbo-plain",
    ],
    [
        "exp-05-13_121553-gpt-4-turbo-detail",
        "exp-05-14_123615-gpt-4-turbo-detail",
        "exp-05-14_194110-gpt-4-turbo-detail",
        "exp-05-15_064859-gpt-4-turbo-detail",
        "exp-05-17_205808-gpt-4-turbo-detail",
    ],
    [
        "exp-05-13_125616-gpt-4-turbo-elitsm",
        "exp-05-14_110804-gpt-4-turbo-elitsm",
        "exp-05-14_111732-gpt-4-turbo-elitsm",
        "exp-05-15_064825-gpt-4-turbo-elitism",
        "exp-05-15_064835-gpt-4-turbo-elitism",
    ],
    # ["exp-05-14_123738-gpt-4-turbo-detail-elitism"],
    [
        "exp-05-14_094348-gpt-3.5-turbo-plain",
        "exp-05-14_131509-gpt-3.5-turbo-plain",
        "exp-05-14_182451-gpt-3.5-turbo-plain",
        "exp-05-16_094926-gpt-3.5-turbo-plain",
        "exp-05-16_094955-gpt-3.5-turbo-plain",
    ],
    [
        "exp-05-14_094717-gpt-3.5-turbo-detail",
        "exp-05-14_192917-gpt-3.5-turbo-detail",
        "exp-05-14_182658-gpt-3.5-turbo-detail",
        "exp-05-16_095027-gpt-3.5-turbo-detail",
        "exp-05-16_095044-gpt-3.5-turbo-detail",
    ],
    [
        "exp-05-14_094824-gpt-3.5-turbo-elitism",
        "exp-05-14_182630-gpt-3.5-turbo-elitism",
        "exp-05-14_182618-gpt-3.5-turbo-elitism",
        "exp-05-16_095114-gpt-3.5-turbo-elitism",
        "exp-05-16_095135-gpt-3.5-turbo-elitism",
    ],
    [
        "exp-05-14_134318-gpt-4o-plain",
        "exp-05-16_145625-gpt-4o-plain",
        "exp-05-16_145649-gpt-4o-plain",
        "exp-05-16_145701-gpt-4o-plain",
        "exp-05-16_145709-gpt-4o-plain",
    ],
    [
        "exp-05-16_145812-gpt-4o-detail",
        "exp-05-16_145821-gpt-4o-detail",
        "exp-05-16_145835-gpt-4o-detail",
        "exp-05-16_145842-gpt-4o-detail",
        "exp-05-16_145900-gpt-4o-detail",
    ],
    [
        "exp-05-14_134351-gpt-4o-elitism",
        "exp-05-16_145906-gpt-4o-detail",
        "exp-05-16_145911-gpt-4o-detail",
        "exp-05-16_145917-gpt-4o-elitism",
        "exp-05-16_145922-gpt-4o-detail",
    ],
]

colors = [
    "b",
    "r",
    "g",
    #'c',
    "b",
    "r",
    "g",
    "b",
    "r",
    "g",
]
linestyles = [
    "solid",
    "solid",
    "solid",
    #'solid',
    "dotted",
    "dotted",
    "dotted",
    "dashed",
    "dashed",
    "dashed",
]
labels = [
    "1,1-GPT4-ES",
    "1,1 GPT4-ES w/ Details",
    "1+1 GPT4-ES",
    #'GPT-4-ES w/ Elitism+Detail',
    "1,1 GPT3.5-ES",
    "1,1 GPT3.5-ES w/ Details",
    "1+1 GPT3.5-ES",
    "1,1 GPT4o-ES",
    "1,1 GPT4o-ES w/ Details",
    "1+1 GPT4o-ES",
]
for i in range(len(exp_dirs)):
    exp_dir = exp_dirs[i]
    color = colors[i]
    ls = linestyles[i]
    label = labels[i]
    mean_aucs = []
    current_best = 0
    best_aucs = []
    std_aucs = []
    error_bars = []
    best_try = ""
    for k in range(budget):
        m_aucs = []
        for d in exp_dir:
            if os.path.isfile(f"{d}/try-{k}-aucs.txt"):
                aucs = np.loadtxt(f"{d}/try-{k}-aucs.txt")
                m_aucs.append(np.mean(aucs))
                if np.mean(aucs) > current_best:
                    current_best = np.mean(aucs)
                    best_try = f"{d}/try-{k}-aucs.txt"
        if len(m_aucs) > 0:
            mean_aucs.append(np.mean(m_aucs))
            std_aucs.append(np.std(m_aucs))
        else:
            mean_aucs.append(np.nan)
            std_aucs.append(np.nan)

        best_aucs.append(current_best)
    # print("best:",best_try,current_best)

    mean_aucs = np.array(mean_aucs)
    std_aucs = np.array(std_aucs)
    error_bars = np.array(error_bars)
    x = np.arange(budget)

    plt.plot(x, mean_aucs, color=color, linestyle=ls, label=label)
    plt.fill_between(
        x, mean_aucs - std_aucs, mean_aucs + std_aucs, color=color, alpha=0.05
    )
    # plt.fill_between(x, 0, 1, where=error_bars, color='r', alpha=0.2)
    plt.ylim(0.0, 0.7)
    plt.xlim(0, 100)
    # plt.plot(x, mean_2, 'r-', label='mean_2')
    # plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig(f"comparison_plot.png")
plt.clf()


## Plot diversity for each run (difference between pop and next pop)
import re

fig = plt.figure(figsize=(8, 6))


for i in range(len(exp_dirs)):  # len(exp_dirs)
    exp_dir = exp_dirs[i]
    color = colors[i]
    label = labels[i]
    ls = linestyles[i]
    current_best = []
    elitism = False

    all_locations = []
    all_aocs = []
    if i == 2 or i == 5 or i == 8:
        elitism = True

    ratios = [[], [], [], [], []]
    previous_codes = ["", "", "", "", ""]

    if False:  # and os.path.isfile(f"ratios{i}.npy"):
        ratios = np.load(f"ratios{i}.npy")
    else:
        for k in range(budget):
            for j in range(len(exp_dir)):
                d = exp_dir[j]
                # get code file
                found = False
                for filename in os.listdir(f"{d}/code/"):
                    if re.search(f"try-{k}-\w*\.py", filename):
                        # print(filename)
                        file = open(f"{d}/code/{filename}", "r")
                        algorithm_name = re.findall(
                            "try-\d*-(\w*)\.py", filename, re.IGNORECASE
                        )[0]
                        # print(" ".join(re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', algorithm_name)), end=' ')
                        new_code = file.read()
                        file.close()
                        if k == 0:
                            ratios[j].append(0)
                        else:
                            rat = code_compare(previous_codes[j], new_code)
                            ratios[j].append(rat)
                        if os.path.isfile(f"{d}/try-{k}-aucs.txt"):
                            aucs = np.loadtxt(f"{d}/try-{k}-aucs.txt")
                            # store all code files with at least an auc

                            with open(f"algorithms/{algorithm_name}.py", "w") as file:
                                file.write(new_code)
                            while j >= len(current_best):
                                current_best.append(0)
                            if current_best[j] <= np.mean(aucs):
                                current_best[j] = np.mean(aucs)
                                if elitism:
                                    previous_codes[j] = new_code
                            all_aocs.append(np.mean(aucs))
                            all_locations.append(f"{d}/code/{filename}")
                        if not elitism:
                            previous_codes[j] = new_code
                        found = True
                        break
                if not found:
                    ratios[j].append(0)
        ratios = np.array(ratios)
        mean_ratios = np.mean(ratios, axis=0)
        # std_aucs = np.array(std_aucs)
        # np.save(f"ratios{i}.npy", ratios)

    mean_ratios = np.mean(ratios, axis=0)
    x = np.arange(budget)
    for j in range(ratios.shape[0]):
        plt.plot(x, ratios[j], color=color, linestyle=ls, alpha=0.02)
    plt.plot(x, mean_ratios, color=color, linestyle=ls, label=label)
    # plt.fill_between(x, mean_aucs - std_aucs, mean_aucs + std_aucs, color=color, alpha=0.05)
    # plt.fill_between(x, 0, 1, where=error_bars, color='r', alpha=0.2)

    # best 3 algorithms per run
    # all_aocs = np.array(all_aocs)
    # all_locations = np.array(all_locations)
    # indices_best = np.argsort(all_aocs)[-3:]
    # print(all_locations[indices_best], all_aocs[indices_best])

    # plt.plot(x, mean_2, 'r-', label='mean_2')
    # plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
# plt.yscale("log")
# plt.xscale("log")
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 100)
plt.ylabel(
    "pairwise difference ratio"
)  # Bold lines depict averages over all runs, more transparent lines depict independent runs.
plt.xlabel("Iterations")
plt.legend()
plt.tight_layout()

plt.savefig(f"diff_plot0.pdf")
plt.clf()


# diff names
fig = plt.figure(figsize=(8, 4))


for i in range(len(exp_dirs)):  # len(exp_dirs)
    exp_dir = exp_dirs[i]
    color = colors[i]
    label = labels[i]
    ls = linestyles[i]
    current_best = []
    current_best_name = []
    elitism = False

    all_locations = []
    all_aocs = []
    if i == 2 or i == 5 or i == 8:
        elitism = True

    ratios = [[], [], [], [], []]
    ratios_name = [[], [], [], [], []]
    previous_codes = ["", "", "", "", ""]
    previous_names = ["", "", "", "", ""]

    if False:  # and os.path.isfile(f"ratios{i}.npy"):
        ratios = np.load(f"ratios{i}.npy")
    else:
        for k in range(budget):
            for j in range(len(exp_dir)):
                d = exp_dir[j]
                # get code file
                found = False
                for filename in os.listdir(f"{d}/code/"):
                    if re.search(f"try-{k}-\w*\.py", filename):
                        # print(filename)
                        algorithm_name = re.findall(
                            "try-\d*-(\w*)\.py", filename, re.IGNORECASE
                        )[0]
                        if k == 0:
                            ratios_name[j].append(1)
                        else:
                            rat2 = jellyfish.jaro_similarity(
                                previous_names[j], algorithm_name
                            )
                            ratios[j].append(rat)
                            ratios_name[j].append(rat2)
                        if os.path.isfile(f"{d}/try-{k}-aucs.txt"):
                            aucs = np.loadtxt(f"{d}/try-{k}-aucs.txt")
                            # store all code files with at least an auc

                            # with open(f"algorithms/{algorithm_name}.py", "w") as file:
                            #     file.write(new_code)
                            while j >= len(current_best):
                                current_best.append(0)
                            if current_best[j] <= np.mean(aucs):
                                current_best[j] = np.mean(aucs)
                                if elitism:
                                    previous_codes[j] = new_code
                                    previous_names[j] = algorithm_name
                            all_aocs.append(np.mean(aucs))
                        if not elitism:
                            previous_codes[j] = new_code
                            previous_names[j] = algorithm_name
                        found = True
                        break
                if not found:
                    ratios_name[j].append(0)
        print(ratios_name)
        ratios_name = np.array(ratios_name)
        mean_ratios_name = np.mean(ratios_name, axis=0)

    mean_ratios_name = np.mean(ratios_name, axis=0)
    x = np.arange(budget)
    # for j in range(ratios_name.shape[0]):
    #    plt.plot(x, ratios_name[j], color=color, linestyle=ls, alpha=0.02)
    plt.plot(x, mean_ratios_name, color=color, linestyle=ls, label=label)
    # plt.fill_between(x, mean_aucs - std_aucs, mean_aucs + std_aucs, color=color, alpha=0.05)
    # plt.fill_between(x, 0, 1, where=error_bars, color='r', alpha=0.2)

    # best 3 algorithms per run
    # all_aocs = np.array(all_aocs)
    # all_locations = np.array(all_locations)
    # indices_best = np.argsort(all_aocs)[-3:]
    # print(all_locations[indices_best], all_aocs[indices_best])

    # plt.plot(x, mean_2, 'r-', label='mean_2')
    # plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
# plt.yscale("log")
# plt.xscale("log")
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 100)
plt.ylabel(
    "Pairwise Jaro similarity"
)  # Bold lines depict averages over all runs, more transparent lines depict independent runs.
plt.xlabel("Iterations")
plt.legend()
plt.tight_layout()

plt.savefig(f"diff_plot_names.pdf")
plt.clf()


# diff names independent runs
# diff names
fig = plt.figure(figsize=(6, 4))


for i in range(len(exp_dirs)):  # len(exp_dirs)
    exp_dir = exp_dirs[i]
    color = colors[i]
    label = labels[i]
    ls = linestyles[i]
    current_best = []
    current_best_name = []
    elitism = False

    all_locations = []
    all_aocs = []
    if i == 2 or i == 5 or i == 8:
        elitism = True

    ratios = [[], [], [], [], []]
    ratios_name = [[], [], [], [], []]
    previous_codes = ["", "", "", "", ""]
    previous_names = ["", "", "", "", ""]

    if False:  # and os.path.isfile(f"ratios{i}.npy"):
        ratios = np.load(f"ratios{i}.npy")
    else:
        for k in range(budget):
            for j in range(len(exp_dir)):
                d = exp_dir[j]
                # get code file
                found = False
                for filename in os.listdir(f"{d}/code/"):
                    if re.search(f"try-{k}-\w*\.py", filename):
                        # print(filename)
                        algorithm_name = re.findall(
                            "try-\d*-(\w*)\.py", filename, re.IGNORECASE
                        )[0]
                        if k == 0:
                            ratios_name[j].append(1)
                        else:
                            rat2 = jellyfish.jaro_similarity(
                                previous_names[j], algorithm_name
                            )
                            ratios[j].append(rat)
                            ratios_name[j].append(rat2)
                        if os.path.isfile(f"{d}/try-{k}-aucs.txt"):
                            aucs = np.loadtxt(f"{d}/try-{k}-aucs.txt")
                            # store all code files with at least an auc

                            # with open(f"algorithms/{algorithm_name}.py", "w") as file:
                            #     file.write(new_code)
                            while j >= len(current_best):
                                current_best.append(0)
                            if current_best[j] <= np.mean(aucs):
                                current_best[j] = np.mean(aucs)
                                if elitism:
                                    previous_codes[j] = new_code
                                    previous_names[j] = algorithm_name
                            all_aocs.append(np.mean(aucs))
                        if not elitism:
                            previous_codes[j] = new_code
                            previous_names[j] = algorithm_name
                        found = True
                        break
                if not found:
                    ratios_name[j].append(0)
        print(ratios_name)
        ratios_name = np.array(ratios_name)
        mean_ratios_name = np.mean(ratios_name, axis=0)

    mean_ratios_name = np.mean(ratios_name, axis=0)
    x = np.arange(budget)
    for j in range(ratios_name.shape[0]):
        if j == 0:
            plt.plot(x, ratios_name[j], color=color, linestyle=ls, label=label)
        else:
            plt.plot(x, ratios_name[j], color=color, linestyle=ls)
    # plt.plot(x, mean_ratios_name, color=color, linestyle=ls, label=label)
    # plt.fill_between(x, mean_aucs - std_aucs, mean_aucs + std_aucs, color=color, alpha=0.05)
    # plt.fill_between(x, 0, 1, where=error_bars, color='r', alpha=0.2)

    # best 3 algorithms per run
    # all_aocs = np.array(all_aocs)
    # all_locations = np.array(all_locations)
    # indices_best = np.argsort(all_aocs)[-3:]
    # print(all_locations[indices_best], all_aocs[indices_best])

    # plt.plot(x, mean_2, 'r-', label='mean_2')
    # plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
    # plt.yscale("log")
    # plt.xscale("log")
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 100)
    plt.ylabel(
        "Pairwise Jaro similarity"
    )  # Bold lines depict averages over all runs, more transparent lines depict independent runs.
    plt.xlabel("Iterations")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"diff_plot_names_{i}.pdf")
    plt.clf()

# GPT 4
fig = plt.figure(figsize=(8, 6))
# exp_dirs = [["exp-05-07_14:52:37 baseline","exp-05-14_12:34:00-gpt-4-turbo-plain", "exp-05-14_12:34:25-gpt-4-turbo-plain"], ["exp-05-13_12:15:53 detailed", "exp-05-14_12:36:03-gpt-4-turbo-detail", "exp-05-14_12:36:15-gpt-4-turbo-detail"], ["exp-05-13_12:56:16 elitism", "exp-05-14_11:08:04-gpt-4-turbo-elitsm"], ["exp-05-14_12:37:38-gpt-4-turbo-detail-elitism"]
#     , ["exp-05-14_09:43:48"], ["exp-05-14_09:47:17"], ["exp-05-14_09:48:24"]]
# colors = ['b','r','g','k', 'c', 'm', 'y']
# labels = ['GPT-4-ES', 'GPT-4-ES w/ Detail', 'GPT-4-ES w/ Elitism', 'GPT-4-ES w/ Elitism+Detail', 'GPT-3.5-ES', 'GPT-3.5-ES w/ Details', 'GPT-3.5-ES w/ Elitism']
for i in range(len(exp_dirs)):
    exp_dir = exp_dirs[i]
    color = colors[i]
    label = labels[i]
    ls = linestyles[i]
    mean_aucs = []
    current_best = []
    best_aucs = []
    std_aucs = []
    error_bars = []
    best_try = ""
    for k in range(budget):
        m_aucs = []
        for j in range(len(exp_dir)):
            d = exp_dir[j]
            if os.path.isfile(f"{d}/try-{k}-aucs.txt"):
                aucs = np.loadtxt(f"{d}/try-{k}-aucs.txt")
                m_aucs.append(np.mean(aucs))
                while j >= len(current_best):
                    current_best.append(0)
                if current_best[j] < np.mean(aucs):
                    current_best[j] = np.mean(aucs)
        if len(current_best) > 0:
            mean_aucs.append(np.mean(current_best))
            std_aucs.append(np.std(current_best))
        else:
            mean_aucs.append(np.nan)
            std_aucs.append(np.nan)

    mean_aucs = np.array(mean_aucs)
    std_aucs = np.array(std_aucs)
    x = np.arange(budget)

    plt.plot(x, mean_aucs, color=color, linestyle=ls, label=label)
    plt.fill_between(
        x, mean_aucs - std_aucs, mean_aucs + std_aucs, color=color, alpha=0.05
    )
    # plt.fill_between(x, 0, 1, where=error_bars, color='r', alpha=0.2)

    # plt.plot(x, mean_2, 'r-', label='mean_2')
    # plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
# plt.yscale("symlog")
# plt.xscale("symlog")
plt.ylim(0.0, 0.6)
plt.xlim(0.0, 100)
plt.ylabel("mean AOCC")
plt.xlabel("Iterations")
plt.legend()
plt.tight_layout()

plt.savefig(f"convergence_plot.pdf")
plt.clf()
# plt.plot(x, best_aucs, 'b-', label='baseline')
# plt.ylim(0.0,1.0)
# #plt.plot(x, mean_2, 'r-', label='mean_2')
# #plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
# plt.legend()
# plt.tight_layout()
# plt.savefig(f"{exp_dir}/convergence_plot.png")

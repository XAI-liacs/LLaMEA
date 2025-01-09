# Simple helper file to plot generated auc files (used for the LLaMEA paper).
import re
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json
import difflib
import jellyfish

budget = 100


def code_compare(code1, code2):
    # Parse the Python code into ASTs
    # Use difflib to find differences
    diff = difflib.ndiff(code1.splitlines(), code2.splitlines())
    # Count the number of differing lines
    diffs = sum(1 for x in diff if x.startswith('- ') or x.startswith('+ '))
    # Calculate total lines for the ratio
    total_lines = max(len(code1.splitlines()), len(code2.splitlines()))
    similarity_ratio = (total_lines - diffs) / \
        total_lines if total_lines else 1
    return 1-similarity_ratio


def plot_aucs(exp_dirs, colors, labels, linestyles, title, aocc=False):
    for i in range(len(exp_dirs)):
        exp_dir = exp_dirs[i]
        color = colors[i]
        ls = linestyles[i]
        label = labels[i]
        mean_aucs = []
        current_best = [0. for _ in range(len(exp_dir))]
        best_aucs = []
        std_aucs = []
        for k in range(budget):
            m_aucs = []
            for j in range(len(exp_dir)):
                d = exp_dir[j]
                log_file = f"{d}/log.jsonl"
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) > k:
                        data = json.loads(lines[k])
                        if data["fitness"] == -np.inf:
                            m_aucs += [current_best[j]]
                        else:
                            m_aucs += [np.mean(data["metadata"]["aucs"])]
                            if np.mean(data["metadata"]["aucs"]) > current_best[j]:
                                current_best[j] = np.mean(
                                    data["metadata"]["aucs"])
            if not aocc:
                if len(m_aucs) > 0:
                    mean_aucs += [np.mean(m_aucs)]
                    std_aucs += [np.std(m_aucs)**2]
                else:
                    mean_aucs += [np.nan]
                    std_aucs += [np.nan]
            else:
                mean_aucs += [np.mean(current_best)]
                std_aucs += [np.std(current_best)**2]
        mean_aucs = np.array(mean_aucs)
        std_aucs = np.array(std_aucs)
        np.savetxt(f"results/{title}-{label}-aucs.txt", mean_aucs)
        np.savetxt(f"results/{title}-{label}-std.txt", std_aucs)
        x = np.arange(budget)
        plt.plot(x, mean_aucs, color=color, linestyle=ls, label=label)
        plt.fill_between(x, mean_aucs - std_aucs, mean_aucs +
                         std_aucs, color=color, alpha=0.05)
        plt.xlim(0, 100)
    plt.legend()
    plt.tight_layout()
    if not aocc:
        plt.savefig(f"results/{title}-aucs.png")
    else:
        plt.savefig(f"results/{title}-aocc.png")
    plt.clf()


if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")
    colors = ['b', 'r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r', 'g']
    linestyles = ['solid', 'solid', 'solid',
                  'dotted', 'dotted', 'dotted',
                  'dashed', 'dashed', 'dashed',
                  'dashdot', 'dashdot', 'dashdot']

    models = ["gpt-3.5-turbo", "gpt-4o"]
    mutations = [f"exp{i}" for i in range(1, 3)]
    for model in models:
        exp_dirs = []
        labels = []
        for mutation in mutations:
            folders = os.listdir(
                f"exp_data/CAI/rerun/exp-{model}-{mutation}")
            exp_dirs += [
                [f"exp_data/CAI/rerun/exp-{model}-{mutation}/{f}" for f in folders if f.startswith("exp")]]
            labels += [f"{mutation}"]
        plot_aucs(exp_dirs, colors, labels, linestyles, model)
        plot_aucs(exp_dirs, colors, labels, linestyles, model, aocc=True)
    # for mutation in mutations:
    #     exp_dirs = []
    #     labels = []
    #     for model in models:
    #         if model == "Meta-Llama-3.1-8B" and mutation != "beta1.5":
    #             continue
    #         if model == "Meta-Llama-3.1-70B" and mutation != "beta1.5":
    #             continue
    #         folders = os.listdir(
    #             f"/scratch/hyin/LLaMEA/exp-{model}/{mutation}")
    #         exp_dirs += [
    #             [f"/scratch/hyin/LLaMEA/exp-{model}/{mutation}/{f}" for f in folders if f.startswith("exp")]]
    #         labels += [f"{model}"]
    #     plot_aucs(exp_dirs, colors, labels, linestyles, mutation)
    #     plot_aucs(exp_dirs, colors, labels, linestyles, mutation, aocc=True)

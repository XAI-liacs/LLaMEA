# Simple helper file to plot generated auc files.
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import tqdm
import ast
import difflib
import jellyfish
import jsonlines


def code_compare(code1, code2, printdiff=False):
    # Parse the Python code into ASTs
    # Use difflib to find differences
    diff = difflib.ndiff(code1.splitlines(), code2.splitlines())
    # Count the number of differing lines
    diffs = sum(1 for x in diff if x.startswith("- ") or x.startswith("+ "))
    # Calculate total lines for the ratio
    total_lines = max(len(code1.splitlines()), len(code2.splitlines()))
    similarity_ratio = (total_lines - diffs) / total_lines if total_lines else 1
    return 1 - similarity_ratio


experiments_dirs = [
    "exp-08-20_122254-gpt-4o-2024-05-13-ES gpt-4o-HPO",  # /log.jsonl
    "exp-08-20_123922-gpt-4o-2024-05-13-ES gpt-4o-HPO",
    "exp-08-26_090633-gpt-4o-2024-05-13-ES gpt-4o-HPO",
    "exp-08-26_090643-gpt-4o-2024-05-13-ES gpt-4o-HPO",
    "exp-08-26_090744-gpt-4o-2024-05-13-ES gpt-4o-HPO",
]
budget = 100

label_main = "GPT-4o-HPO"

convergence_lines = []
code_diff_ratios_lines = []


for i in range(len(experiments_dirs)):
    convergence = np.zeros(budget)
    code_diff_ratios = np.zeros(budget)
    best_so_far = -np.Inf
    previous_code = ""
    log_file = experiments_dirs[i] + "/log.jsonl"
    if os.path.exists(log_file):
        with jsonlines.open(log_file) as reader:
            for obj in reader.iter(type=dict, skip_invalid=True):
                gen = 0
                fitness = None
                code_diff = 0
                code = ""
                if "_solution" in obj.keys():
                    code = obj["_solution"]
                if "_generation" in obj.keys():
                    gen = obj["_generation"]
                if "_fitness" in obj.keys():
                    fitness = obj["_fitness"]
                else:
                    fitness = None

                if fitness <= best_so_far:
                    code_diff = code_compare(previous_code, code, False)
                else:
                    code_diff = code_compare(previous_code, code, True)
                    best_so_far = fitness
                    previous_code = code
                code_diff_ratios[gen] = code_diff
                convergence[gen] = fitness

    # now fix the holes
    best_so_far = 0
    for i in range(len(convergence)):
        if convergence[i] >= best_so_far:
            best_so_far = convergence[i]
        else:
            convergence[i] = best_so_far
    convergence_lines.append(convergence)
    code_diff_ratios_lines.append(code_diff_ratios)


plt.figure(figsize=(6, 4))
for i in range(len(convergence_lines)):
    plt.plot(np.arange(budget), convergence_lines[i], linestyle="dashed")

# convergence curves
mean_convergence = np.array(convergence_lines).mean(axis=0)
std = np.array(convergence_lines).std(axis=0)
plt.plot(
    np.arange(budget),
    mean_convergence,
    color="b",
    linestyle="solid",
    label=label_main,
)
plt.fill_between(
    np.arange(budget),
    mean_convergence - std,
    mean_convergence + std,
    color="b",
    alpha=0.05,
)
# plt.fill_between(x, 0, 1, where=error_bars, color='r', alpha=0.2)
plt.ylim(0.0, 0.7)
plt.xlim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig(f"plot_aucs_HPO.png")
plt.clf()


# Code diff curves
plt.figure(figsize=(6, 4))
for i in range(len(code_diff_ratios_lines)):
    plt.plot(np.arange(budget), code_diff_ratios_lines[i], linestyle="dashed")

mean_code_diff = np.array(code_diff_ratios_lines).mean(axis=0)
std = np.array(code_diff_ratios_lines).std(axis=0)
plt.plot(
    np.arange(budget),
    mean_code_diff,
    color="b",
    linestyle="solid",
    label=label_main,
)
plt.fill_between(
    np.arange(budget),
    mean_code_diff - std,
    mean_code_diff + std,
    color="b",
    alpha=0.05,
)
# plt.fill_between(x, 0, 1, where=error_bars, color='r', alpha=0.2)
plt.ylim(0.0, 1.0)
plt.xlim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig(f"plot_diffratio_HPO.png")
plt.clf()

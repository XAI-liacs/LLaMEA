# Simple helper file to plot generated auc files.
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import difflib
import jsonlines
import json
from ioh import get_problem, logger
from misc import aoc_logger, correct_aoc, OverBudgetException

try:
    from colorama import Fore, Back, Style, init
    init()
except ImportError:  # fallback so that the imported classes always exist
    class ColorFallback():
        __getattr__ = lambda self, name: ''
    Fore = Back = Style = ColorFallback()

def color_diff(diff):
    for line in diff:
        if line.startswith('+'):
            yield Fore.GREEN + line + Fore.RESET
        elif line.startswith('-'):
            yield Fore.RED + line + Fore.RESET
        elif line.startswith('^'):
            yield Fore.BLUE + line + Fore.RESET
        else:
            yield line

def code_compare(code1, code2, printdiff=False):
    # Parse the Python code into ASTs
    # Use difflib to find differences
    diff = difflib.ndiff(code1.splitlines(), code2.splitlines())
    
    # Count the number of differing lines
    diffs = sum(1 for x in diff if x.startswith("- ") or x.startswith("+ "))
    if printdiff and code1 != "":
        diff = difflib.ndiff(code1.splitlines(), code2.splitlines())
        print('\n'.join(color_diff(diff)))
    # Calculate total lines for the ratio
    total_lines = max(len(code1.splitlines()), len(code2.splitlines()))
    similarity_ratio = (total_lines - diffs) / total_lines if total_lines else 1
    if printdiff:
        print(similarity_ratio)
    return 1 - similarity_ratio



## Plot BP curves
experiments_dirs = [
    "exp-08-29_201655-gpt-4o-2024-05-13-ES TSP-HPO",
    #"exp-08-30_114546-gpt-4o-2024-05-13-ES TSP-HPO",
    #"exp-08-30_114606-gpt-4o-2024-05-13-ES TSP-HPO",
    "exp-08-30_142330-gpt-4o-2024-05-13-ES TSP-HPO-deter",
    "exp-09-02_105043-gpt-4o-2024-05-13-ES TSP-HPO-deter"
]
budget = 100

label_main = "LLaMEA-HPO"

convergence_lines = []
convergence_default_lines = []
code_diff_ratios_lines = []
best_ever_name = ""
best_ever_code = ""
best_ever_fitness = -np.Inf
best_ever_config = {}

for i in range(len(experiments_dirs)):
    convergence = np.ones(budget) * -100
    #convergence_default = np.zeros(budget)
    code_diff_ratios = np.zeros(budget)
    best_so_far = -np.Inf
    best_so_far_default = 0
    previous_code = ""
    previous_config = {}
    previous_name = ""
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

                if fitness > best_ever_fitness:
                    best_ever_fitness = fitness
                    best_ever_code = code
                    best_ever_config = obj["incumbent"]
                    best_ever_name = obj["_name"]

                if fitness >= -1.850371707708594e-10:
                    name = obj["_name"]
                    if False:
                        file = open(f"problems/user_tsp_gls/algs/{name}.py", 'w')
                        file.write(code)
                        conf = obj["incumbent"]
                        maincall = f"""
# initialize
config = {conf}
scoringalg = {name}(**config)

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # {fitness}
    return scoringalg.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)"""
                        file.write(maincall)
                        file.close()

                if fitness <= best_so_far:
                    code_diff = code_compare(previous_code, code, False)
                else:
                    name = obj["_name"]
                    #print(f"-- {gen} -- {previous_name} --> {name}")
                    code_diff = code_compare(previous_code, code, False)
                    best_so_far = fitness

                    previous_code = code
                    previous_name = name
                    previous_config = obj["incumbent"]
                
                code_diff_ratios[gen] = code_diff
                convergence[gen] = fitness
    #print("Best algorithm: ", previous_name, "with config:", previous_config)
    #print(previous_code)


    # now fix the holes
    best_so_far = -np.Inf
    for i in range(len(convergence)):
        if convergence[i] >= best_so_far:
            best_so_far = convergence[i]
        else:
            convergence[i] = best_so_far
    convergence_lines.append(convergence)
    code_diff_ratios_lines.append(code_diff_ratios)

print("Best algorithm: ", best_ever_name, "with config:", best_ever_config)
print(best_ever_code)

x_line = np.arange(budget)

plt.figure(figsize=(6, 4))
convergence_lines_HPO = convergence_lines
for i in range(len(convergence_lines)):
    plt.plot(x_line, -1*convergence_lines[i], linestyle="dashed",color='C0')
    #print(convergence_lines[i])

# convergence curves

#np.save("HPO-BPconvergence_lines.npy", convergence_lines)
mean_convergence_HPO = -1 * np.array(convergence_lines).mean(axis=0)
std_HPO = np.array(convergence_lines).std(axis=0)
plt.plot(
    x_line,
    mean_convergence_HPO,
    color="C0",
    linestyle="solid",
    label=label_main,
)
plt.fill_between(
    x_line,
    mean_convergence_HPO - std_HPO,
    mean_convergence_HPO + std_HPO,
    color="C0",
    alpha=0.05,
)


#Plot the EOH baseline runs
exp_dirs = ["EoHresults/Prob2_TSP_GLS/run1", "EoHresults/Prob2_TSP_GLS/run2", "EoHresults/Prob2_TSP_GLS/run3"]
convergence_lines = []
for exp_dir in exp_dirs:
    conv_line = np.ones(budget*100) * -np.Inf
    best_so_far = -np.Inf
    teller = 0
    for k in range(20):
        with open(exp_dir + f"/population_generation_{k}.json") as f:
            pop = json.load(f)
        for ind in pop:
            if -1*ind["objective"] > best_so_far:
                best_so_far = -1*ind["objective"]
            conv_line[teller] = best_so_far
            if k == -1:
                teller+=1
            else:
                for x in range(10):#EhO creates 5 offspring per individual and uses 2 prompts per eval
                    conv_line[teller] = best_so_far
                    teller+=1
        if teller > budget*100:
            break
    convergence_lines.append(np.array(conv_line))


for i in range(len(convergence_lines)):
    plt.plot(np.arange(len(convergence_lines[i])), -convergence_lines[i], linestyle="dotted",color='C1')

mean_convergence = -1 * np.array(convergence_lines).mean(axis=0)
std = np.array(convergence_lines).std(axis=0)
plt.plot(
    np.arange(budget*100),
    mean_convergence,
    color="C1",
    linestyle="solid",
    label="EoH",
)
plt.fill_between(
    np.arange(budget*100),
    mean_convergence - std,
    mean_convergence + std,
    color="C1",
    alpha=0.05,
)
# plt.fill_between(x, 0, 1, where=error_bars, color='r', alpha=0.2)
#plt.ylim(0.0, 0.04)

#plt.yscale('symlog')
#plt.xscale('symlog')
plt.ylabel("Objective")
plt.xlabel("LLM prompts")
plt.title("Convergence on TSP problems")
plt.ylim(0, 0.5)
plt.xlim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig(f"plot_TSP_HPO_prompts.png")


plt.clf()
### Plot per benchmark evaluation
x_line = np.arange(budget) * 5 #5 full benchmark evaluations per prompt for LLaMEA-HPO

for i in range(len(convergence_lines_HPO)):
    plt.plot(x_line, -convergence_lines_HPO[i], linestyle="dotted",color='C1')
#mean_convergence_HPO = -1 * np.array(convergence_lines).mean(axis=0)
#std_HPO = np.array(convergence_lines).std(axis=0)
plt.plot(
    x_line,
    mean_convergence_HPO,
    color="C0",
    linestyle="solid",
    label=label_main,
)
plt.fill_between(
    x_line,
    mean_convergence_HPO - std_HPO,
    mean_convergence_HPO + std_HPO,
    color="C0",
    alpha=0.05,
)


#Plot the EOH baseline runs
exp_dirs = ["EoHresults/Prob2_TSP_GLS/run1", "EoHresults/Prob2_TSP_GLS/run2", "EoHresults/Prob2_TSP_GLS/run3"]
convergence_lines = []
for exp_dir in exp_dirs:
    conv_line = np.ones(budget*100) * -np.Inf
    best_so_far = -np.Inf
    teller = 0
    for k in range(20):
        with open(exp_dir + f"/population_generation_{k}.json") as f:
            pop = json.load(f)
        for ind in pop:
            if -1*ind["objective"] > best_so_far:
                best_so_far = -1*ind["objective"]
            conv_line[teller] = best_so_far
            if k == -1:
                teller+=1
            else:
                for x in range(5):#EhO creates 5 offspring per individual
                    conv_line[teller] = best_so_far
                    teller+=1
        if teller > budget*100:
            break
    convergence_lines.append(np.array(conv_line))

for i in range(len(convergence_lines)):
    plt.plot(np.arange(len(convergence_lines[i])), -convergence_lines[i], linestyle="dotted",color='C1')

mean_convergence = -1 * np.array(convergence_lines).mean(axis=0)
std = np.array(convergence_lines).std(axis=0)
plt.plot(
    np.arange(budget*100),
    mean_convergence,
    color="C1",
    linestyle="solid",
    label="EoH",
)
plt.fill_between(
    np.arange(budget*100),
    mean_convergence - std,
    mean_convergence + std,
    color="C1",
    alpha=0.05,
)
# plt.fill_between(x, 0, 1, where=error_bars, color='r', alpha=0.2)
#plt.ylim(0.0, 0.04)

#plt.yscale('symlog')
#plt.xscale('symlog')
plt.ylabel("Objective")
plt.xlabel("Benchmark evaluations")
plt.title("Convergence on TSP problems")
plt.ylim(0, 0.5)
plt.xlim(0, 500)
plt.legend()
plt.tight_layout()
plt.savefig(f"plot_TSP_HPO_eval.png")


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
plt.savefig(f"plot_diffratio_TSP_HPO.png")
plt.clf()

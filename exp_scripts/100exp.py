import os
import json
import difflib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def check_exp_num(files):
    mutations = {"2": 0, "5": 0, "10": 0, "20": 0}
    for f in files:
        mutation = f.split("-")[-1]
        mutations[mutation] += 1
    print(mutations)


def check_same_code(files, exp_dir):
    countera = 0
    counterb = 0
    for f in files:
        codes = os.listdir(f"{exp_dir}/{f}/code/")
        if len(codes) != 2:
            continue
        with open(f"{exp_dir}/{f}/code/{codes[0]}", "r") as file:
            code1 = file.readlines()
        with open(f"{exp_dir}/{f}/code/{codes[1]}", "r") as file:
            code2 = file.readlines()
        if code_compare(code1, code2) == 0:
            countera += 1
            log_file = f"{exp_dir}/{f}/log.jsonl"
            with open(log_file, 'r', encoding='utf-8') as fread:
                lines = fread.readlines()
                line = lines[1]
                data = json.loads(line)
                if "No code was extracted." in data["feedback"]:
                    # print(data["feedback"])
                    counterb += 1
                    print(f"No code was extracted. {f}/conversationlog.jsonl")
                # else:
                #     print("==============")
                #     print(data["feedback"])
        else:
            print(f"{f}/conversationlog.jsonl")
            # log_data = []
            # with open(log_file
    print(counterb / countera)
    #         log_data += [data]
    # for data in log_data:
    #     print(data["feedback"])


def code_compare(code1, code2):
    modified_lines = 0
    total_lines = max(len(code1), len(code2))
    matcher = difflib.SequenceMatcher(None, code1, code2)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            modified_lines += max(i2 - i1, j2 - j1)
        elif tag in ('delete', 'insert'):
            modified_lines += (i2 - i1) if tag == 'delete' else (j2 - j1)
    diff_ratio = modified_lines / total_lines if total_lines > 0 else 0
    return diff_ratio


def code_diff(files, exp_dir):
    diffs = {"2": [], "5": [], "10": [], "20": []}
    for f in files:
        mutation = f.split("-")[-1]
        codes = os.listdir(f"{exp_dir}/{f}/code/")
        codes = sorted(codes, key=lambda x: int(x.split('-')[1]))
        for i in range(1, len(codes)):
            with open(f"{exp_dir}/{f}/code/{codes[0]}", "r") as file:
                code1 = file.readlines()
            with open(f"{exp_dir}/{f}/code/{codes[i]}", "r") as file:
                code2 = file.readlines()
            diff_ratio = code_compare(code1, code2)
            if diff_ratio == 0:
                continue
            diffs[mutation].append(diff_ratio)
    print(f"Code difference calculation for {exp_dir} done.")
    for mutation in diffs:
        if len(diffs[mutation]) == 0:
            continue
        print(f"{len(diffs[mutation])} samples, mutation {mutation}: " +
              f"{sum(diffs[mutation]) / len(diffs[mutation])}")
    return diffs


def plot_diffs(diffs, exp_name):
    # draw violin plot of diffs for each mutation
    sns.violinplot(data=[diffs["2"], diffs["5"],
                   diffs["10"], diffs["20"]], cut=True)
    plt.xticks([0, 1, 2, 3], ["2", "5", "10", "20"])
    plt.savefig(f"{exp_name}-diffs.png")
    plt.clf()


def violin_plot(df, prompt, llm, title):
    sns.violinplot(x="Mutation", y="Code Difference", data=df, cut=0,
                   inner="stick", palette="muted", hue="Mutation",
                   legend=False)
    plt.xlabel("requested mutation rate")
    plt.ylabel("code difference")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.xticks(ticks=list(range(5)), labels=["2%", "5%", "10%", "20%", "40%"])
    plt.title(f"Code Difference Distribution\nwhen Using {prompt} with {llm}")
    plt.tight_layout()
    plt.savefig(title)
    plt.cla()


def ratio_plot(df, prompt, llm, title):
    df_temp = df.copy()
    df_temp["ratio"] = (df["Code Difference"] * 100) / df["Mutation"]
    sns.stripplot(x="Mutation", y="ratio", data=df_temp, jitter=True,
                  palette="muted", hue="Mutation", legend=False)
    plt.axhline(y=1, color='r', linestyle='--',
                label="delivered code difference = requested mutation rate")
    plt.yscale("log")
    plt.xticks(ticks=list(range(5)), labels=["2%", "5%", "10%", "20%", "40%"])
    plt.xlabel("requested mutation rate")
    plt.ylabel("ratio")
    plt.title(
        f"Ratio of Delivered Code Difference to Requested Mutation Rate\nwhen Using {prompt} with {llm}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(title)
    plt.cla()


# os.chdir("exp_data/CAI")
# exp_dirs = ["100-3.5", "100-4o", "100-Llama-3.3"]
# models = ["gpt-3.5-turbo", "gpt-4o", "Llama-3.3"]
# for exp_dir in exp_dirs:
#     model = models[exp_dirs.index(exp_dir)]
#     files = os.listdir(exp_dir)
#     # check_exp_num(files)
#     # check_same_code(files, exp_dir)
#     diffs = code_diff(files, exp_dir)
#     keys = ["model", "prompt", "Mutation",
#             "Code Difference", "Requested", "iteration"]
#     values = []
#     for mutation in ["2", "5", "10", "20"]:
#         if len(diffs[mutation]) == 0:
#             continue
#         for i in range(len(diffs[mutation])):
#             v = diffs[mutation][i]
#             values += [[model, "prompt5", float(mutation), v,
#                        float(mutation), i+1]]
#     df = pd.DataFrame(values, columns=keys)
#     title1 = f"/scratch-shared/hyin/LLaMEA/results/CAI/{model}_code-diff.png"
#     title2 = f"/scratch-shared/hyin/LLaMEA/results/CAI/{model}_ratio.png"
#     print(f"Plotting prompt5 with {model}...")
#     violin_plot(df, "prompt5", model, title1)
#     ratio_plot(df, "prompt5", model, title2)

os.chdir("exp_data/CAI/100-4o/new-100")
keys = ["model", "prompt", "Mutation",
        "Code Difference", "Requested", "iteration"]
values = []
for exp_dir in os.listdir("."):
    exp_id = exp_dir.split("-")[-1]
    mutation = exp_dir.split("-")[-2].split("_")[-1]
    codes = os.listdir(f"./{exp_dir}/code/")
    codes = sorted(codes, key=lambda x: int(x.split('-')[1]))
    with open(f"./{exp_dir}/code/{codes[0]}", "r") as f:
        code1 = f.readlines()
    with open(f"./{exp_dir}/code/{codes[1]}", "r") as f:
        code2 = f.readlines()
    diff_ratio = code_compare(code1, code2)
    values += [["gpt-4o", "prompt5", float(mutation), diff_ratio,
                float(mutation), int(exp_id)]]
df = pd.DataFrame(values, columns=keys)
title1 = f"/scratch/hyin/LLaMEA/results/CAI/gpt-4o_code-diff_differnt-code.png"
title2 = f"/scratch/hyin/LLaMEA/results/CAI/gpt-4o_ratio_differnt-code.png"
print(f"Plotting prompt5 with gpt-4o...")
violin_plot(df, "prompt5", "gpt-4o", title1)
ratio_plot(df, "prompt5", "gpt-4o", title2)

    # # check_exp_num(files)
    # # check_same_code(files, exp_dir)
    # diffs = code_diff(files, exp_dir)
    # keys = ["model", "prompt", "Mutation",
    #         "Code Difference", "Requested", "iteration"]
    # values = []
    # for mutation in ["2", "5", "10", "20"]:
    #     if len(diffs[mutation]) == 0:
    #         continue
    #     for i in range(len(diffs[mutation])):
    #         v = diffs[mutation][i]
    #         values += [[model, "prompt5", float(mutation), v,
    #                    float(mutation), i+1]]
    # df = pd.DataFrame(values, columns=keys)
    # title1 = f"/scratch-shared/hyin/LLaMEA/results/CAI/{model}_code-diff.png"
    # title2 = f"/scratch-shared/hyin/LLaMEA/results/CAI/{model}_ratio.png"
    # print(f"Plotting prompt5 with {model}...")
    # violin_plot(df, "prompt5", model, title1)
    # ratio_plot(df, "prompt5", model, title2)

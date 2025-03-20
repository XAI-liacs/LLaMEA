import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def split_dat(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    table_value = []
    runs = 1
    xval = 1
    yval_min = np.inf
    for line in lines:
        if line[:11] == "evaluations":
            if table_value != []:
                runs += 1
                xval = 1
                yval_min = np.inf
        else:
            elements = line[:-2].split(" ")
            yval = float(elements[1])
            yval_min = min(yval_min, yval)
            table_value += [[xval, yval, yval_min, runs]]
            xval += 1
    df = pd.DataFrame(table_value, columns=["xval", "yval", "yval_min", "run"])
    return df


def plot_convergence(df_best_algs, df_baselines, title):
    cud = ["#e69f00", "#56b4e9", "#009e73", "#f0e442",
           "#0072b2", "#d55e00", "#cc79a7", "#000000"]
    maker = ["o", "s", "D", "v", "^", ">", "<", "p"]
    i = 0
    baseline_algs = ["CMA", "BFGS", "DE", "QODE", "QNDE"]
    xvals = df_baselines["xval"].unique()
    df_best_alg_extracts = [df[df["xval"].isin(xvals)] for df in df_best_algs]
    for alg in baseline_algs:
        df_alg = df_baselines[df_baselines["alg"] == alg]
        sns.lineplot(x="xval", y="yval", data=df_alg,
                     label=alg, color=cud[i], marker=maker[i])
        i += 1
    j = 0
    for df_best_alg_extract in df_best_alg_extracts:
        if j == 0:
            temp_label = "1st best algorithm\nfound by LLaMEA"
        elif j == 1:
            temp_label = "2nd best algorithm\nfound by LLaMEA"
        else:
            temp_label = "3rd best algorithm\nfound by LLaMEA"
        sns.lineplot(x="xval", y="yval_min", data=df_best_alg_extract,
                    label=temp_label, color=cud[i], marker=maker[i])
        i += 1
        j += 1
    plt.xscale("log")
    if "Ellipsometry" not in title:
        plt.yscale("log")
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    # plt.title(f"{title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/CAI/{title}.png")
    plt.cla()


def plot_final_fitness_distribution(df_best_algs, df_baselines, title):
    cud = ["#e69f00", "#56b4e9", "#009e73", "#f0e442",
           "#0072b2", "#d55e00", "#cc79a7", "#000000"]
    maker = ["o", "s", "D", "v", "^", ">", "<", "p"]
    i = 0
    baseline_algs = ["CMA", "BFGS", "DE", "QODE", "QNDE"]
    xvals = df_baselines["xval"].unique()
    df_best_alg_extracts = [df[df["xval"].isin(xvals)] for df in df_best_algs]
    df_baseline_final = df_baselines[df_baselines["xval"] == np.max(xvals)]
    df_baseline_final = df_baseline_final.drop("run", axis=1)
    df_baseline_final = df_baseline_final.drop("xval", axis=1)
    df_best_alg_finals = [df[df["xval"] == np.max(xvals)] for df in df_best_alg_extracts]
    df_best_alg_finals = [df.drop("run", axis=1) for df in df_best_alg_finals]
    df_best_alg_finals = [df.drop("xval", axis=1) for df in df_best_alg_finals]
    df_best_alg_finals = [df.drop("yval", axis=1) for df in df_best_alg_finals]
    for i in range(len(df_best_alg_finals)):
        df_best_alg_final = df_best_alg_finals[i]
        for value in df_best_alg_final.values:
            if i == 0:
                new_row = pd.DataFrame(
                    {"alg": "1st Best\nalgorithm\nfound by\nLLaMEA", "yval": value[0]}, index=[0])
            elif i == 1:
                new_row = pd.DataFrame(
                    {"alg": "2nd Best\nalgorithm\nfound by\nLLaMEA", "yval": value[0]}, index=[0])
            else:
                new_row = pd.DataFrame(
                    {"alg": "3rd Best\nalgorithm\nfound by\nLLaMEA", "yval": value[0]}, index=[0])
            df_baseline_final = pd.concat(
                [df_baseline_final, new_row], ignore_index=True)
    sns.boxplot(x="alg", y="yval", data=df_baseline_final, palette=cud)
    sns.stripplot(x="alg", y="yval", data=df_baseline_final,
                  color="black", size=5, jitter=True, edgecolor="white", linewidth=1)
    plt.yscale("log")
    plt.xlabel("Algorithms")
    plt.ylabel("Fitness")
    # plt.title(f"{title}")
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/CAI/{title}_boxplot.png")
    plt.cla()


def build_photovoltaic_baselines():
    folder_path = "exp_data/CAI/benchmark/baselines/photovoltaics/"
    files = os.listdir(folder_path)
    df_photovoltaic = []
    df_bigphotovoltaic = []
    df_hugephotovoltaic = []
    for f_name in files:
        if "big" in f_name:
            df_bigphotovoltaic.append(pd.read_csv(folder_path + f_name))
        elif "huge" in f_name:
            df_hugephotovoltaic.append(pd.read_csv(folder_path + f_name))
        else:
            df_photovoltaic.append(pd.read_csv(folder_path + f_name))
    df_photovoltaic = pd.concat(df_photovoltaic)
    df_bigphotovoltaic = pd.concat(df_bigphotovoltaic)
    df_hugephotovoltaic = pd.concat(df_hugephotovoltaic)
    df_photovoltaic.to_csv(
        "exp_data/CAI/benchmark/baselines/photovoltaic.csv", index=False)
    df_bigphotovoltaic.to_csv(
        "exp_data/CAI/benchmark/baselines/bigphotovoltaic.csv", index=False)
    df_hugephotovoltaic.to_csv(
        "exp_data/CAI/benchmark/baselines/hugephotovoltaic.csv", index=False)


def build_photovoltaic_best_alg():
    folder_path = "exp_data/CAI/benchmark/best_algs/photovoltaics/"
    files = os.listdir(folder_path)
    df_photovoltaic = []
    df_bigphotovoltaic = []
    df_hugephotovoltaic = []
    for f_name in files:
        elements = f_name.split("_")
        dim = int(elements[1])
        run_id = int(elements[2])
        df_temp = split_dat(
            f"{folder_path}{f_name}/data_f1123_sophisticated_antireflection_design/IOHprofiler_f1123_DIM{dim}.dat")
        if dim == 10:
            df_photovoltaic.append(df_temp)
        elif dim == 20:
            df_bigphotovoltaic.append(df_temp)
        else:
            df_hugephotovoltaic.append(df_temp)
    df_photovoltaic = pd.concat(df_photovoltaic)
    df_bigphotovoltaic = pd.concat(df_bigphotovoltaic)
    df_hugephotovoltaic = pd.concat(df_hugephotovoltaic)
    df_photovoltaic.to_csv(
        "exp_data/CAI/benchmark/best_algs/photovoltaic.csv", index=False)
    df_bigphotovoltaic.to_csv(
        "exp_data/CAI/benchmark/best_algs/bigphotovoltaic.csv", index=False)
    df_hugephotovoltaic.to_csv(
        "exp_data/CAI/benchmark/best_algs/hugephotovoltaic.csv", index=False)


os.chdir("exp_data/CAI/benchmark")
df_mini_bragg_baseline = pd.read_csv("baselines/minibragg.csv")
df_mini_bragg_best_algs = [
    split_dat("mini_bragg_best_alg3_HybridOptimization/data_f1121_mini_bragg/IOHprofiler_f1121_DIM10.dat"),
    split_dat("mini_bragg_best_alg4_HybridOptimization/data_f1121_mini_bragg/IOHprofiler_f1121_DIM10.dat"),
    split_dat("mini_bragg_best_alg6_HybridOptimization/data_f1121_mini_bragg/IOHprofiler_f1121_DIM10.dat")
]
df_bragg_baseline = pd.read_csv("baselines/bragg.csv")
df_bragg_best_algs = [
    split_dat("bragg_best_alg3_HybridOptimization/data_f1122_bragg/IOHprofiler_f1122_DIM20.dat"),
    split_dat("bragg_best_alg4_HybridOptimization/data_f1122_bragg/IOHprofiler_f1122_DIM20.dat"),
    split_dat("bragg_best_alg6_HybridOptimization/data_f1122_bragg/IOHprofiler_f1122_DIM20.dat"),
]
df_ellipsometry_baseline = pd.read_csv("baselines/ellipsometry.csv")
df_ellipsometry_best_algs = [
    split_dat("ellipsometry_best_alg0_AdaptiveBoundaryRefinement/data_f1123_ellipsometry/IOHprofiler_f1123_DIM2.dat"),
    split_dat("ellipsometry_best_alg1_AdaptiveBoundaryRefinement/data_f1123_ellipsometry/IOHprofiler_f1123_DIM2.dat"),
    split_dat("ellipsometry_best_alg2_AdaptiveBoundaryRefinement/data_f1123_ellipsometry/IOHprofiler_f1123_DIM2.dat"),
]
df_photovoltaic_baseline = pd.read_csv("baselines/photovoltaic.csv")
df_photovoltaic_best_algs = [
    split_dat("photovoltaic_best_alg0_HSBO/data_f1124_photovoltaic/IOHprofiler_f1124_DIM10.dat"),
    split_dat("photovoltaic_best_alg1_HSBO/data_f1124_photovoltaic/IOHprofiler_f1124_DIM10.dat"),
    split_dat("photovoltaic_best_alg2_HSBO/data_f1124_photovoltaic/IOHprofiler_f1124_DIM10.dat"),
]
df_big_photovoltaic_best_algs = [
    split_dat("big_photovoltaic_best_alg0_HSBO/data_f1125_big_photovoltaic/IOHprofiler_f1125_DIM20.dat"),
    split_dat("big_photovoltaic_best_alg1_HSBO/data_f1125_big_photovoltaic/IOHprofiler_f1125_DIM20.dat"),
    split_dat("big_photovoltaic_best_alg2_HSBO/data_f1125_big_photovoltaic/IOHprofiler_f1125_DIM20.dat"),
]
df_huge_photovoltaic_best_algs = [
    split_dat("huge_photovoltaic_best_alg0_HSBO/data_f1121_huge_photovoltaic/IOHprofiler_f1121_DIM32.dat"),
    split_dat("huge_photovoltaic_best_alg1_HSBO/data_f1121_huge_photovoltaic/IOHprofiler_f1121_DIM32.dat"),
    split_dat("huge_photovoltaic_best_alg2_HSBO/data_f1121_huge_photovoltaic/IOHprofiler_f1121_DIM32.dat"),
]
df_bigphotovoltaic_baseline = pd.read_csv("baselines/bigphotovoltaic.csv")
# df_bigphotovoltaic_best_alg = pd.read_csv("best_algs/bigphotovoltaic.csv")
df_hugephotovoltaic_baseline = pd.read_csv("baselines/hugephotovoltaic.csv")
# df_hugephotovoltaic_best_alg = pd.read_csv("best_algs/hugephotovoltaic.csv")


os.chdir("../../..")
plot_convergence(df_mini_bragg_best_algs, df_mini_bragg_baseline, "Mini Bragg")
plot_convergence(df_bragg_best_algs, df_bragg_baseline, "Bragg")
plot_convergence(df_ellipsometry_best_algs,
                 df_ellipsometry_baseline, "Ellipsometry")
plot_convergence(df_photovoltaic_best_algs,
                 df_photovoltaic_baseline, "Photovoltaics")
plot_convergence(df_big_photovoltaic_best_algs,
                 df_bigphotovoltaic_baseline, "Big Photovoltaics")
plot_convergence(df_huge_photovoltaic_best_algs,
                 df_hugephotovoltaic_baseline, "Huge Photovoltaics")
plot_final_fitness_distribution(df_mini_bragg_best_algs,
                                df_mini_bragg_baseline, "Mini Bragg")
plot_final_fitness_distribution(df_bragg_best_algs, df_bragg_baseline, "Bragg")
plot_final_fitness_distribution(df_ellipsometry_best_algs,
                                df_ellipsometry_baseline, "Ellipsometry")
plot_final_fitness_distribution(df_photovoltaic_best_algs,
                                df_photovoltaic_baseline, "Photovoltaics")
plot_final_fitness_distribution(df_big_photovoltaic_best_algs,
                                df_bigphotovoltaic_baseline, "Big Photovoltaics")
plot_final_fitness_distribution(df_huge_photovoltaic_best_algs,
                                df_hugephotovoltaic_baseline, "Huge Photovoltaics")

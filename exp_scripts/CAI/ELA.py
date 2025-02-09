import os
import sys
import ioh
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(".")
sys.path.append("./benchmarks/tuto_global_optimization_photonics/")
from photonics_benchmark import *
from pflacco.sampling import create_initial_sample
from pflacco.classical_ela_features import calculate_dispersion
from pflacco.classical_ela_features import calculate_ela_distribution
from pflacco.classical_ela_features import calculate_ela_level
from pflacco.classical_ela_features import calculate_ela_meta
from pflacco.classical_ela_features import calculate_information_content
from pflacco.classical_ela_features import calculate_nbc
from pflacco.classical_ela_features import calculate_pca


problems = {
    "bragg": [10, 20],
    "ellipsometry": [2],
    "photovoltaic": [10, 20, 32]
}


def get_photonic_instances(name, dim):
    if name == "bragg":
        nb_layers = dim
        target_wl = 600.0
        mat_env = 1.0
        mat1 = 1.4
        mat2 = 1.8
        prob = brag_mirror(nb_layers, target_wl, mat_env, mat1, mat2)
        ioh.problem.wrap_real_problem(
            prob, name="brag_mirror",
            optimization_type=ioh.OptimizationType.MIN)
        problem = ioh.get_problem("brag_mirror", dimension=prob.n)
        problem.bounds.lb = prob.lb
        problem.bounds.ub = prob.ub
        return problem
    elif name == "ellipsometry":
        mat_env = 1.0
        mat_substrate = 'Gold'
        nb_layers = 1
        min_thick = 50
        max_thick = 150
        min_eps = 1.1
        max_eps = 3
        wavelengths = np.linspace(400, 800, 31)
        angle = 40*np.pi/180
        prob = ellipsometry(mat_env, mat_substrate, nb_layers, min_thick,
                            max_thick, min_eps, max_eps, wavelengths, angle)
        ioh.problem.wrap_real_problem(
            prob, name="ellipsometry",
            optimization_type=ioh.OptimizationType.MIN)
        problem = ioh.get_problem("ellipsometry", dimension=prob.n)
        problem.bounds.lb = prob.lb
        problem.bounds.ub = prob.ub
        return problem
    elif name == "photovoltaic":
        nb_layers = dim
        min_thick = 30
        max_thick = 250
        wl_min = 375
        wl_max = 750
        prob = sophisticated_antireflection_design(
            nb_layers, min_thick, max_thick, wl_min, wl_max)
        ioh.problem.wrap_real_problem(
            prob, name="sophisticated_antireflection_design",
            optimization_type=ioh.OptimizationType.MIN)
        problem = ioh.get_problem("sophisticated_antireflection_design",
                                  dimension=prob.n)
        problem.bounds.lb = prob.lb
        problem.bounds.ub = prob.ub
        return problem
    else:
        return None


def create_samplingX():
    for problem_name in problems.keys():
        for dim in problems[problem_name]:
            samplingX = []
            problem = get_photonic_instances(problem_name, dim)
            for _ in range(100):
                X = create_initial_sample(dim, n=dim*50, sample_type='lhs',
                                          lower_bound=problem.bounds.lb,
                                          upper_bound=problem.bounds.ub
                                          ).values.tolist()
                samplingX.append(X)
            samplingX = np.array(samplingX)
            np.save(f"exp_data/CAI/ELA/samplingX/{problem_name}_{dim}.npy",
                    samplingX)


def create_samplingY():
    for problem_name in problems.keys():
        for dim in problems[problem_name]:
            problem = get_photonic_instances(problem_name, dim)
            samplingX = np.load(
                f"exp_data/CAI/ELA/samplingX/{problem_name}_{dim}.npy")
            samplingY = []
            for i in range(100):
                print(i)
                X = samplingX[i]
                Y = problem(np.array(X))
                samplingY.append(Y)
            samplingY = np.array(samplingY)
            print(samplingY.shape)
            np.savetxt(f"exp_data/CAI/ELA/samplingY/{problem_name}_{dim}.txt",
                       samplingY)


def ela_calculation(X, y):
    keys = []
    values = []
    disp = calculate_dispersion(X, y)
    keys += list(disp.keys())[:-1]
    values += list(disp.values())[:-1]
    ela_distr = calculate_ela_distribution(X, y)
    keys += list(ela_distr.keys())[:-1]
    values += list(ela_distr.values())[:-1]
    ela_level = calculate_ela_level(X, y)
    keys += list(ela_level.keys())[:-1]
    values += list(ela_level.values())[:-1]
    ela_meta = calculate_ela_meta(X, y)
    keys += list(ela_meta.keys())[:-1]
    values += list(ela_meta.values())[:-1]
    ic = calculate_information_content(X, y)
    keys += list(ic.keys())[:-1]
    values += list(ic.values())[:-1]
    nbc = calculate_nbc(X, y)
    keys += list(nbc.keys())[:-1]
    values += list(nbc.values())[:-1]
    pca = calculate_pca(X, y)
    keys += list(pca.keys())[:-1]
    values += list(pca.values())[:-1]
    return keys, values


def create_ELA_table():
    prefix_name = ["problem_name", "dim"]
    for problem_name in problems.keys():
        for dim in problems[problem_name]:
            if os.path.exists(f"exp_data/CAI/ELA/ELA_{problem_name}_{dim}.csv"):
                continue
            records = []
            prefix = [problem_name, dim]
            samplingX = np.load(
                f"exp_data/CAI/ELA/samplingX/{problem_name}_{dim}.npy")
            samplingY = np.loadtxt(
                f"exp_data/CAI/ELA/samplingY/{problem_name}_{dim}.txt")
            for i in range(100):
                print(i)
                X = samplingX[i]
                y = samplingY[i]
                keys, values = ela_calculation(X, y)
                records += [prefix + values]
            column_name = prefix_name + keys
            dataset_df = pd.DataFrame(records, columns=column_name)
            dataset_df.to_csv(f"exp_data/CAI/ELA/ELA_{problem_name}_{dim}.csv",
                              index=False)


def draw_distribution():
    rows, cols = 8, 7
    fig, axes = plt.subplots(rows, cols, figsize=(30, 20))
    prefix_name = ["problem_name", "dim"]
    dfs = []
    for problem_name in problems.keys():
        for dim in problems[problem_name]:
            if not os.path.exists(f"exp_data/CAI/ELA/ELA_{problem_name}_{dim}.csv"):
                continue
            dataset_df = pd.read_csv(
                f"exp_data/CAI/ELA/ELA_{problem_name}_{dim}.csv")
            dfs += [dataset_df]
    dataset_df = pd.concat(dfs, axis=0)
    dataset_df["label"] = dataset_df["problem_name"] + " " + dataset_df["dim"].astype(str) + "D"
    for i, ax in enumerate(axes.flat):
        if i >= len(dataset_df.columns) - 2:
            break
        sns.violinplot(x="label", y=dataset_df.columns[i+2], data=dataset_df, hue="label", ax=ax)
        ax.set_title(dataset_df.columns[i+2])
        ax.set_xticklabels([])
    plt.tight_layout()
    plt.savefig("exp_data/CAI/ELA/ELA_distribution.png")



if __name__ == "__main__":
    # create_samplingX()
    # create_samplingY()
    # create_ELA_table()
    draw_distribution()

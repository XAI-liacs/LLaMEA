import sys
import ioh
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(".")
sys.path.append("./benchmarks/tuto_global_optimization_photonics/")
from photonics_benchmark import *

def get_photonic_instances():
    problems = []
    # ------- define "mini-bragg" optimization problem
    nb_layers = 2     # number of layers of full stack
    target_wl = 600.0  # nm
    mat_env = 1.0      # materials: ref. index
    mat1 = 1.4
    mat2 = 1.8
    prob = brag_mirror(nb_layers, target_wl, mat_env, mat1, mat2)
    ioh.problem.wrap_real_problem(prob, name="brag_mirror",
                                  optimization_type=ioh.OptimizationType.MIN)
    problem = ioh.get_problem("brag_mirror", dimension=prob.n)
    problem.bounds.lb = prob.lb
    problem.bounds.ub = prob.ub
    problems.append(problem)
    # ------- define "ellipsometry" optimization problem
    mat_env = 1.0
    mat_substrate = 'Gold'
    nb_layers = 1
    min_thick = 50     # nm
    max_thick = 150
    min_eps = 1.1      # permittivity
    max_eps = 3
    wavelengths = np.linspace(400, 800, 31)  # nm
    angle = 40*np.pi/180  # rad
    prob = ellipsometry(mat_env, mat_substrate, nb_layers, min_thick, max_thick,
                        min_eps, max_eps, wavelengths, angle)
    ioh.problem.wrap_real_problem(prob, name="ellipsometry",
                                  optimization_type=ioh.OptimizationType.MIN,)
    problem = ioh.get_problem("ellipsometry", dimension=prob.n)
    problem.bounds.lb = prob.lb
    problem.bounds.ub = prob.ub
    problems.append(problem)
    # ------- define "sophisticated antireflection" optimization problem
    nb_layers = 2
    min_thick = 30
    max_thick = 250
    wl_min = 375
    wl_max = 750
    prob = sophisticated_antireflection_design(nb_layers, min_thick, max_thick,
                                               wl_min, wl_max)
    ioh.problem.wrap_real_problem(prob, name="sophisticated_antireflection_design",
                                  optimization_type=ioh.OptimizationType.MIN)
    problem = ioh.get_problem("sophisticated_antireflection_design",
                              dimension=prob.n)
    problem.bounds.lb = prob.lb
    problem.bounds.ub = prob.ub
    problems.append(problem)
    return problems


problems = get_photonic_instances()
# draw the landscape of problems as heatmap and save as png figures, problems are in 2D
for k in range(len(problems)):
    problem = problems[k]
    x = np.linspace(problem.bounds.lb[0], problem.bounds.ub[0], 100)
    y = np.linspace(problem.bounds.lb[1], problem.bounds.ub[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = problem(np.array([X[i, j], Y[i, j]]))
    np.savetxt(f"exp_data/CAI/real/{k}.txt", Z)
    # draw heatmap of Z in 100x100 resolution
    sns.heatmap(Z, cmap='viridis', cbar=False, square=True)
    # hide axis and ticks
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"results/CAI/{k}.png")
    plt.cla()
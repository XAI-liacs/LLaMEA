
import sys
import ioh
import numpy as np
import best_algs.best1 as best1
import best_algs.best2 as best2
import best_algs.best3 as best3
sys.path.append(".")
sys.path.append("./benchmarks/tuto_global_optimization_photonics/")
from photonics_benchmark import *

def get_photonic_instances():
    problems = []
    # ------- define "mini-bragg" optimization problem
    nb_layers = 20     # number of layers of full stack
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
    wavelengths = np.linspace(400, 800, 100)  # nm
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
    nb_layers = 10
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

def run_experiment(problem, n_runs=5):
    for run in range(n_runs):
        algorithm = best1.EnhancedHybridPSO_DE(budget, dim)
        # Run the algorithm on the problem
        algorithm(problem)
        # print the best found for this run
        print(f"run: {run+1} - best found:{problem.state.current_best.y: .3f}")
        # Reset the problem
        problem.reset()

problems = get_photonic_instances()
f = problems[0]
dim = f.meta_data.n_variables
budget = 1000 * dim
print(f.state)
print(f.meta_data)
print(f.bounds)
logger_params = dict(
    triggers=[ioh.logger.trigger.ALWAYS, ioh.logger.trigger.ON_VIOLATION],
    additional_properties=[],
    root="exp_data/CAI/",
    folder_name="bragg",
    algorithm_name='best_alg1',
    store_positions=True,
)
logger = ioh.logger.Analyzer(**logger_params)
logger.reset()
f.attach_logger(logger)
run_experiment(f, n_runs=15)
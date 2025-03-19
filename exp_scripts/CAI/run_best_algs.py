import os
import sys
import ioh
import jsonlines
import numpy as np
import best_algs.best1 as best1
import best_algs.best2 as best2
import best_algs.best3 as best3
sys.path.append(".")
sys.path.append("./benchmarks/tuto_global_optimization_photonics/")
from photonics_benchmark import *
from misc import OverBudgetException

def get_photonic_instances(instance_id):
    if instance_id == 0:
        # ------- define "mini-bragg" optimization problem
        nb_layers = 10     # number of layers of full stack
        target_wl = 600.0  # nm
        mat_env = 1.0      # materials: ref. index
        mat1 = 1.4
        mat2 = 1.8
        prob = brag_mirror(nb_layers, target_wl, mat_env, mat1, mat2)
        ioh.problem.wrap_real_problem(prob, name="mini_bragg",
                                      optimization_type=ioh.OptimizationType.MIN)
        problem = ioh.get_problem("mini_bragg", dimension=prob.n)
        problem.bounds.lb = prob.lb
        problem.bounds.ub = prob.ub
        return problem
    elif instance_id == 1:
        # ------- define "mini-bragg" optimization problem
        nb_layers = 20     # number of layers of full stack
        target_wl = 600.0  # nm
        mat_env = 1.0      # materials: ref. index
        mat1 = 1.4
        mat2 = 1.8
        prob = brag_mirror(nb_layers, target_wl, mat_env, mat1, mat2)
        ioh.problem.wrap_real_problem(prob, name="bragg",
                                      optimization_type=ioh.OptimizationType.MIN)
        problem = ioh.get_problem("bragg", dimension=prob.n)
        problem.bounds.lb = prob.lb
        problem.bounds.ub = prob.ub
        return problem
    elif instance_id == 2:
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
        return problem
    elif instance_id == 3:
        # ------- define "sophisticated antireflection" optimization problem
        nb_layers = 10
        min_thick = 30
        max_thick = 250
        wl_min = 375
        wl_max = 750
        prob = sophisticated_antireflection_design(nb_layers, min_thick, max_thick,
                                                wl_min, wl_max)
        ioh.problem.wrap_real_problem(prob, name="photovoltaic", 
                                      optimization_type=ioh.OptimizationType.MIN)
        problem = ioh.get_problem("photovoltaic", dimension=prob.n)
        problem.bounds.lb = prob.lb
        problem.bounds.ub = prob.ub
        return problem
    elif instance_id == 4:
        # ------- define "sophisticated antireflection" optimization problem
        nb_layers = 20
        min_thick = 30
        max_thick = 250
        wl_min = 375
        wl_max = 750
        prob = sophisticated_antireflection_design(nb_layers, min_thick, max_thick,
                                                wl_min, wl_max)
        ioh.problem.wrap_real_problem(prob, name="big_photovoltaic",
                                      optimization_type=ioh.OptimizationType.MIN)
        problem = ioh.get_problem("big_photovoltaic", dimension=prob.n)
        problem.bounds.lb = prob.lb
        problem.bounds.ub = prob.ub
        return problem
    elif instance_id == 5:
        # ------- define "sophisticated antireflection" optimization problem
        nb_layers = 32
        min_thick = 30
        max_thick = 250
        wl_min = 375
        wl_max = 750
        prob = sophisticated_antireflection_design(nb_layers, min_thick, max_thick,
                                                wl_min, wl_max)
        ioh.problem.wrap_real_problem(prob, name="huge_photovoltaic",
                                      optimization_type=ioh.OptimizationType.MIN)
        problem = ioh.get_problem("huge_photovoltaic", dimension=prob.n)
        problem.bounds.lb = prob.lb
        problem.bounds.ub = prob.ub
        return problem


def find_best_algs():
    n_algs = 10
    problems = [
        "bragg",
        "ellipsometry",
        "photovoltaic"
    ]
    best_alg_fitness = [[0. for _ in range(n_algs)] for _ in range(len(problems))]
    best_alg_solution = [[None for _ in range(n_algs)] for _ in range(len(problems))]
    best_alg_name = [[None for _ in range(n_algs)] for _ in range(len(problems))]
    root_path = "exp_data/CAI/populations/"
    folders = os.listdir(root_path)
    for folder in folders:
        folder_problem = ""
        if not folder.startswith("exp"):
            continue
        for problem in problems:
            if problem in folder:
                folder_problem = problem
                break
        if folder_problem == "":
            print(f"Skipping {folder}")
        folder_problem_index = problems.index(folder_problem)
        log_file = f"{root_path}{folder}/log.jsonl"
        with jsonlines.open(log_file) as reader:
            for obj in reader:
                fitness = obj["fitness"]
                solution = obj["solution"]
                name = obj["name"]
                for i in range(n_algs):
                    if fitness > best_alg_fitness[folder_problem_index][i]:
                        for j in range(n_algs-1, i, -1):
                            best_alg_fitness[folder_problem_index][j] = \
                                best_alg_fitness[folder_problem_index][j-1]
                            best_alg_solution[folder_problem_index][j] = \
                                best_alg_solution[folder_problem_index][j-1]
                            best_alg_name[folder_problem_index][j] = \
                                best_alg_name[folder_problem_index][j-1]
                        best_alg_fitness[folder_problem_index][i] = fitness
                        best_alg_solution[folder_problem_index][i] = solution
                        best_alg_name[folder_problem_index][i] = name
                        break
    return best_alg_fitness, best_alg_solution, best_alg_name


def benchmark(best_alg_name, best_alg_solution, instance_id, n_runs=15):
    alg_ids = [
        [3, 4, 6],
        [0, 1, 2],
        [2]
    ]
    best_alg_group_id = -1
    if instance_id in [0, 1]:
        best_alg_group_id = 0
    elif instance_id in [2]:
        best_alg_group_id = 1
    elif instance_id in [3, 4, 5]:
        best_alg_group_id = 2
    if best_alg_group_id == -1:
        print("Instance ID not found")
        return None
    for i in alg_ids[best_alg_group_id]:
        code = best_alg_solution[best_alg_group_id][i]
        algorithm_name = best_alg_name[best_alg_group_id][i]
        exec(code, globals())
        problem = get_photonic_instances(instance_id)
        logger_params = dict(
            triggers=[ioh.logger.trigger.ALWAYS, ioh.logger.trigger.ON_VIOLATION],
            additional_properties=[],
            root="exp_data/CAI/benchmark/",
            folder_name=f"{problem.meta_data.name}_best_alg{i}_{algorithm_name}",
            algorithm_name=f"best_alg{i}_{algorithm_name}",
            store_positions=True,
        )
        logger = ioh.logger.Analyzer(**logger_params)
        logger.reset()
        problem.attach_logger(logger)
        dim = problem.meta_data.n_variables
        if instance_id in [0, 1]:
            budget = 1000 * dim
        elif instance_id in [2, 3, 4, 5]:
            budget = 500 * dim
        for run in range(n_runs):
            try:
                algorithm = globals()[algorithm_name](budget=budget, dim=dim)
                algorithm(problem)
                print(f"run: {run+1} - best found:{problem.state.current_best.y: .3f}")
                problem.reset()
            except OverBudgetException:
                pass


if __name__ == "__main__":
    best_alg_fitness, best_alg_solution, best_alg_name = find_best_algs()
    print(best_alg_name)
    for instance_id in range(5, 6):
        benchmark(best_alg_name, best_alg_solution, instance_id)

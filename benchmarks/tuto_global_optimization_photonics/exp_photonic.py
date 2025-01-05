import os
import sys
import ioh
import numpy as np
sys.path.append(".")
from photonics_benchmark import *
from llamea import LLaMEA
from misc import aoc_logger, correct_aoc, OverBudgetException
from pyGDM2 import materials
from pyGDM2 import fields
from pyGDM2 import core
from pyGDM2 import propagators
from pyGDM2 import structures


def get_photonic_instances():
    problems = []
    # ------- define "mini-bragg" optimization problem
    nb_layers = 10     # number of layers of full stack
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
    # # ------- define "2D grating" optimization problem
    # nb_layers = 5
    # min_w = 0
    # max_w = 600
    # min_thick = 0
    # max_thick = 200
    # min_p = 0
    # max_p = 600
    # prob = grating2D(nb_layers, min_w, max_w,
    #                  min_thick, max_thick, min_p, max_p)
    # ioh.problem.wrap_real_problem(prob, name="grating2D",
    #                               optimization_type=ioh.OptimizationType.MIN)
    # problem = ioh.get_problem("grating2D", dimension=prob.n)
    # problem.bounds.lb = prob.lb
    # problem.bounds.ub = prob.ub
    # problems.append(problem)
    # # ------- define "plasmonic nanostructure" optimization problem
    # N_elements = 40
    # min_pos = -12
    # max_pos = 12
    # method = 'lu'
    # step = 20
    # material = materials.gold()
    # geometry = structures.rect_wire(step, L=2, W=2, H=2)
    # geometry = structures.center_struct(geometry)
    # struct = structures.struct(step, geometry, material)
    # ## environment: air
    # n1 = 1.0
    # dyads = propagators.DyadsQuasistatic123(n1=n1)
    # ## illumination: local quantum emitter (dipole source)
    # field_generator = fields.dipole_electric
    # kwargs = dict(x0=0, y0=0, z0=step, mx=0, my=1, mz=0, R_farfield_approx=5000)
    # wavelengths = [800.]
    # efield = fields.efield(field_generator, wavelengths=wavelengths, kwargs=kwargs)
    # ## simulation object of single element
    # element_sim = core.simulation(struct, efield, dyads)
    # # XY_coords_blocks = np.random.randint(-10, 10, 20) * 5
    # # full_sim = setup_structure(XY_coords_blocks, element_sim)
    # # cost = cost_direct_emission(XY_coords_blocks, element_sim, method='lu')
    # prob = plasmonic_nanostructure(element_sim, method, verbose=False)
    # ioh.problem.wrap_real_problem(prob, name="plasmonic_nanostructure",
    #                               optimization_type=ioh.OptimizationType.MIN,
    #                               lb=-10, ub=10)
    # problem = ioh.get_problem("plasmonic_nanostructure", dimension=prob.n)
    # problems.append(problem)
    return problems


def evaluatePhotonic(solution, details=False):
    auc_mean = 0
    auc_std = 0
    detailed_aucs = [0, 0, 0, 0, 0]
    code = solution.solution
    algorithm_name = solution.name
    exec(code, globals())
    aucs = []
    detail_aucs = []
    algorithm = None
    problems = get_photonic_instances()
    for i in range(len(problems)):
        problem = problems[i]
        dim = problem.meta_data.n_variables
        budget = 100 * dim
        l2 = aoc_logger(budget, upper=1e2, triggers=[
                        ioh.logger.trigger.ALWAYS])
        problem.attach_logger(l2)
        for rep in range(1):
            np.random.seed(rep)
            try:
                algorithm = globals()[algorithm_name](budget=budget, dim=dim)
                algorithm(problem)
            except OverBudgetException:
                pass
            auc = correct_aoc(problem, l2, budget)
            aucs.append(auc)
            detail_aucs.append(auc)
            l2.reset(problem)
            problem.reset()
        detailed_aucs[i] = np.mean(detail_aucs)
        detail_aucs = []
    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)
    i = 0
    while os.path.exists(f"currentexp/aucs-{algorithm_name}-{i}.npy"):
        i += 1
    np.save(f"currentexp/aucs-{algorithm_name}-{i}.npy", aucs)

    feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with standard deviation {auc_std:0.2f}."
    # if details:
    #     feedback = (
    #         f"{feedback}\nThe mean AOCC score of the algorithm {algorithm_name} on Optimization of a Bragg mirror was {detailed_aucs[0]:.02f}, "
    #         f"on Solving of an ellipsometry inverse problem {detailed_aucs[1]:.02f}, "
    #         f"on Design of a sophisticated antireflection coating to optimize solar absorption in a photovoltaic solar cell {detailed_aucs[2]:.02f}, "
    #         f"on Optimization of a 2D grating for minimum specular reflectance at a given wavelength {detailed_aucs[3]:.02f}, "
    #         f"and on Design of a plasmonic nanostructure for directional emission from a local emitter {detailed_aucs[4]:.02f}"
    #     )

    print(algorithm_name, algorithm, auc_mean, auc_std)
    solution.add_metadata("aucs", aucs)
    solution.set_scores(auc_mean, feedback)

    return solution


# api_key = os.getenv("OPENAI_API_KEY")
# ai_model = "gpt-4o"  # gpt-4-turbo or gpt-3.5-turbo gpt-4o llama3:70b
# experiment_name = "Photonic03"

# task_prompt = """
# The optimization algorithm should handle a wide range of tasks, which is evaluated on real-world applications, Global optimization of photonic structures. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
# The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between func.bounds.lb (lower bound) and func.bounds.ub (upper bound). The dimensionality can be varied.
# Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description with the main idea.
# """

# for experiment_i in range(1):
#     # A 1+1 strategy
#     es = LLaMEA(evaluatePhotonic, n_parents=1, n_offspring=1, api_key=api_key,
#                 task_prompt=task_prompt, experiment_name=experiment_name,
#                 model=ai_model, elitism=True, HPO=False, budget=100)
#     print(es.run())


import time

problems = get_photonic_instances()
for prob in problems:
    start_time = time.time()
    dim = prob.meta_data.n_variables
    xs = np.random.uniform(prob.bounds.lb, prob.bounds.ub, (100*dim, dim))
    y_min = np.inf
    y_max = -np.inf
    for x in xs:
        y = prob(x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
    print(f"Min: {y_min}, Max: {y_max}")
    end_time = time.time()
    print(f"Time: {end_time - start_time}")
#     for i in range(100*prob.meta_data.n_variables):
#         x = np.random.uniform(prob.bounds.lb, prob.bounds.ub)
#         y = prob(x)
#         if i % 100 == 0:
#             print(f"{i}: {prob(x)}")
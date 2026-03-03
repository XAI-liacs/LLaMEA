import os
import sys
import ioh
import numpy as np

sys.path.append(".")
from photonics_benchmark import *
from llamea import LLaMEA
from misc import aoc_logger, correct_aoc, OverBudgetException

from llamea import Gemini_LLM, OpenAI_LLM


def get_photonic_instances():
    problems = []
    if problem_name == "bragg":
        # ------- define "mini-bragg" optimization problem
        nb_layers = 10  # number of layers of full stack
        target_wl = 600.0  # nm
        mat_env = 1.0  # materials: ref. index
        mat1 = 1.4
        mat2 = 1.8
        prob = brag_mirror(nb_layers, target_wl, mat_env, mat1, mat2)
        ioh.problem.wrap_real_problem(
            prob, name="brag_mirror", optimization_type=ioh.OptimizationType.MIN
        )
        problem = ioh.get_problem("brag_mirror", dimension=prob.n)
        problem.bounds.lb = prob.lb
        problem.bounds.ub = prob.ub
        problems.append(problem)
    elif problem_name == "ellipsometry":
        # ------- define "ellipsometry" optimization problem
        mat_env = 1.0
        mat_substrate = "Gold"
        nb_layers = 1
        min_thick = 50  # nm
        max_thick = 150
        min_eps = 1.1  # permittivity
        max_eps = 3
        wavelengths = np.linspace(400, 800, 100)  # nm
        angle = 40 * np.pi / 180  # rad
        prob = ellipsometry(
            mat_env,
            mat_substrate,
            nb_layers,
            min_thick,
            max_thick,
            min_eps,
            max_eps,
            wavelengths,
            angle,
        )
        ioh.problem.wrap_real_problem(
            prob,
            name="ellipsometry",
            optimization_type=ioh.OptimizationType.MIN,
        )
        problem = ioh.get_problem("ellipsometry", dimension=prob.n)
        problem.bounds.lb = prob.lb
        problem.bounds.ub = prob.ub
        problems.append(problem)
    elif problem_name == "photovoltaic":
        # ------- define "sophisticated antireflection" optimization problem
        nb_layers = 10
        min_thick = 30
        max_thick = 250
        wl_min = 375
        wl_max = 750
        prob = sophisticated_antireflection_design(
            nb_layers, min_thick, max_thick, wl_min, wl_max
        )
        ioh.problem.wrap_real_problem(
            prob,
            name="sophisticated_antireflection_design",
            optimization_type=ioh.OptimizationType.MIN,
        )
        problem = ioh.get_problem(
            "sophisticated_antireflection_design", dimension=prob.n
        )
        problem.bounds.lb = prob.lb
        problem.bounds.ub = prob.ub
        problems.append(problem)
    # # ------- define "2D grating" optimization problem
    # nb_layers = 2
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
    if problem_name == "bragg":
        auc_lower = 0.1648
        auc_upper = 1.0
    elif problem_name == "ellipsometry":
        auc_lower = 1e-8
        auc_upper = 40.0
    elif problem_name == "photovoltaic":
        auc_lower = 0.1
        auc_upper = 1.0
    auc_mean = 0
    auc_std = 0
    code = solution.code
    algorithm_name = solution.name
    exec(code, globals())
    aucs = []
    detail_aucs = []
    algorithm = None
    problems = get_photonic_instances()

    problem = problems[0]
    dim = problem.meta_data.n_variables
    budget = 500 * dim
    if problem_name == "photovoltaic":
        budget = 100 * dim
    l2 = aoc_logger(
        budget, lower=auc_lower, upper=auc_upper, triggers=[ioh.logger.trigger.ALWAYS]
    )
    problem.attach_logger(l2)
    final_y = []
    for rep in range(3):
        np.random.seed(rep)
        try:
            algorithm = globals()[algorithm_name](budget=budget, dim=dim)
            algorithm(problem)
        except OverBudgetException:
            pass
        auc = correct_aoc(problem, l2, budget)
        aucs.append(auc)
        detail_aucs.append(auc)
        final_y += [problem.state.current_best.y]
        l2.reset(problem)
        problem.reset()
    detail_aucs = []

    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)
    i = 0
    while os.path.exists(f"currentexp/aucs-{algorithm_name}-{i}.npy"):
        i += 1
    np.save(f"currentexp/aucs-{algorithm_name}-{i}.npy", aucs)

    feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.3f} with standard deviation {auc_std:0.3f}. And the mean value of best solutions found was {np.mean(final_y):0.3f} (0. is the best) with standard deviation {np.std(final_y):0.3f}."

    print(algorithm_name, algorithm, auc_mean, auc_std)
    solution.add_metadata("aucs", aucs)
    solution.add_metadata("final_y", final_y)
    solution.set_scores(auc_mean, feedback)

    return solution


ai_model = sys.argv[1]  # gpt-4-turbo or gpt-3.5-turbo gpt-4o llama3:70b
problem_id = int(sys.argv[2])
with_description = True if sys.argv[3] == "1" else False
with_insight = True if sys.argv[4] == "1" else False
parent_size = int(sys.argv[5])
offspring_size = int(sys.argv[6])
es_flag = False if sys.argv[7] == "0" else True
if "gemini" in ai_model:
    api_key = os.getenv("GEMINI_API_KEY")
    llm = Gemini_LLM(api_key, ai_model)
else:
    api_key = os.getenv("OPENAI_API_KEY")
    llm = OpenAI_LLM(api_key, ai_model)

problem_types = ["bragg", "ellipsometry", "photovoltaic"]
problem_name = problem_types[problem_id]
n1 = "_with_description" if with_description else ""
n2 = "_insight" if with_insight else ""
n3 = " + " if es_flag else ", "
experiment_name = f"{problem_name}{n1}{n2}_({parent_size}{n3}{offspring_size})"


descriptions = {
    "bragg": "The Bragg mirror optimization aims to maximize reflectivity at a wavelength of 600 nm using a multilayer structure with alternating refractive indices (1.4 and 1.8). The structure's thicknesses are varied to find the configuration with the highest reflectivity. The problem involves two cases: one with 10 layers (minibragg) and another with 20 layers (bragg), with the latter representing a more complex inverse design problem. The known optimal solution is a periodic Bragg mirror, which achieves the best reflectivity by leveraging constructive interference. This case exemplifies challenges such as multiple local minima in the optimization landscape. ",
    "ellipsometry": "The ellipsometry problem involves retrieving the material and thickness of a reference layer by matching its reflectance properties using a known spectral response. The optimization minimizes the difference between the calculated and measured ellipsometric parameters for wavelengths between 400 and 800 nm and a fixed incidence angle of 40°. The parameters to be optimized include the thickness (30 to 250 nm) and refractive index (1.1 to 3) of the test layer. This relatively straightforward problem models a practical scenario where photonics researchers fine-tune a small number of parameters to achieve a desired spectral fit. ",
    "photovoltaic": "The photovoltaics problem optimizes the design of an antireflective multilayer coating to maximize the absorption in the active silicon layer of a solar cell. The goal is to achieve maximum short-circuit current in the 375 to 750 nm wavelength range. The structure consists of alternating materials with permittivities of 2 and 3, built upon a 30,000 nm thick silicon substrate. Three subcases with increasing complexity are explored, involving 10 layers (photovoltaics), 20 layers (bigphotovoltaics), and 32 layers (hugephotovoltaics). The optimization challenges include balancing high absorption with a low reflectance while addressing the inherent noise and irregularities in the solar spectrum. ",
}
algorithmic_insights = {
    "bragg": "For this problem, the optimization landscape contains multiple local minima due to the wave nature of the problem. And periodic solutions are known to provide near-optimal results, suggesting the importance of leveraging constructive interference principles. Here are some suggestions for designing algorithms: 1. Use global optimization algorithms like Differential Evolution (DE) or Genetic Algorithms (GA) to explore the parameter space broadly. 2. Symmetric initialization strategies (e.g., Quasi-Oppositional DE) can improve exploration by evenly sampling the search space. 3. Algorithms should preserve modular characteristics in solutions, as multilayer designs often benefit from distinct functional blocks. 4. Combine global methods with local optimization (e.g., BFGS) to fine-tune solutions near promising regions. 5. Encourage periodicity in solutions via tailored cost functions or constraints. ",
    "ellipsometry": "This problem has small parameter space with fewer variables (thickness and refractive index), and the cost function is smooth and relatively free of noise, making it amenable to local optimization methods. Here are suggestions for designing algorithms: 1. Use local optimization algorithms like BFGS or Nelder-Mead, as they perform well in low-dimensional, smooth landscapes. 2. Uniform sampling across the parameter space ensures sufficient coverage for initial guesses. 3. Utilize fast convergence algorithms that can quickly exploit the smooth cost function landscape. 4. Iteratively adjust bounds and constraints to improve parameter estimates once initial solutions are obtained. ",
    "photovoltaic": "This problem is a challenging high-dimensional optimization problem with noisy cost functions due to the realistic solar spectrum, and it requires maximizing absorption while addressing trade-offs between reflectance and interference effects. Here are the suggestions for designing algorithms: 1. Combine global methods (e.g., DE, CMA-ES) for exploration with local optimization for refinement. 2. Use consistent benchmarking and convergence analysis to allocate computational resources effectively. 3. Encourage algorithms to detect and preserve modular structures (e.g., layers with specific roles like anti-reflective or coupling layers). 4. Gradually increase the number of layers during optimization to balance problem complexity and computational cost. 5. Integrate robustness metrics into the cost function to ensure the optimized design tolerates small perturbations in layer parameters. ",
}
description = descriptions[problem_name] if with_description == True else ""
algorithmic_insight = algorithmic_insights[problem_name] if with_insight == True else ""
task_prompt = f"""
The optimization algorithm should be able to find high-performing solutions to a wide range of tasks, which include evaluation on real-world applications such as, e.g., optimization of multilayered photonic structures. {description}{algorithmic_insight}Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between func.bounds.lb (lower bound) and func.bounds.ub (upper bound). The dimensionality can be varied.
Give an excellent and novel heuristic algorithm to solve this task and include it's one-line description with the main idea of the algorithm.
"""

es = LLaMEA(
    evaluatePhotonic,
    llm=llm,
    n_parents=parent_size,
    n_offspring=offspring_size,
    task_prompt=task_prompt,
    experiment_name=experiment_name,
    elitism=es_flag,
    HPO=False,
    budget=100,
)
print(es.run())

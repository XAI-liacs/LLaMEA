import os
import re
import time

import numpy as np
from nevergrad import benchmark

import ioh
from benchmarks.minibrag.problem import cost_minibragg, objective_f, upper_lower_bound
from ioh import get_problem, logger, wrap_problem
from llamea import LLaMEA
from misc import OverBudgetException, aoc_logger, correct_aoc

from llamea import Gemini_LLM

api_key = os.getenv("GEMINI_API_KEY")
ai_model = "gemini-2.0-flash"  # gpt-4-turbo or gpt-3.5-turbo gpt-4o llama3:70b gpt-4o-2024-05-13, gemini-1.5-flash gpt-4-turbo-2024-04-09
experiment_name = "gemini-minibrag"
llm = Gemini_LLM(api_key, ai_model)

log_folder = ""


def evaluate(solution, explogger=None):
    # first we wait a bit to not hit the Gemini threshold
    time.sleep(10)

    code = solution.code
    algorithm_name = solution.name
    exec(code, globals())

    lb, ub = upper_lower_bound()

    dim = 10
    budget = 50000
    error = ""
    algorithm = None

    wrap_problem(
        objective_f,
        "cost_minibragg",
        ioh.ProblemClass.REAL,
        dimension=dim,
        instance=0,
        lb=lb,
        ub=ub,
    )

    # perform a small run to check for any code errors
    l2_temp = aoc_logger(
        10, upper=1.0, lower=0.15, scale_log=True, triggers=[logger.trigger.ALWAYS]
    )
    problem = get_problem("cost_minibragg", instance=0, dimension=dim)
    problem.attach_logger(l2_temp)

    try:
        algorithm = globals()[algorithm_name](budget=10, dim=dim)
        algorithm(problem)
    except OverBudgetException:
        pass

    # last but not least, perform the final validation
    # l1 = ioh.logger.Analyzer(
    #     root=f"{log_folder}/ioh",
    #     folder_name=f"LLaMEA",
    #     algorithm_name=algorithm_name,
    # )
    l2 = aoc_logger(
        budget, upper=1.0, lower=0.15, scale_log=True, triggers=[logger.trigger.ALWAYS]
    )
    # combined_logger = ioh.logger.Combine([l1,l2])

    problem = get_problem("cost_minibragg", instance=0, dimension=dim)
    problem.attach_logger(l2)

    try:
        algorithm = globals()[algorithm_name](budget=budget, dim=dim)
        f_opt, x_opt = algorithm(problem)
    except OverBudgetException:
        pass
    auc = correct_aoc(problem, l2, budget)

    feedback = f"The algorithm {algorithm_name} got an Area over the convergence curve (AOCC, 1.0 is the best) score of {auc:0.2f}."
    # print(algorithm_name, algorithm, auc)
    # print(f_opt, x_opt)

    solution.set_scores(auc, feedback)
    return solution


role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."
task_prompt = """
The optimization algorithm should handle the optimization of a Bragg mirror. We will optimize the thicknesses of a layer-stack of two alterning dielectric materials, in total `dim` layers. We want to minimize one minus the reflectance of the layer-stack.
Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The budget is typically set at 10000+, so do not use any gaussian process surrogate models as that would cause too long optimization runs.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between 0.0 (lower bound) and 214.28 (upper bound). The dimensionality can be varied.
An example of such code (a simple random search), is as follows:
```python
import numpy as np

class RandomSearch:
    def __init__(self, budget=50000, dim=10):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        for i in range(self.budget):
            x = np.random.uniform(0.0, 214.28)
            
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            
        return self.f_opt, self.x_opt
```

Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code: <code>
"""

feedback_prompts = [
    f"Either refine or redesign to improve the solution (and give it a distinct one-line description)."
]


# test run
if False:
    from scipy.optimize import differential_evolution

    lb, ub = upper_lower_bound()
    bounds = [(lb, ub)] * 10

    dim = 10
    wrap_problem(
        objective_f,
        "cost_minibragg",
        ioh.ProblemClass.REAL,
        dimension=dim,
        instance=0,
        lb=lb,
        ub=ub,
    )
    budget = 50000

    print("Doing large run")

    # last but not least, perform the final validation
    l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
    problem = get_problem("cost_minibragg", instance=0, dimension=dim)
    problem.attach_logger(l2)

    try:
        result = differential_evolution(problem, bounds, maxiter=10000)
        print(result.x, result.fun)
    except OverBudgetException:
        pass
    auc = correct_aoc(problem, l2, budget)
    print(auc)

for experiment_i in [1]:
    es = LLaMEA(
        evaluate,
        llm=llm,
        n_parents=4,
        n_offspring=12,
        budget=250,
        eval_timeout=360,
        role_prompt=role_prompt,
        task_prompt=task_prompt,
        mutation_prompts=feedback_prompts,
        experiment_name=experiment_name,
        elitism=True,
        HPO=False,
    )
    log_folder = es.logger.dirname
    print(es.run())

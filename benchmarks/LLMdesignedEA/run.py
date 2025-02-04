import os
import numpy as np
from ioh import get_problem, wrap_problem, logger
import ioh
import re
from misc import (
    aoc_logger,
    correct_aoc,
    OverBudgetException,
    budget_logger,
    ThresholdReachedException,
)
from llamea import LLaMEA
from llamea.individual import Individual
from benchmarks.LLMdesignedEA.GNBG.GNBG_instances import load_problem
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from modcma import c_maes

import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def time_it() -> Iterator[None]:
    tic: float = time.perf_counter()
    try:
        yield
    finally:
        toc: float = time.perf_counter()
        print(f"Computation time = {1000*(toc - tic):.3f}ms")


# Execution code starts here
api_key = os.getenv("OPENAI_API_KEY")
ai_model = "gpt-4o-2024-08-06"  # "gpt-4o-2024-05-13"  # gpt-4-turbo or gpt-3.5-turbo gpt-4o llama3:70b gpt-4o-2024-05-13, gemini-1.5-flash gpt-4-turbo-2024-04-09
experiment_name = "competition"
if "gemini" in ai_model:
    api_key = os.environ["GEMINI_API_KEY"]

from itertools import product
from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
from smac import Scenario
from smac import AlgorithmConfigurationFacade

import concurrent.futures
from functools import partial


def calculate_objective(intance, dim):
    # print(dim, intance)
    gnbg = load_problem(intance, 0)
    # print(gnbg.OptimumPosition.flatten().tolist())
    return gnbg.OptimumPosition.flatten().tolist(), 0.0


def run_single_problem(fid, seed, algorithm_name, all_auc, all_f, all_x):
    gnbg = load_problem(fid, seed)
    dim = gnbg.Dimension
    budget = gnbg.MaxEvals

    wrap_problem(
        lambda x: gnbg.fitness(x) - gnbg.OptimumValue,
        f"gnbg{fid}",
        ioh.ProblemClass.REAL,
        lb=-100,
        ub=100,
        dimension=dim,
        instance=fid,
        calculate_objective=calculate_objective,
    )

    problem = get_problem(
        f"gnbg{fid}", dimension=dim, instance=fid, problem_class=ioh.ProblemClass.REAL
    )

    bl = aoc_logger(
        budget,
        upper=1e6,
        lower=1e-8,
        stop_on_threshold=True,
        triggers=[logger.trigger.ALWAYS],
    )
    problem.attach_logger(bl)

    try:
        algorithm = globals()[algorithm_name](budget=budget, dim=dim)
        algorithm(problem)
    except OverBudgetException:
        pass
    except ThresholdReachedException:
        pass
    except Exception as e:
        print(problem.state, budget, e)

    error = abs(gnbg.BestFoundResult - gnbg.OptimumValue)
    auc = correct_aoc(problem, bl, budget)
    all_auc.append(auc)
    all_f.append(gnbg.BestFoundResult)
    all_x.append(gnbg.BestFoundPosition)
    bl.reset(problem)
    problem.reset()


def evaluateWithHPO(solution, explogger=None):
    code = solution.solution
    algorithm_name = solution.name
    exec(code, globals())

    error = ""
    algorithm = None

    def get_fitness_all():
        np.random.seed(0)
        all_f = []
        all_x = []
        all_auc = []
        for seed in range(1):
            for fid in range(1, 25):
                gnbg = load_problem(fid, seed)
                dim = gnbg.Dimension
                budget = gnbg.MaxEvals

                wrap_problem(
                    lambda x: gnbg.fitness(x) - gnbg.OptimumValue,
                    f"gnbg{fid}",
                    ioh.ProblemClass.REAL,
                    lb=-100,
                    ub=100,
                    dimension=dim,
                    instance=fid,
                    calculate_objective=calculate_objective,
                )

                problem = get_problem(
                    f"gnbg{fid}",
                    dimension=dim,
                    instance=fid,
                    problem_class=ioh.ProblemClass.REAL,
                )

                bl = aoc_logger(
                    budget,
                    upper=1e6,
                    lower=1e-8,
                    stop_on_threshold=True,
                    triggers=[logger.trigger.ALWAYS],
                )
                problem.attach_logger(bl)

                try:
                    algorithm = globals()[algorithm_name](budget=budget, dim=dim)
                    algorithm(problem)
                except OverBudgetException:
                    pass
                except ThresholdReachedException:
                    pass
                except Exception as e:
                    print(problem.state, budget, e)

                error = abs(gnbg.BestFoundResult - gnbg.OptimumValue)
                auc = correct_aoc(problem, bl, budget)
                all_auc.append(auc)
                all_f.append(gnbg.BestFoundResult)
                all_x.append(gnbg.BestFoundPosition)
                bl.reset(problem)
                problem.reset()

        return all_auc, all_f, all_x

    # last but not least, perform the final validation

    all_auc, all_f, all_x = get_fitness_all()
    final_f = np.mean(all_auc)

    feedback = f"The algorithm {algorithm_name} got an average area over the convergence curve (1.0 is the best) of {final_f:0.2f}."
    print(algorithm_name, algorithm, final_f)

    solution.add_metadata("all_auc", all_auc)
    solution.add_metadata("all_f", all_f)
    solution.add_metadata("all_x", all_x)
    solution.set_scores(final_f, feedback)

    return solution


role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."
task_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the GNBG test suite of 24 functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function with optional additional arguments and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -100.0 (lower bound) and 100.0 (upper bound). The dimensionality will be varied.
An example of such code (a simple random search), is as follows:
```python
import numpy as np

class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        for i in range(self.budget):
            x = np.random.uniform(-100, 100)
            
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            
        return self.f_opt, self.x_opt
```

Give an excellent and novel metaheuristic algorithm to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code: <code>
"""

feedback_prompts = [
    f"Either refine or redesign to improve the solution (and give it a distinct one-line description)."
]


if False:
    # first test the code
    # now optimize the hyper-parameters
    with time_it():
        for fid in range(1, 25):
            gnbg = load_problem(fid, 0)
            dim = gnbg.Dimension
            print(dim)
            budget = gnbg.MaxEvals
            print(budget)

            wrap_problem(
                lambda x: gnbg.fitness(x) - gnbg.OptimumValue,
                f"gnbg{fid}",
                ioh.ProblemClass.REAL,
                lb=-100,
                ub=100,
                dimension=dim,
                instance=fid,
                calculate_objective=calculate_objective,
            )

            problem = get_problem(
                f"gnbg{fid}",
                dimension=dim,
                instance=fid,
                problem_class=ioh.ProblemClass.REAL,
            )
            print(problem)

            lb = -100 * np.ones(dim)
            ub = 100 * np.ones(dim)
            bounds = []
            for i in range(0, dim):
                bounds.append(tuple((lb[i], ub[i])))

            bl = aoc_logger(
                budget,
                upper=1e6,
                lower=1e-8,
                stop_on_threshold=True,
                triggers=[logger.trigger.ALWAYS],
            )
            # bl = budget_logger(budget, triggers=[logger.trigger.ALWAYS])
            problem.attach_logger(bl)

            # Instantate a modules object
            modules = c_maes.parameters.Modules()
            modules.restart_strategy = c_maes.options.RestartStrategy.IPOP
            # Create a settings object, here also optional parameters such as sigma0 can be specified
            settings = c_maes.parameters.Settings(dim, modules)
            # Create a parameters object
            parameters = c_maes.Parameters(settings)
            # Pass the parameters object to the ModularCMAES optimizer class
            cma = c_maes.ModularCMAES(parameters)

            try:
                cma.run(problem)
            except OverBudgetException:
                pass
            except ThresholdReachedException:
                pass
            except Exception as e:
                print(problem.state, budget, e)
            # After running the algorithm, the best fitness value is stored in gnbg.BestFoundResult.
            # The best found position is stored in gnbg.BestFoundPosition.
            # The function evaluation number where the algorithm reached the acceptance threshold is stored in gnbg.AcceptanceReachPoint. If the algorithm did not reach the acceptance threshold, it is set to infinity.
            # For visualizing the convergence behavior, the history of the objective values is stored in gnbg.FEhistory, however it needs to be processed as follows:
            auc = correct_aoc(problem, bl, budget)
            fitness = gnbg.BestFoundResult
            print(fitness, auc)

            OptimumValue = gnbg.OptimumValue

            convergence = []
            best_error = float("inf")
            for value in gnbg.FEhistory:
                error = abs(value - OptimumValue)
                if error < best_error:
                    best_error = error
                convergence.append(best_error)

            bl.reset(problem)
            problem.reset()

            # Plotting the convergence
            plt.plot(range(1, len(convergence) + 1), convergence)
            plt.xlabel("Function Evaluation Number (FE)")
            plt.ylabel("Error")
            plt.title("Convergence Plot")
            plt.yscale("log")  # Set y-axis to logarithmic scale
            plt.savefig(f"convergence_gnbg_{fid}.png")
            plt.clf()
else:
    for experiment_i in [1]:
        es = LLaMEA(
            evaluateWithHPO,
            budget=1000,
            n_offspring=20,
            n_parents=10,
            role_prompt=role_prompt,
            task_prompt=task_prompt,
            mutation_prompts=feedback_prompts,
            api_key=api_key,
            experiment_name=experiment_name,
            model=ai_model,
            elitism=True,
            HPO=False,
        )
        print(es.run())

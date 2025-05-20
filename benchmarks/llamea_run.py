import llamea


### Testing the Optimization Algorithm Wrapper in Kernel Tuner
import os
from kernel_tuner import util
#from kernel_tuner.strategies.wrapper import OptAlgWrapper
import os
import numpy as np
from ioh import get_problem, logger
import re
import json
from llamea import LLaMEA, Gemini_LLM
import time
import traceback
import math
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm # optional, for progress bars
import os
import random

from kernel_tuner import util
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.common import CostFunc
from kernel_tuner import tune_kernel_T1
from pathlib import Path

import ioh
from ioh import get_problem, logger, wrap_problem
from llamea import LLaMEA
from misc import OverBudgetException, aoc_logger, correct_aoc

application = "gemm"
gpu = "A100"

# TODO: run for all.
applications = ["gemm", "convolution", "dedispersion", "hotspot"]
gpus = ["A100", "A4000", "A6000", "MI250X", "W6600", "W7700", "W7800"]



class OverBudgetException(Exception):
    """The algorithm tried to do more evaluations than allowed."""

    pass

class problem_wrapper:
    def __init__(self, f, budget, scale_log=True):
        self.f = f
        self.budget = budget
        self.aoc = 0
        self.lower = 5
        self.upper = 1e2
        self.budget = budget
        self.eval_count = 0
        self.raw_y_best = self.upper
        self.transform = lambda x: np.log10(x) if scale_log else (lambda x: x)
    
    def __call__(self, x):
        if self.eval_count > self.budget:
            raise OverBudgetException("Budget exceeded")
        y = self.f(x)
        if y < self.raw_y_best:
            self.raw_y_best = y
        y_value = np.clip(self.raw_y_best, self.lower, self.upper)
        self.aoc += (self.transform(y_value) - self.transform(self.lower)) / (
            self.transform(self.upper) - self.transform(self.lower)
        )
        self.eval_count += 1
        return y

    def get_aoc(self):
        while self.eval_count < self.budget:
            y_value = np.clip(self.raw_y_best, self.lower, self.upper)
            self.aoc += (self.transform(y_value) - self.transform(self.lower)) / (
                self.transform(self.upper) - self.transform(self.lower)
            )
            self.eval_count += 1
        return 1 - (self.aoc / self.budget)
    

class OptAlgWrapper:
    """Wrapper class for user-defined optimization algorithms"""

    def __init__(self, optimizer, budget=100, scaling=False):
        self.optimizer = optimizer
        self.scaling = scaling
        self.budget = budget
        self.aoc = 0


    def tune(self, searchspace: Searchspace, runner, tuning_options):
        cost_func = CostFunc(searchspace, tuning_options, runner, scaling=self.scaling)

        # l2 = aoc_logger(
        #     self.budget, upper=1e2, lower=1e-1, scale_log=True, triggers=[logger.trigger.ALWAYS]
        # )
        #problem = get_problem(f"{application}-{gpu}", instance=0, problem_class=ioh.ProblemClass.INTEGER, dimension=len(tuning_options))
        #problem.attach_logger(l2)

        problem = problem_wrapper(cost_func, self.budget)

        self.tuning_options = tuning_options
        self.searchspace = searchspace

        if self.scaling:
            # Initialize costfunc for scaling
            cost_func.get_bounds_x0_eps()

        try:
            self.optimizer(problem, searchspace)
        except OverBudgetException:
            pass
        except util.StopCriterionReached as e:
            if tuning_options.verbose:
                print(e)

        self.aoc = problem.get_aoc() #correct_aoc(problem, l2, self.budget)
        return problem.f.results




# Execution code starts here
api_key = os.getenv("GEMINI_API_KEY")
ai_model = "gemini-2.0-flash"  # -thinking-exp-01-21 gpt-4-turbo or gpt-3.5-turbo gpt-4o llama3:70b gpt-4o-2024-05-13, gemini-1.5-flash gpt-4-turbo-2024-04-09
experiment_name = "gemini-tuner"
llm = Gemini_LLM(api_key, ai_model)





def evaluateTuner(
    solution, explogger = None
):
    code = solution.code
    algorithm_name = solution.name
    error = ""
    algorithm = None

    budget = 500
    
    # strategy = "genetic_algorithm"
    strategy_options = {
    #    "budget": 15
    }
    iterations = 1 #number of kernel runs (1 because we use cached results anyways, by default 7)
    input_filepath = Path(f"/data/neocortex/repos/benchmark_hub/kernels/{application}_milo.json")
    cache_filepath = Path(f"/data/neocortex/repos/benchmark_hub/cachefiles/{application}_milo/{gpu}.json")

    # with open(input_filepath) as json_data:
    #     tune_params = json.load(json_data)

    safe_globals = {
        "np": np,
        "math": math,
        "random": random,
    #    "tune_params": tune_params,
    }
    local_env = {}
    exec(code, safe_globals, local_env)
    optimizer = local_env[algorithm_name](budget=budget)

    # Wrap the algorithm class in the OptAlgWrapper
    # for use in Kernel Tuner
    strategy = OptAlgWrapper(optimizer, budget=budget)

    results, env = tune_kernel_T1(
        input_filepath,
        cache_filepath,
        objective="time",
        objective_higher_is_better=False,
        simulation_mode=True,
        output_T4=False,
        iterations=iterations,
        device=gpu,
        strategy=strategy,
        strategy_options=strategy_options,
    )
    aoc = strategy.aoc
    print(aoc)

    #print(results)
    
    score = util.get_best_config(results, "time", False)["time"]
    feedback = f"The algorithm {algorithm_name} got an AOCC score of {aoc:0.4f} (higher is better)."
    print(algorithm_name, optimizer, score, aoc)
    #solution.add_metadata("tuner_results", results)
    #solution.add_metadata("tuner_env", env)
    solution.set_scores(aoc, feedback)
    return solution


input_filepath = Path(f"/data/neocortex/repos/benchmark_hub/kernels/{application}_milo.json")
kernel_file = open(input_filepath, "r")
kernel = kernel_file.read()
kernel_file.close()
#print(kernel)

#The specific kernel configuration is as follows:
#{kernel}
#We optimize the kernl for the following GPU: {gpu} (feel free to use an informed guess to warm-start the search process).

role_prompt = "You are a highly skilled computer scientist in the field of natural computing and hardware kernel tuning. Your task is to design novel metaheuristic algorithms to solve kernel tuner problems (integer, variable dimension, contraint)."
task_prompt = f"""
The optimization algorithm should handle a kernel tuning task with a given evaluation budget. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget)` function with optional additional arguments and the function `def __call__(self, func, searchspace)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. The `searchspace` object can be used to sample random instances, neighbouring instances using `searchspace.get_neighbors(param_config: tuple, neighbor_method='Hamming')` where neighbor_method can be any of ["strictly-adjacent", "adjacent", "Hamming"] and to check validity of parameter settings using `searchspace.is_param_config_valid(tuple(instance))`, nothing else. The dimensionality can be varied.
In addition, the variable `tune_params` is a dictionary containing the tuning parameters with their ranges and constraints, it can be obtained directly from the searchspace object `searchspace.tune_params`. The algorithm should be able to handle any number of tuning parameters, and the search space can be continuous or discrete. The algorithm should be able to handle any type of kernel tuning problem, including but not limited to vector addition, matrix multiplication, and convolution.

The specific kernel configuration is as follows:
```
{kernel}
```
We optimize the kernl for the following GPU: {gpu} (feel free to use an informed guess to warm-start the search process).

An example code structure is as follows:
```python
import numpy as np
import random

class GeneticAlgorithm:
    "Template for a genetic algorithm"

    def __init__(self, searchspace):
        self.pop_size = 20
        self.searchspace = searchspace
        self.tune_params = searchspace.tune_params.copy()
        self.mutation_chance = 10

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        # create initial population and run the search till evaluation budget is exhausted.
        # then retur the best solution found

    def generate_population(self):
        "Constraint-aware population creation method "
        pop = list(list(p) for p in self.searchspace.get_random_sample(self.pop_size))
        return pop

    def uniform_crossover(self, dna1, dna2):
        "Randomly crossover genes between dna1 and dna2."
        ind = np.random.random(len(dna1)) > 0.5
        child1 = [dna1[i] if ind[i] else dna2[i] for i in range(len(ind))]
        child2 = [dna2[i] if ind[i] else dna1[i] for i in range(len(ind))]
        return child1, child2

    def crossover(self, dna1, dna2):
        "Apply crossover method, repair dna if constraint-aware "
        dna1, dna2 = self.uniform_crossover(dna1, dna2)
        return self.repair(dna1), self.repair(dna2)

    def weighted_choice(self, population, n):
        "Randomly select n unique individuals from a weighted population, fitness determines probability of being selected."

        def random_index_betavariate(pop_size):
            # has a higher probability of returning index of item at the head of the list
            alpha = 1
            beta = 2.5
            return int(random.betavariate(alpha, beta) * pop_size)

        def random_index_weighted(pop_size):
            "Use weights to increase probability of selection."
            weights = [w for _, w in population]
            # invert because lower is better
            inverted_weights = [1.0 / w for w in weights]
            prefix_sum = np.cumsum(inverted_weights)
            total_weight = sum(inverted_weights)
            randf = random.random() * total_weight
            # return first index of prefix_sum larger than random number
            return next(i for i, v in enumerate(prefix_sum) if v > randf)

        random_index = random_index_betavariate
        indices = [random_index(len(population)) for _ in range(n)]
        chosen = []
        for ind in indices:
            while ind in chosen:
                ind = random_index(len(population))
            chosen.append(ind)
        return [population[ind][0] for ind in chosen]

    def mutate(self, dna):
        "Mutate DNA with 1/mutation_chance chance."
        # this is actually a neighbors problem with Hamming distance, choose randomly from returned searchspace list
        if int(random.random() * self.mutation_chance) == 0:
            neighbors = self.searchspace.get_neighbors(tuple(dna), neighbor_method="Hamming")
            if len(neighbors) > 0:
                return list(random.choice(neighbors))
        return dna


    def repair(self, dna):
        "It is possible that crossover methods yield a configuration that is not valid. "
        if not self.searchspace.is_param_config_valid(tuple(dna)):
            # dna is not valid, try to repair it
            # search for valid configurations neighboring this config
            # start from strictly-adjacent to increasingly allowing more neighbors
            for neighbor_method in ["strictly-adjacent", "adjacent", "Hamming"]:
                neighbors = self.searchspace.get_neighbors_no_cache(tuple(dna), neighbor_method=neighbor_method)

                # if we have found valid neighboring configurations, select one at random
                if len(neighbors) > 0:
                    new_dna = list(random.choice(neighbors))
                    return new_dna

        return dna

```

Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code: 
```python
<code>
```
"""

feedback_prompts = [
    "Either refine or redesign to improve the solution (and give it a distinct one-line description). Be concise.",
    "Refine and simplify the selected solution to improve it.",  # simplify mutation
    "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
]


import numpy as np

class RandomSearch:
    def __init__(self, budget=10000):
        self.budget = budget

    def __call__(self, func, searchspace):
        self.f_opt = np.Inf
        self.x_opt = None
        for i in range(self.budget):
            x = list(searchspace.get_random_sample(1))[0]

            print(x)
            f = func(x)
            print(f)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            
        return self.f_opt, self.x_opt


debug = False
if debug==True:

    algorithm_name = "RS"
    error = ""
    budget = 100
    
    # strategy = "genetic_algorithm"
    strategy_options = {
    #    "budget": 15
    }
    iterations = 1
    input_filepath = Path(f"/data/neocortex/repos/benchmark_hub/kernels/{application}_milo.json")
    cache_filepath = Path(f"/data/neocortex/repos/benchmark_hub/cachefiles/{application}_milo/{gpu}.json")

    with open(input_filepath) as json_data:
        tune_params = json.load(json_data)

    optimizer = RandomSearch(budget=budget)

    # Wrap the algorithm class in the OptAlgWrapper
    # for use in Kernel Tuner
    strategy = OptAlgWrapper(optimizer)

    results, env = tune_kernel_T1(
        input_filepath,
        cache_filepath,
        objective="time",
        objective_higher_is_better=False,
        simulation_mode=True,
        output_T4=False,
        iterations=iterations,
        device=gpu,
        strategy=strategy,
        strategy_options=strategy_options,
    )
    aoc = strategy.aoc
    print(aoc)

    print(results)
    
    score = util.get_best_config(results, "time", False)["time"]
    feedback = f"The algorithm {algorithm_name} got a score of {score:0.4f} (closer to zero is better)."
    print("RS", score)

else:

    for experiment_i in [1]:
        es = LLaMEA(
            evaluateTuner,
            llm=llm,
            budget=200,
            n_parents=4,
            n_offspring=12,
            eval_timeout=int(30), #30 seconds per algorithm
            role_prompt=role_prompt,
            task_prompt=task_prompt,
            mutation_prompts=feedback_prompts,
            experiment_name=experiment_name,
            elitism=False,
            HPO=False,
            max_workers=2,
        )
        print(es.run())




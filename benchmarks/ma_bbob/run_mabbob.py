import os
import numpy as np
from ioh import get_problem, logger
import re
from misc import aoc_logger, correct_aoc, OverBudgetException
from llamea import LLaMEA, Gemini_LLM
import time
import traceback
from itertools import product
from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
from smac import Scenario
from smac import AlgorithmConfigurationFacade

import iohinspector
import ioh
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm  # optional, for progress bars
import os

# Execution code starts here
api_key = os.getenv("GEMINI_API_KEY")
ai_model = "gemini-2.0-flash"  # -thinking-exp-01-21 gpt-4-turbo or gpt-3.5-turbo gpt-4o llama3:70b gpt-4o-2024-05-13, gemini-1.5-flash gpt-4-turbo-2024-04-09
experiment_name = "gemini-mabbob"
llm = Gemini_LLM(api_key, ai_model)

# Read in the instance specifications
weights = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "weights.csv"), index_col=0
)
iids = pd.read_csv(os.path.join(os.path.dirname(__file__), "iids.csv"), index_col=0)
opt_locs = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "opt_locs.csv"), index_col=0
)


def evaluateMABBOBWithHPO(solution, explogger=None):
    """
    Evaluates an optimization algorithm on the BBOB (Black-Box Optimization Benchmarking) suite and computes
    the Area Over the Convergence Curve (AOCC) to measure performance. In addddition, if a configuration space is provided, it
    applies Hyper-parameter optimization with SMAC first.

    Parameters:
    -----------
    solution : dict
        A dictionary containing "_solution" (the code to evaluate), "_name", "_description" and "_configspace"

    explogger : logger
        A class to log additional stuff for the experiment.

    Returns:
    --------
    solution : dict
        Updated solution with "_fitness", "_feedback", "incumbent" and optional "_error"

    Functionality:
    --------------
    - Executes the provided `code` string in the global context, allowing for dynamic inclusion of necessary components.
    - Iterates over a predefined set of dimensions (currently only 5), function IDs (1 to 24), and instance IDs (1 to 3).
    - For each problem, the specified algorithm is instantiated and executed with a defined budget.
    - AOCC is computed for each run, and the results are aggregated across all runs, problems, and repetitions.
    - The function handles cases where the algorithm exceeds its budget using an `OverBudgetException`.
    - Logs the results if an `explogger` is provided.
    - The function returns a feedback string, the mean AOCC score, and an error placeholder.

    Notes:
    ------
    - The budget for each algorithm run is set to 10,000.
    - The function currently only evaluates a single dimension (5), but this can be extended.
    - Hyperparameter Optimization (HPO) with SMAC is mentioned but not implemented.
    - The AOCC score is a metric where 1.0 is the best possible outcome, indicating optimal convergence.

    """
    auc_mean = 0
    auc_std = 0
    code = solution.code
    algorithm_name = solution.name
    safe_globals = {
        "np": np,
    }
    local_env = {}
    exec(code, safe_globals, local_env)

    budget_factor = 2000
    error = ""
    algorithm = None

    # perform a small run to check for any code errors
    l2_temp = aoc_logger(100, upper=1e2, triggers=[logger.trigger.ALWAYS])
    problem = get_problem(11, 1, 2)
    problem.attach_logger(l2_temp)
    try:
        algorithm = local_env[algorithm_name](budget=100, dim=2)
        algorithm(problem)
    except OverBudgetException:
        pass
    # Other exceptions are catched at LLaMEA level.

    # now optimize the hyper-parameters
    def get_mabbob_performance(config: Configuration, instance: str, seed: int = 0):
        dim, idx = instance.split(",")
        dim = int(dim[1:])
        idx = int(idx[:-1])
        budget = budget_factor * dim

        f_new = ioh.problem.ManyAffine(
            xopt=np.array(opt_locs.iloc[idx])[:dim],
            weights=np.array(weights.iloc[idx]),
            instances=np.array(iids.iloc[idx], dtype=int),
            n_variables=dim,
        )
        f_new.set_id(100)
        f_new.set_instance(idx)
        np.random.seed(seed)
        l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
        f_new.attach_logger(l2)
        try:
            algorithm = local_env[algorithm_name](
                budget=budget, dim=dim, **dict(config)
            )
            algorithm(f_new)
        except OverBudgetException:
            pass
        except Exception as e:
            pass
        auc = correct_aoc(f_new, l2, budget)
        return 1 - auc

    args = list(product([2, 5], range(0, 100)))
    np.random.shuffle(args)
    inst_feats = {str(arg): [arg[0]] for idx, arg in enumerate(args)}
    # inst_feats = {str(arg): [idx] for idx, arg in enumerate(args)}
    error = ""

    if solution.configspace == None:
        # No HPO possible, evaluate only the default
        incumbent = {}
        error = "The configuration space was not properly formatted or not present in your answer. The evaluation was done on the default configuration."
    else:
        configuration_space = solution.configspace
        # First try if the algorithm can run with a random configuration
        config = configuration_space.sample_configuration()
        l2_temp = aoc_logger(100, upper=1e2, triggers=[logger.trigger.ALWAYS])
        problem = get_problem(11, 1, 2)
        problem.attach_logger(l2_temp)
        try:
            algorithm = local_env[algorithm_name](budget=100, dim=2, **config)
            algorithm(problem)
        except OverBudgetException:
            pass
        # Other exceptions are catched at LLaMEA level.

        # Now that we are certain stuff runs, we can start the optimization
        scenario = Scenario(
            configuration_space,
            name=str(int(time.time())) + "-" + algorithm_name,
            deterministic=False,
            min_budget=10,
            max_budget=50,
            n_trials=500,
            walltime_limit=2000,
            instances=args,
            instance_features=inst_feats,
            output_directory="smac3_output"
            if explogger is None
            else explogger.dirname + "/smac"
            # n_workers=10
        )
        smac = AlgorithmConfigurationFacade(
            scenario, get_mabbob_performance, logging_level=30
        )
        incumbent = smac.optimize()

    # last but not least, perform the final validation

    aucs = []
    error = ""
    for dim in [2, 5]:
        for idx in range(100):
            budget = budget_factor * dim
            f_new = ioh.problem.ManyAffine(
                xopt=np.array(opt_locs.iloc[idx])[:dim],
                weights=np.array(weights.iloc[idx]),
                instances=np.array(iids.iloc[idx], dtype=int),
                n_variables=dim,
            )
            f_new.set_id(100)
            f_new.set_instance(idx)
            l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
            f_new.attach_logger(l2)

            try:
                algorithm = local_env[algorithm_name](
                    budget=budget, dim=dim, **dict(incumbent)
                )
                algorithm(f_new)
            except OverBudgetException:
                pass
            except Exception as e:
                error = f"There was an error in the algorithm configuration: An exception occured: {traceback.format_exc()}."
                auc = 0
                aucs.append(auc)
                l2.reset(f_new)
                f_new.reset()
                break  # stop the loop

            auc = correct_aoc(f_new, l2, budget)
            aucs.append(auc)
            l2.reset(f_new)
            f_new.reset()

    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)
    dict_hyperparams = dict(incumbent)
    feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.3f} with optimal hyperparameters {dict_hyperparams}. {error}"
    print(algorithm_name, algorithm, auc_mean, auc_std)

    solution.add_metadata("aucs", aucs)
    solution.add_metadata("incumbent", dict_hyperparams)
    solution.set_scores(auc_mean, feedback)

    return solution


role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."
task_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the Many Affine BBOB test suite of noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function with optional additional arguments and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
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
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            
        return self.f_opt, self.x_opt
```

In addition, any hyper-parameters the algorithm uses will be optimized by SMAC, for this, provide a Configuration space as Python dictionary (without the dim and budget parameters) and include all hyper-parameters in the __init__ function header.
An example configuration space is as follows:

```python
{
    "float_parameter": (0.1, 1.5),
    "int_parameter": (2, 10), 
    "categoral_parameter": ["mouse", "cat", "dog"]
}
```

Give an excellent and novel heuristic algorithm including its configuration space to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code: 
```python
<code>
```
# Space: 
```python
<configuration_space>
```
"""

feedback_prompts = [
    f"Either refine or redesign to improve the solution (and give it a distinct one-line description)."
]

for experiment_i in [1]:
    es = LLaMEA(
        evaluateMABBOBWithHPO,
        llm=llm,
        budget=500,
        n_parents=2,
        n_offspring=8,
        eval_timeout=int(3600),  # 1 hours per algorithm
        role_prompt=role_prompt,
        task_prompt=task_prompt,
        mutation_prompts=feedback_prompts,
        experiment_name=experiment_name,
        elitism=True,
        HPO=True,
    )
    print(es.run())

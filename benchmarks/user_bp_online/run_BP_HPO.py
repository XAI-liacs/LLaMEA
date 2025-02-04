import os
import numpy as np
import re
from llamea import LLaMEA
import warnings
import time

# Execution code starts here
api_key = os.getenv("OPENAI_API_KEY")
ai_model = "codellama:7b"  # "gpt-4o-2024-05-13"  # gpt-4-turbo or gpt-3.5-turbo gpt-4o llama3:70b gpt-4o-2024-05-13, gemini-1.5-flash gpt-4-turbo-2024-04-09
experiment_name = "BP-HPO-10-50"
if "gemini" in ai_model:
    api_key = os.environ["GEMINI_API_KEY"]

from itertools import product
from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
from smac import Scenario
from smac import HyperparameterOptimizationFacade
from benchmarks.user_bp_online.prob import BPONLINE

bp_prob = BPONLINE()


def evaluateWithHPO(solution, explogger=None):
    code = solution.solution
    algorithm_name = solution.name
    configuration_space = solution.configspace

    def evaluate(config, seed=0):
        np.random.seed(seed)
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Execute the code string in the new module's namespace
                exec(code, globals())
                alg = globals()[algorithm_name](**dict(config))

                return bp_prob.evaluateGreedy(alg)
        except Exception as e:
            return 100

    # inst_feats = {str(arg): [idx] for idx, arg in enumerate(args)}
    error = ""
    if configuration_space is None:
        # No HPO possible, evaluate only the default
        incumbent = {}
        error = "The configuration space was not properly formatted or not present in your answer. The evaluation was done on the default configuration."
        fitness = evaluate({})
    else:
        scenario = Scenario(
            configuration_space,
            # name=algorithm_name,
            walltime_limit=600,  # 10 minutes
            name=str(int(time.time())) + "-" + algorithm_name,
            deterministic=True,
            # min_budget=None,
            # max_budget=None,
            n_trials=100,
            # instances=None,
            output_directory="smac3_output"
            if explogger is None
            else explogger.dirname + "/smac"
            # n_workers=10
        )
        smac = HyperparameterOptimizationFacade(
            scenario, evaluate, overwrite=True, logging_level=30
        )
        incumbent = smac.optimize()
        fitness = evaluate(dict(incumbent))

    fitness = -1 * fitness  # we optimize (not minimize)
    dict_hyperparams = dict(incumbent)
    feedback = f"The heuristic {algorithm_name} got an average fitness of {fitness:0.2f} (closer to zero is better)  with optimal hyperparameters {dict_hyperparams}."

    solution.add_metadata("incumbent", dict_hyperparams)
    solution.set_scores(fitness, feedback)

    return solution


role_prompt = "You are a highly skilled computer scientist your task it to design novel and efficient heuristics in Python."
task_prompt = """
I need help designing a novel score function that scoring a set of bins to assign an item.
In each step, the item will be assigned to the bin with the maximum score. If the rest capacity of a bin equals the maximum capacity, it will not be used. The final goal is to minimize the number of used bins.

The heuristic algorithm class should contain two functions an "__init__()" function containing any hyper-parameters that can be optimmized, and a "score(self, item, bins)" function, which gives back the 'scores'.
'item' and 'bins' are the size of the current item and the rest capacities of feasible bins, which are larger than the item size.
The output named 'scores' is the scores for the bins for assignment.
Note that 'item' is of type int, while 'bins' and 'scores' are both Numpy arrays. The novel function should be sufficiently complex in order to achieve better performance. It is important to ensure self-consistency.

An example code template to show the structure is as follows:
```python
import numpy as np

class Sample:
    def __init__(self, s1=1.0, s2=100):
        self.s1 = s1
        self.s2 = s2

    def score(self, item, bins):
        # heuristic here
    return scores
```

In addition, any hyper-parameters the algorithm uses will be optimized by SMAC, for this, provide a Configuration space as Python dictionary (without the item and bins parameters) and include all hyper-parameters to be optimized in the __init__ function header.
An example configuration space is as follows:

```python
{
    "float_parameter": (0.1, 1.5),
    "int_parameter": (2, 10), 
    "categoral_parameter": ["mouse", "cat", "dog"]
}
```

Give an excellent and novel heuristic including its configuration space to solve this task and also give it a one-line description with the main idea. 
"""

feedback_prompts = [
    "Either refine or redesign to improve the solution (and give it a one-line description with the main idea).",
    "Make small changes to the solution in order to improve it and give a new one-line description.",
    "Generate a completely new heuristic, approaching it from a different angle.",
]

for experiment_i in [1]:
    es = LLaMEA(
        evaluateWithHPO,
        n_parents=10,
        n_offspring=50,
        role_prompt=role_prompt,
        task_prompt=task_prompt,
        mutation_prompts=feedback_prompts,
        api_key=api_key,
        experiment_name=experiment_name,
        model=ai_model,
        budget=1000,
        eval_timeout=1200,
        elitism=True,
        HPO=True,
    )
    print(es.run())

# This is a more advanced example of how to use the LLaMEA algorithm with the Gemini LLM to generate optimization algorithms for the BBOB test suite.
# It includes the in-the-loop hyper-parameter optimization (HPO) using SMAC extension of LLaMEA and a more complex evaluation function.
# We have to define the following components for LLaMEA to work:
# - An evaluation function that executes the generated code and evaluates its performance.
# - A task prompt that describes the problem to be solved.
# - An LLM instance that will generate the code based on the task prompt.


import os
import re
import time
from itertools import product

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ioh import get_problem, logger
from smac import AlgorithmConfigurationFacade, Scenario

from llamea import Gemini_LLM, LLaMEA
from misc import OverBudgetException, aoc_logger, correct_aoc

if __name__ == "__main__":
    # Execution code starts here
    api_key = os.getenv("GEMINI_API_KEY")
    ai_model = "gemini-1.5-flash"
    experiment_name = "pop1-5-HPO"
    llm = Gemini_LLM(api_key, ai_model)

    def evaluateBBOBWithHPO(solution, explogger=None):
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
        exec(code, globals())
        dim = 5
        budget = 2000 * dim
        error = ""
        algorithm = None

        # perform a small run to check for any code errors
        l2_temp = aoc_logger(100, upper=1e2, triggers=[logger.trigger.ALWAYS])
        problem = get_problem(11, 1, dim)
        problem.attach_logger(l2_temp)
        try:
            algorithm = globals()[algorithm_name](budget=100, dim=dim)
            algorithm(problem)
        except OverBudgetException:
            pass

        # now optimize the hyper-parameters
        def get_bbob_performance(config: Configuration, instance: str, seed: int = 0):
            np.random.seed(seed)
            fid, iid = instance.split(",")
            fid = int(fid[1:])
            iid = int(iid[:-1])
            problem = get_problem(fid, iid, dim)
            l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
            problem.attach_logger(l2)
            try:
                algorithm = globals()[algorithm_name](
                    budget=budget, dim=dim, **dict(config)
                )
                algorithm(problem)
            except OverBudgetException:
                pass
            except Exception as e:
                print(problem.state, budget, e)
            auc = correct_aoc(problem, l2, budget)
            return 1 - auc

        args = list(product(range(1, 25), range(1, 4)))
        np.random.shuffle(args)
        inst_feats = {str(arg): [arg[0]] for idx, arg in enumerate(args)}
        # inst_feats = {str(arg): [idx] for idx, arg in enumerate(args)}
        error = ""

        if "_configspace" not in solution.keys():
            # No HPO possible, evaluate only the default
            incumbent = {}
            error = "The configuration space was not properly formatted or not present in your answer. The evaluation was done on the default configuration."
        else:
            configuration_space = solution.configspace
            scenario = Scenario(
                configuration_space,
                name=str(int(time.time())) + "-" + algorithm_name,
                deterministic=False,
                min_budget=12,
                max_budget=200,
                n_trials=2000,
                instances=args,
                instance_features=inst_feats,
                output_directory="smac3_output"
                if explogger is None
                else explogger.dirname + "/smac"
                # n_workers=10
            )
            smac = AlgorithmConfigurationFacade(
                scenario, get_bbob_performance, logging_level=30
            )
            incumbent = smac.optimize()

        # last but not least, perform the final validation

        l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
        aucs = []
        for fid in np.arange(1, 25):
            for iid in [1, 2, 3]:  # , 4, 5]
                problem = get_problem(fid, iid, dim)
                problem.attach_logger(l2)
                for rep in range(3):
                    np.random.seed(rep)
                    try:
                        algorithm = globals()[algorithm_name](
                            budget=budget, dim=dim, **dict(incumbent)
                        )
                        algorithm(problem)
                    except OverBudgetException:
                        pass
                    auc = correct_aoc(problem, l2, budget)
                    aucs.append(auc)
                    l2.reset(problem)
                    problem.reset()

        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)
        dict_hyperparams = dict(incumbent)
        feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with optimal hyperparameters {dict_hyperparams}."
        print(algorithm_name, algorithm, auc_mean, auc_std)

        solution.add_metadata("aucs", aucs)
        solution.add_metadata("incumbent", dict_hyperparams)
        solution.set_scores(auc_mean, feedback)

        return solution

    role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."
    task_prompt = """
    The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function with optional additional arguments and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
    The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.

    In addition, any hyper-parameters the algorithm uses will be optimized by SMAC, for this, provide a Configuration space as Python dictionary (without the dim and budget parameters) and include all hyper-parameters in the __init__ function header.
    An example configuration space is as follows:

    ```python
    {
        "float_parameter": (0.1, 1.5),
        "int_parameter": (2, 10), 
        "categoral_parameter": ["mouse", "cat", "dog"]
    }
    ```

    Give an excellent and novel heuristic algorithm including its configuration space to solve this task and also give it a one-line description, describing the main idea. 
    """

    format_prompt = """
    Give the response in the format:
    # Description: <short-description>
    # Code: <code>
    # Space: <configuration_space>
    """

    example_prompt = """
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
    """

    feedback_prompts = [
        f"Either refine or redesign to improve the solution (and give it a distinct one-line description)."
    ]

    for experiment_i in [1]:
        es = LLaMEA(
            evaluateBBOBWithHPO,
            llm=llm,
            role_prompt=role_prompt,
            task_prompt=task_prompt,
            example_prompt=example_prompt,
            output_format_prompt=format_prompt,
            mutation_prompts=feedback_prompts,
            experiment_name=experiment_name,
            elitism=True,
            HPO=True,
        )
        print(es.run())

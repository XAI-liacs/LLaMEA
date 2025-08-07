# This is a minimal example without any other dependencies than LLaMEA and the Gemini LLM.

import os
import re

import numpy as np

from llamea import Dummy_LLM, Gemini_LLM, LLaMEA, OpenAI_LLM

if __name__ == "__main__":
    # Execution code starts here
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    ai_model = "gpt-4.1-nano-2025-04-14"
    experiment_name = "minimal_example"
    # llm = OpenAI_LLM(api_key=openai_api_key, model=ai_model)
    # llm = Gemini_LLM(api_key=gemini_api_key, model=ai_model)
    llm = Dummy_LLM(ai_model)

    # We define the evaluation function that executes the generated algorithm (solution.code) and return a random fitness.
    def evaluate(solution, explogger=None):
        code = solution.code
        algorithm_name = solution.name
        # Execute the generated code in the global scope
        exec(code, globals())

        # Initialize the algorithm with a budget and dimension
        budget = 10
        dim = 5
        algorithm = globals()[algorithm_name](budget=budget, dim=dim)
        # Simulate the algorithm execution
        # Here we just return a random fitness value for demonstration purposes
        fitness = np.random.rand()
        feedback = f"The algorithm {algorithm_name} got score of {fitness:0.2f} (1.0 is the best)."
        solution.set_scores(fitness, feedback)
        return solution

    # The task prompt describes the problem to be solved by the LLaMEA algorithm.
    task_prompt = """
    The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
    The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
    Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description with the main idea. You can only use the numpy (1.26) library, no other libraries are allowed.
    """
    es = LLaMEA(
        evaluate,
        n_parents=1,
        n_offspring=1,
        llm=llm,
        task_prompt=task_prompt,
        experiment_name=experiment_name,
        elitism=True,
        HPO=True,
        budget=5,
    )
    print(es.run())

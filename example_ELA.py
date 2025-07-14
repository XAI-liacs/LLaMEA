# This is a basic example on how to use LLaMEA for Automated Machine Learning tasks.
# Here we evolve ML pipelines to solve a breast-cancer classification task.

# We have to define the following components for LLaMEA to work:
# - An evaluation function that executes the generated code and evaluates its performance. In this case we evaluate the accuracy of the generated ML pipeline on a breast cancer dataset.
# - A task prompt that describes the problem to be solved. In this case, we describe the task of classifying breast cancer using a machine learning pipeline.
# - An LLM instance that will generate the code based on the task prompt.

import json
import math
import os
import random
import re
import time
import traceback

import numpy as np
import pandas as pd

from llamea import Gemini_LLM, LLaMEA, OpenAI_LLM
from misc import OverBudgetException

import numpy as np
import xgboost as xgb
from pflacco.classical_ela_features import *
from pflacco.sampling import create_initial_sample
import pandas as pd


def preprocess_data(data):
    has_group = False
    if "group" in data:
        group = data["group"]
        data = data.drop("group", axis=1)
        has_group = True
    
    data = data.dropna()
    data = data[data.columns.drop(list(data.filter(regex='costs_runtime')))]
    #data = data.drop("ela_level.mmce_lda_10", axis=1)
    data = data.drop("pca.expl_var_PC1.cor_x", axis=1)
    data = data.drop("pca.expl_var_PC1.cov_x", axis=1)
    data = data.drop("pca.expl_var.cov_x", axis=1)
    data = data.drop("pca.expl_var.cor_x", axis=1)
    
    
    if has_group:
        data["group"] = group
    return(data)



def test_func(x):
    return np.sum(x**2)

def evaluate_function(solution, logger=None):
    code = solution.code
    algorithm_name = solution.name

    exec(code, globals())

    algorithm = None

    # Final validation
    DIM = 5 #change to appropriate dimensionality
    algorithm = globals()[algorithm_name](DIM)
    f = algorithm.f
    
    
    model1 = xgb.XGBClassifier(objective="binary:logistic")
    model1.load_model("models/model_5d_basins_scaled.json")
    
    model2 = xgb.XGBClassifier(objective="binary:logistic")
    model2.load_model("models/model_5d_separable_scaled.json")
    
    problem = f
    X = create_initial_sample(DIM,n=250*DIM, lower_bound = -5, upper_bound = 5)
    y = X.apply(problem, axis = 1)
    
    
    y[y==0] = 0.1**100 #since y=0 breaks log
    if y.max() == y.min():
        for i in range(len(y)):
            y[i] = 0
    else:
        X_scaled=(X-X.min())/(X.max()-X.min())
        y_scaled=(y-y.min())/(y.max()-y.min())
    
    
    ela_meta_scaled = calculate_ela_meta(X_scaled, y_scaled)

    ela_distr_scaled = calculate_ela_distribution(X_scaled, y_scaled)
    
    nbc_scaled = calculate_nbc(X_scaled, y_scaled)
    
    disp_scaled = calculate_dispersion(X_scaled, y_scaled)
    
    pca_scaled = calculate_pca(X_scaled, y_scaled)
    
    ic_scaled = calculate_information_content(X_scaled, y_scaled)
    
    
    all_features_scaled = {**ela_meta_scaled, **ela_distr_scaled, **nbc_scaled, **disp_scaled, **pca_scaled, **ic_scaled}
    
    
    all_features_scaled = {k:[v] for k,v in all_features_scaled.items()} 
    all_features_scaled = pd.DataFrame.from_dict(all_features_scaled)
    all_features_scaled = preprocess_data(all_features_scaled)
    
    
    result_1 = model1.predict_proba(all_features_scaled)
    result_2 = model2.predict_proba(all_features_scaled)
    score = ( result_1[0][1] + result_2[0][1] ) / 2
    solution.set_scores(
        score,
        f"The optimization landscape {algorithm_name} scored {result_1[0][1]:.3f} on the basins feature and {result_2[0][1]:.3f}  on separability (higher is better, 1.0 is the best).",
    )
    return solution



class ELAproblem:
    """
    Problem class for evaluating ELA landscapes.

    """

    def __init__(
        self, logger=None, name="ELA", eval_timeout=360
    ):
        

        self.task_prompt = f"""
You are a highly skilled computer scientist in the field optimization and benchmarking. Your task is to design novel mathematical functions to be used as black-box optimization benchmark landscapes.
The code you need to write is a class with a function `f` with one parameter `x` which is a realvalued sample (numpy array). 
The optimization function should be separable with basins and it should be able to handle different dimensionalities.
The class should also have a __init__(dim) function, that received the number of dimensions for the function.
The function will be evaluated between per dimension lower bound of -5.0 and upper bound of 5.0.
"""
        self.example_prompt = """
An example code structure is as follows:
```python
import numpy as np

class landscape:
    
    def __init__(dim=5):
        self.dim = dim

    def test_func(self, x):
        return np.sum(x**2)
```
"""
        self.format_prompt = """

Give a novel Python class with an optimization landscape function and a short description with the main idea of the benchmark function. Give the response in the format:
# Description: <short-description>
# Code: 
```python
<code>
```
"""

    def get_prompt(self):
        """
        Returns the problem description and answer format.
        """
        return self.task_prompt + self.example_prompt + self.format_prompt

budget = 200
if __name__ == "__main__":
    # Execution code starts here
    api_key = os.getenv("OPENAI_API_KEY")
    #api_key = os.getenv("GEMINI_API_KEY")
    llm = OpenAI_LLM(api_key,"o4-mini-2025-04-16") #Done
    experiment_name = "ELA"
    #llm = Gemini_LLM(api_key, ai_model)

    problem = ELAproblem()

    mutation_prompts = []
    mutation_prompts.append("Create a new landscape class based on the selected code and improve the separability score (make sure the function is separable, meaning independent functions per dimension.)")
    mutation_prompts.append("Create a new landscape class based on the selected code and improve the basins score.")
    mutation_prompts.append("Create a new landscape class that is completely different from the selected solution but still be seperable with basins.")


    for experiment_i in [1]:
        es = LLaMEA(
            evaluate_function,
            n_parents=8,
            n_offspring=16,
            llm=llm,
            task_prompt=problem.task_prompt,
            example_prompt=problem.example_prompt,
            output_format_prompt=problem.format_prompt,
            mutation_prompts=mutation_prompts,
            experiment_name=experiment_name,
            elitism=True,
            HPO=False,
            budget=budget,
            max_workers=1,
            parallel_backend="threading",
        )
        print(es.run())

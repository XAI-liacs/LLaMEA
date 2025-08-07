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

from llamea import Gemini_LLM, LLaMEA, OpenAI_LLM, Ollama_LLM

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




class ELAproblem:
    """
    Problem class for evaluating ELA landscapes.

    """

    def __init__(
        self, logger=None, name="ELA", features=["basins_scaled", "separable_scaled"], eval_timeout=360
    ):

        self.features = features # choice from ["basins_scaled", "separable_scaled", "globallocal_scaled", "multimodal_scaled", "structure_scaled"]
        self.feature_descriptions = {
            "basins_scaled": "multiple basins of attraction",
            "separable_scaled": "separable, meaning independent functions per dimension",
            "globallocal_scaled": "global and local optima of the landscape should have a large fitness difference",
            "multimodal_scaled": "it should be multimodal, meaning multiple local/global optima",
            "structure_scaled": "it should have a global structure."
        }
        self.task_prompt = f"""
You are a highly skilled computer scientist in the field optimization and benchmarking. Your task is to design novel mathematical functions to be used as black-box optimization benchmark landscapes.
The code you need to write is a class with a function `f` with one parameter `x` which is a realvalued sample (numpy array). 
The optimization function should have the following properties:"""
        for feature in self.features:
            self.task_prompt += f"\n- {self.feature_descriptions[feature]} ({feature})"
        self.task_prompt += """
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

    def evaluate_function(self, solution, logger=None):
        code = solution.code
        algorithm_name = solution.name

        exec(code, globals())

        algorithm = None
        # Final validation
        DIM = 5 #change to appropriate dimensionality
        algorithm = globals()[algorithm_name](DIM)
        f = algorithm.f

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
        solution.add_metadata("ela_features", all_features_scaled.to_numpy())


        feature_results = {}
        results = []
        feedback = f"The optimization landscape {algorithm_name} scored on:"
        for feature in self.features:
            model = xgb.XGBClassifier(objective="binary:logistic")
            model.load_model(f"models/model_5d_{feature}.json")
            feature_results[feature] = model.predict_proba(all_features_scaled)[0][1]
            results.append(feature_results[feature])
            solution.add_metadata(f"score_{feature}", feature_results[feature])
            feedback += f"{feature} {feature_results[feature]:.3f}, "


        score = np.mean(results, axis=0)
        solution.set_scores(
            score,
            f"{feedback} (higher is better, 1.0 is the best).",
        )
        return solution


budget = 200
if __name__ == "__main__":
    # use argparse to select the LLM.
    import os
    import argparse
    parser = argparse.ArgumentParser(description="Run ELA problem with LLaMEA.")
    parser.add_argument(
        "--llm",
        type=str,
        choices=["openai", "gemini", "ollama"],
        default="ollama",
        help="Select the LLM to use for code generation.",

    )
    parser.add_argument(
        "--ai_model",
        type=str,
        default="gemma3:12b",
        help="Select the AI model to use for code generation.",
    )
    
    args = parser.parse_args()
    ai_model = args.ai_model
    if args.llm == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        llm = OpenAI_LLM(api_key, "o4-mini-2025-04-16")
    elif args.llm == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        llm = Gemini_LLM(api_key, ai_model)
    elif args.llm == "ollama": 
        llm = Ollama_LLM(ai_model)
    # Execution code starts here
    api_key = os.getenv("OPENAI_API_KEY")
    #api_key = os.getenv("GEMINI_API_KEY")
    #llm = OpenAI_LLM(api_key,"o4-mini-2025-04-16") #Done
    
    #llm = Gemini_LLM(api_key, ai_model)

    feature_combinations = [
        ["basins_scaled", "separable_scaled"],
        ["multimodal_scaled", "structure_scaled"],
        ["multimodal_scaled", "separable_scaled"],
        ["multimodal_scaled", "globallocal_scaled"],
        ["structure_scaled", "separable_scaled"],
    ]

    experiment_name = f"ELA-{ai_model}"
    for combi in feature_combinations:
        problem = ELAproblem(name=f"ELA_{'_'.join(combi)}", features=combi, eval_timeout=360)

        mutation_prompts = []
        for feature in problem.features:
            mutation_prompts.append(f"Create a new landscape class based on the selected code and improve the {feature} score, meaning: {problem.feature_descriptions[feature]}.")
        mutation_prompts.append("Create a new landscape class that is completely different from the selected solution but still adheres to the properties outlined in the task description.")

        for experiment_i in [1,2,3,4,5]:
            es = LLaMEA(
                problem.evaluate_function,
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
                max_workers=4,
                parallel_backend="threading",
            )
            print(es.run())

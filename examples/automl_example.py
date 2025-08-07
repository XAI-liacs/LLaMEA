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
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from llamea import Gemini_LLM, LLaMEA
from misc import OverBudgetException, aoc_logger, correct_aoc


class AutoML:
    """
    Problem class for evaluating AutoML pipelines (sample).

    """

    def __init__(
        self, logger=None, datasets=None, name="AutoML-breast_cancer", eval_timeout=360
    ):
        X, y = load_breast_cancer(return_X_y=True)
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(X, y, random_state=1)

        self.task_prompt = f"""
You are a highly skilled computer scientist in the field machine learning. Your task is to design novel machine learning pipelines for a given dataset and task.
The pipeline in this case should handle a breast cancer classification task. Your task is to write the Python code. The code should contain an `__init__(self, X, y)` function that trains a machine learning model and the function `def __call__(self, X)`, which should predict the samples in X and return the predictions.
The training data X has shape {self.X_train.shape} and y has shape {self.y_train.shape}.
"""
        self.example_prompt = """
An example code structure is as follows:
```python
import numpy as np
import sklearn

class AlgorithmName:
    "Template for a ML pipeline"

    def __init__(self, X, y):
        self.train(X, y)

    def train(self, X, y):
        # Standardize the feature data
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Let's create and train a logistic regression model
        lr_model = sklearn.linear_model.LogisticRegression()
        lr_model.fit(X_train, y_train)
        self.model = lr_model
        
    def __call__(self, X):
        # predict using the trained model
        return self.model.predict(X)
```
"""
        self.format_prompt = """

Give an excellent and novel ML pipeline to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
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


if __name__ == "__main__":
    # Execution code starts here
    api_key = os.getenv("GEMINI_API_KEY")
    ai_model = "gemini-1.5-flash"
    experiment_name = "automl-breast-cancer"
    llm = Gemini_LLM(api_key, ai_model)

    AutoML_problem = AutoML()

    def evaluate(solution, explogger=None):
        """
        Evaluates a solution on the AutoML task.
        """
        code = solution.code
        algorithm_name = solution.name
        safe_globals = {
            "sklearn": sklearn,
            "math": math,
            "random": random,
            "np": np,
            "pd": pd,
        }

        exec(code, globals())

        algorithm = None

        # Final validation
        algorithm = globals()[algorithm_name](
            AutoML_problem.X_train, AutoML_problem.y_train
        )
        y_pred = algorithm(AutoML_problem.X_test)
        score = accuracy_score(AutoML_problem.y_test, y_pred)

        solution.set_scores(
            score,
            f"The algorithm {algorithm_name} scored {score:.3f} on accuracy (higher is better, 1.0 is the best).",
        )

        return solution

    for experiment_i in [1]:
        # A 1+1 strategy
        es = LLaMEA(
            evaluate,
            n_parents=1,
            n_offspring=1,
            llm=llm,
            task_prompt=AutoML_problem.task_prompt,
            example_prompt=AutoML_problem.example_prompt,
            output_format_prompt=AutoML_problem.format_prompt,
            experiment_name=experiment_name,
            elitism=True,
            HPO=False,
            budget=100,
        )
        print(es.run())

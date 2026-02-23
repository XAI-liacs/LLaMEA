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

from llamea import Gemini_LLM, LLaMEA, OpenAI_LLM, Ollama_LLM, Multi_LLM

import numpy as np
import xgboost as xgb
from pflacco.classical_ela_features import *
from pflacco.sampling import create_initial_sample
import pandas as pd
import joblib

from dataclasses import dataclass
import json

@dataclass
class SeparabilityReport:
    separable: bool
    percent_noncompliance: float
    mean_norm_interaction: float
    max_norm_interaction: float
    per_pair_score: dict  # {(i,j): score}
    details: dict         # misc diagnostics

def evaluate_separability(f, dim, bounds=(-5.0, 5.0), samples=128, h=None,
                          tol=1e-6, rng=None):
    """
    Black-box separability test for f: R^dim -> R.
    Uses finite-difference Hessian cross-terms + a superposition check.

    Parameters
    ----------
    f : callable
        Function taking 1D np.ndarray (len=dim) -> float.
    dim : int
        Dimensionality.
    bounds : tuple(float, float)
        Box to sample from (same for all dims).
    samples : int
        Number of random points for finite-difference probing.
    h : float or None
        Step size for finite differences. If None, set to 1e-3 * (ub-lb).
    tol : float
        Threshold on normalized interaction to count as non-compliant.
    rng : np.random.Generator or None
        For reproducibility.

    Returns
    -------
    SeparabilityReport
    """
    if rng is None:
        rng = np.random.default_rng()
    lb, ub = map(float, bounds)
    box = ub - lb
    if h is None:
        h = 1e-3 * box
    eps = 1e-12

    # Draw sample points inside the box
    X = rng.uniform(lb + 2*h, ub - 2*h, size=(samples, dim))

    # Helpers for finite differences
    def e(i):
        v = np.zeros(dim); v[i] = 1.0
        return v

    def f_at(x):
        return float(f(x))

    def d2_ij(x, i, j):
        """Central difference mixed second derivative ∂^2 f / ∂x_i ∂x_j."""
        ei, ej = e(i), e(j)
        return (
            f_at(x + h*ei + h*ej) - f_at(x + h*ei - h*ej)
            - f_at(x - h*ei + h*ej) + f_at(x - h*ei - h*ej)
        ) / (4*h*h)

    def d2_ii(x, i):
        """Central difference second derivative ∂^2 f / ∂x_i^2."""
        ei = e(i)
        return (f_at(x + h*ei) - 2.0*f_at(x) + f_at(x - h*ei)) / (h*h)

    # Estimate diagonal curvature scales per dim for normalization
    Hii_vals = np.zeros((samples, dim))
    for s in range(samples):
        x = X[s]
        fx = f_at(x)  # cache center eval to reduce calls on diagonal calc
        for i in range(dim):
            ei = np.zeros(dim); ei[i] = 1.0
            Hii_vals[s, i] = (f_at(x + h*ei) - 2.0*fx + f_at(x - h*ei)) / (h*h)

    # Robust scale per axis: median absolute curvature (avoid zero division)
    scale_i = np.median(np.abs(Hii_vals), axis=0) + eps

    # Compute normalized cross interactions
    pair_scores = {}  # aggregated per pair over samples (median of |H_ij|/sqrt(scale_i*scale_j))
    all_norm_ijs = []  # for overall stats
    per_pair_all = { (i,j): [] for i in range(dim) for j in range(i+1, dim) }

    for s in range(samples):
        x = X[s]
        # Precompute f(x) once for superposition later
        for i in range(dim):
            pass  # nothing extra here; left for symmetry

        for i in range(dim):
            for j in range(i+1, dim):
                Hij = d2_ij(x, i, j)
                norm = abs(Hij) / np.sqrt(scale_i[i]*scale_i[j])
                per_pair_all[(i,j)].append(norm)
                all_norm_ijs.append(norm)

    for (i,j), arr in per_pair_all.items():
        pair_scores[(i,j)] = float(np.median(arr))

    mean_norm_interaction = float(np.mean(all_norm_ijs)) if all_norm_ijs else 0.0
    max_norm_interaction = float(np.max(all_norm_ijs)) if all_norm_ijs else 0.0

    # Superposition sanity check: does Δ_i depend on j?
    # If separable, Δ_i(x) ≈ f(x+he_i)-f(x) should not change when j is perturbed.
    sup_violations = 0
    sup_total = 0
    for s in range(samples):
        x = X[s]
        fx = f_at(x)
        for i in range(dim):
            ei = e(i)
            d_i = f_at(x + h*ei) - fx
            for j in range(dim):
                if j == i: continue
                ej = e(j)
                d_i_with_j = f_at(x + h*ej + h*ei) - f_at(x + h*ej)
                # Relative interaction score (bounded, scale-free-ish)
                denom = abs(d_i) + abs(d_i_with_j) + eps
                rel = abs(d_i_with_j - d_i) / denom
                sup_total += 1
                if rel > tol:
                    sup_violations += 1

    # Combine both notions into a single % non-compliance.
    # 50/50 weight between Hessian cross-terms and superposition violations.
    # For Hessian: count fraction above tol
    hess_violations = np.sum(np.array(all_norm_ijs) > tol)
    hess_total = len(all_norm_ijs)
    frac_hess = (hess_violations / max(1, hess_total))
    frac_sup = (sup_violations / max(1, sup_total))
    percent_noncompliance = 100.0 * (0.5*frac_hess + 0.5*frac_sup)

    separable = percent_noncompliance < 0.5  # basically “all but numerical noise”

    details = {
        "tol": tol,
        "h": h,
        "samples": samples,
        "bounds": (lb, ub),
        "diag_curvature_scale_per_dim": scale_i.tolist(),
        "hessian_fraction_violations": frac_hess,
        "superposition_fraction_violations": frac_sup,
    }

    return SeparabilityReport(
        separable=separable,
        percent_noncompliance=float(percent_noncompliance),
        mean_norm_interaction=mean_norm_interaction,
        max_norm_interaction=max_norm_interaction,
        per_pair_score={str(k): v for k,v in pair_scores.items()},
        details=details,
    )

def preprocess_data(data):
    has_group = False
    if "group" in data:
        group = data["group"]
        data = data.drop("group", axis=1)
        has_group = True
    
    data = data.dropna()
    data = data[data.columns.drop(list(data.filter(regex='costs_runtime')))]
    #data = data.drop("ela_level.mmce_lda_10", axis=1)

    if has_group:
        data["group"] = group
    return(data)



def test_func(x):
    return np.sum(x**2)


def ela_distance(s1, s2):
    """
    Calculate the ELA distance between two solutions based on their metadata.
    """
    if "ela_features" not in s1.metadata or "ela_features" not in s2.metadata:
        return 0.0  # No features to compare

    features1 = s1.metadata["ela_features"]
    features2 = s2.metadata["ela_features"]

    # Replace NaN values with zeros
    features1 = np.nan_to_num(features1, nan=0.0)
    features2 = np.nan_to_num(features2, nan=0.0)

    try:
        scaler = joblib.load(f"ela_scaler.joblib")
        features1 = scaler.transform(features1)
        features2 = scaler.transform(features2)
    except:
        pass

    # Calculate the Manhattan distance between the two feature vectors
    if len(features1) != len(features2):
        # fallback to Euclidean distance if lengths differ
        return np.linalg.norm(features1 - features2)
    return np.sum(np.abs(features1 - features2))


class ELAproblem:
    """
    Problem class for evaluating ELA landscapes.

    """

    def __init__(
        self, logger=None, name="ELA", features=["Basins", "Separable"], dims=[2,5,10], eval_timeout=360
    ):
        self.dims = dims
        self.features = features # choice from ["Basins", "Separable", "GlobalLocal", "Multimodality", "Structure", "Homogeneous"]
        self.feature_descriptions = {
            "Basins": "Basin size homogeneity, meaning the size relation (largest to smallest) of all basins of attraction should be homogeneous.",
            "Separable": "Separable, meaning independent functions per dimension. Meaning, a problem may be partitioned into subproblems which are then of lower dimensionality and should be considerably easier to solve.",
            "GlobalLocal": "It should have a global local minima contrast, GlobalLocal refers to the difference between global and local peaks in comparison to the average fitness level of a problem. It thus determines if very good peaks are easily recognized as such.",
            "Multimodality": "it should be multimodal, Multimodality refers to the number of local minima of a problem.",
            "Structure": "It should have a clear global structure. Global structure is what remains after deleting all non-optimal points.",
            "Homogeneous": "The search space should be homogeneous. Which refers to a search space without phase transitions. Its overall appearance is similar in different search space areas.",
            "NOT Homogeneous": "The search space should be not homogeneous. Which refers to a search space with phase transitions. Its overall appearance is different in different search space areas.",
            "NOT Basins": "The search space should be not have basin size homogeneity. Which refers to a search space where the size relation (largest to smallest) of all basins of attraction is not homogeneous.",
        }
        self.task_prompt = f"""
You are a highly skilled computer scientist in the field optimization and benchmarking. Your task is to design novel mathematical functions to be used as black-box optimization benchmark landscapes.
The code you need to write is a class with a function `f` with one parameter `x` which is a realvalued sample (numpy array). 
The optimization function should have the following properties: \n- it will be used as minimization problem (so the global optimum should be the minimum value of the function)."""
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

    def f(self, x):
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
        feature_results = {}
        results = []
        for DIM in self.dims:
            #DIM = 5 #change to appropriate dimensionality
            algorithm = globals()[algorithm_name](DIM)
            f = algorithm.f

            problem = f

            all_features = []
            for seed in range(5):
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
                #ela_level = calculate_ela_level(X, y)
                ela_distr_scaled = calculate_ela_distribution(X_scaled, y_scaled)
                
                nbc_scaled = calculate_nbc(X_scaled, y_scaled)
                
                disp_scaled = calculate_dispersion(X_scaled, y_scaled)
                
                pca_scaled = calculate_pca(X_scaled, y_scaled)
                
                ic_scaled = calculate_information_content(X_scaled, y_scaled)
                
                all_features_scaled = {**ela_meta_scaled, **ela_distr_scaled, **nbc_scaled, **disp_scaled, **pca_scaled, **ic_scaled}
                
                
                all_features_scaled = {k:[v] for k,v in all_features_scaled.items()} 
                all_features_scaled = pd.DataFrame.from_dict(all_features_scaled)
                
                all_features_scaled = preprocess_data(all_features_scaled)
                all_features.append(all_features_scaled)
                
            all_features_pandas = pd.concat(all_features)
            all_features_mean = all_features_pandas.mean()
            
            d =  {"dim": DIM} 
            
            if DIM == 5:
                solution.add_metadata("ela_features", all_features_mean.to_numpy())
            else:
                solution.add_metadata(f"ela_features_{DIM}D", all_features_mean.to_numpy())


            
            feedback = f"The optimization landscape {algorithm_name} scored on:"
            for feature in self.features:
                if feature == "Separable":
                    bounds = (-5.0, 5.0)
                    report = evaluate_separability(problem, DIM, bounds=bounds, samples=1024)
                    feature_results[f"{feature} - {DIM}D"] = 1 - (report.percent_noncompliance / 100.0)
                else:
                    inverse = False
                    if feature in ["NOT Homogeneous", "NOT Basins"]:
                        feature_key = feature.replace("NOT ", "")
                        inverse = True
                    else:
                        feature_key = feature
                    model = xgb.XGBClassifier(objective="binary:logistic")
                    model.load_model(f"dimensions/model_Groups_{feature_key}_scaled_new.json")

                    input_df = pd.DataFrame([all_features_mean], columns=all_features_pandas.columns)
                    if inverse:
                        feature_results[f"{feature} - {DIM}D"] = 1 - model.predict_proba(input_df)[0][1]
                    else:
                        feature_results[f"{feature} - {DIM}D"] = model.predict_proba(input_df)[0][1]

                temp_res = feature_results[f"{feature} - {DIM}D"]
                results.append(temp_res)
                solution.add_metadata(f"score_{feature}_{DIM}D", temp_res)
                feedback += f"{feature} {temp_res:.3f}, "


        score = np.mean(results, axis=0)
        solution.set_scores(
            score,
            f"{feedback} (higher is better, 1.0 is the best).",
        )
        return solution

# "not homogeneous", "not basins"

budget = 100
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
        default="devstral-small-2",
        help="Select the AI model to use for code generation.",
    )
    
    args = parser.parse_args()
    ai_model = args.ai_model
    if args.llm == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        llm = OpenAI_LLM(api_key, ai_model, temperature=1.0)
    elif args.llm == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        llm = Gemini_LLM(api_key, ai_model)
    elif args.llm == "ollama": 
        llm = Ollama_LLM(ai_model)
    # Execution code starts here
    #api_key = os.getenv("OPENAI_API_KEY")
    api_key_gemini = os.getenv("GEMINI_API_KEY")
    api_key_openai = os.getenv("OPENAI_API_KEY")
    llm1 = OpenAI_LLM(api_key_openai, "gpt-5-mini", temperature=1.0)
    llm2 = Gemini_LLM(api_key_gemini, "gemini-3-flash-preview")
    llm3 = Gemini_LLM(api_key_gemini, "gemini-2.5-flash-lite")
    llm4 = Ollama_LLM("devstral-small-2")
    
    #llm = Gemini_LLM(api_key, ai_model)

    llm = Multi_LLM([llm1,llm2,llm3,llm4])

    all_features = ["Separable", "GlobalLocal", "Multimodality", "Basins", "Homogeneous"] 
    feature_combinations = []
    for i in range(len(all_features)):
        for j in range(i+1, len(all_features)):
            feature_combinations.append([all_features[i], all_features[j]])
        feature_combinations.append([all_features[i]])
    
    not_features = ["NOT Basins", "NOT Homogeneous"] 
    rest_features = ["Separable", "GlobalLocal", "Multimodality"] 
    for i in range(len(not_features)):
        for j in range(len(rest_features)):
            feature_combinations.append([not_features[i], rest_features[j]])
        feature_combinations.append([not_features[i]])

    # print(len(feature_combinations))
    # exit()
    for combi in feature_combinations[0:1]:
        niching=None
        experiment_name = f"ELA-{'_'.join([f for f in combi])}"

        niching="novelty"
        experiment_name = f"ELA-{'_'.join([f for f in combi])}-{niching}"
        problem = ELAproblem(name=f"ELA_{'_'.join(combi)}", features=combi, dims=[2,5,10], eval_timeout=1200)

        mutation_prompts = []
        for feature in problem.features:
            mutation_prompts.append(f"Create a new and novel landscape class based on the selected code and improve the {feature} score, meaning: {problem.feature_descriptions[feature]}.")
        mutation_prompts.append("Create a new landscape class that is completely different from the selected solution but still adheres to the properties outlined in the task description.")

        

        for experiment_i in [1]:
            es = LLaMEA(
                problem.evaluate_function,
                n_parents=10,
                n_offspring=30,
                llm=llm,
                task_prompt=problem.task_prompt,
                example_prompt=problem.example_prompt,
                output_format_prompt=problem.format_prompt,
                mutation_prompts=mutation_prompts,
                experiment_name=experiment_name,
                elitism=False,
                HPO=False,
                budget=budget,
                max_workers=10,
                parallel_backend="loky",
                niching=niching,
                novelty_k=5,
                distance_metric=ela_distance,
                eval_timeout=3600,
            )
            print(es.run())

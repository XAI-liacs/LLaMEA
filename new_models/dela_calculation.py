#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:54:36 2026

@author: urbanskvorc
"""
from ioh import get_problem, ProblemClass
from pflacco.classical_ela_features import *
from pflacco.sampling import create_initial_sample
import pandas as pd
from pflacco.deep_ela import load_large_25d_v1
from pflacco.deep_ela import load_large_50d_v1


model_25 = load_large_25d_v1()
model_50 = load_large_50d_v1()


def problem(x):
    return np.sum(x*2)

DIM=10
problem_features_25 = None
problem_features_50 = None
for seed in range(5):
    seed = seed
    X = create_initial_sample(DIM,n=250*DIM, lower_bound = -5, upper_bound = 5, seed=seed)
    y = X.apply(lambda x: problem(list(np.array(x))), axis = 1)
    
    fdc25 = model_25(X, y, include_costs=False)
    fdc25 =  pd.DataFrame.from_dict(fdc25, orient='index')    
    fdc50 = model_50(X, y, include_costs=False)
    fdc50 =  pd.DataFrame.from_dict(fdc50, orient='index') 
    
    fdc25 = fdc25.transpose()
    fdc50 = fdc50.transpose()

    problem_features_25 = pd.concat([problem_features_25, fdc25])
    problem_features_50 = pd.concat([problem_features_50, fdc50])

fdc25 = pd.DataFrame(problem_features_25.mean()).transpose()
fdc50 = pd.DataFrame(problem_features_50.mean()).transpose()

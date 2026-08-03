"""
Samples the COCO and LLaMEA problems, and collects the maximum fitness value for use in normalization
"""
from ioh import get_problem, ProblemClass
import pandas as pd
import numpy as np
import os
import natsort
import re
import itertools
from itertools import product, combinations
import math

np.random.seed(0)



def find_class_name(code):
    regex_string = "class (.+):\n"
    result = re.search(regex_string,code)
    return result[1]



def get_LLM_problem(group,problem_num, instance_num, dim, folder = "/local/bodasap/LLaMEA/basinsattribution-main/outputs_hybrid"):
    all_files = os.listdir(folder)
    all_files = [x for x in all_files if ".DS_Store" not in x]
    all_files = natsort.natsorted(all_files)
    file = all_files[group]
    
    problem_file = f"{folder}/{file}"
    try:
        
        jsonObj = pd.read_json(path_or_buf=problem_file, lines=True)
        if problem_num >= len(jsonObj):
            return None
        code=jsonObj.iloc[problem_num].code
        exec(code)
        class_name = find_class_name(code)
        #check if we imported the problem correctly
        problem_class = locals()[class_name]
        problem = problem_class(dim)
        problem.f([1]*dim)
    except Exception as e:
        print(f"Error in problem group {group} in problem {problem_num}")
        print(code)
        print(e)
        return None
        
    return (problem.f, jsonObj.iloc[problem_num].id)




rows = []
DIM = 2
#group is the property/property combination number
for group in range(23):
    for function_id in range(56):
        print(group, function_id)
        result = get_LLM_problem(group, function_id, 1, DIM)
        if result is None:
            continue
        problem, _id = result
        X = pd.DataFrame(np.random.uniform(-5,5,size=(10000,2)))
        y = X.apply(lambda x: problem(list(np.array(x))), axis = 1)
        
        min_orig = np.min(y)
        y = y - min_orig
        _min = np.min(y)
        _max = np.max(y)
        _mean = np.mean(y)
        df_dict = {"min_orig": min_orig, "min": [_min], "max": [_max], "mean": [_mean], "id": [_id]}
        df = pd.DataFrame.from_dict(df_dict)
        rows.append(df)
            
final_result = pd.concat(rows)
final_result.to_csv(f"llm_max.csv")


"""
#Code for the BBOB problems
rows = []
DIM = 2
for instance_index in range(20):
    for i in range(24):
        print(instance_index, i+1)
        problem = get_problem(i+1, instance_index, DIM, ProblemClass.BBOB)
        X = pd.DataFrame(np.random.uniform(-5,5,size=(10000,2)))
        y = X.apply(lambda x: problem(list(np.array(x))), axis = 1)
        
        min_orig = np.min(y)
        y = y - min_orig
        _min = np.min(y)
        _max = np.max(y)
        _mean = np.mean(y)
        df_dict = {"min_orig": min_orig, "min": [_min], "max": [_max], "mean": [_mean], "function_id": [i+1], "instance_id": [instance_index]}
        df = pd.DataFrame.from_dict(df_dict)
        rows.append(df)
            
final_result = pd.concat(rows)
final_result.to_csv(f"ioh_max.csv")
"""




        


        
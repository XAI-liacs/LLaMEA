import os
import numpy as np
import re
from llamea import LLaMEA
import warnings
import time

# Execution code starts here
api_key = os.getenv("OPENAI_API_KEY")
ai_model = "gpt-4o-2024-05-13"  # gpt-4-turbo or gpt-3.5-turbo gpt-4o llama3:70b gpt-4o-2024-05-13, gemini-1.5-flash gpt-4-turbo-2024-04-09
experiment_name = "TSP"
if "gemini" in ai_model:
    api_key = os.environ["GEMINI_API_KEY"]

from itertools import product
import numpy as np
from problems.user_tsp_gls.prob import TSPGLS

tsp_prob = TSPGLS()


def evaluate(
    code, algorithm_name, algorithm_name_long, configuration_space=None, explogger=None
):
    
    
    def evaluateAll():
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Execute the code string in the new module's namespace
                exec(code, globals())
                alg = globals()[algorithm_name]()

                return tsp_prob.evaluateGLSfast(alg)
        except Exception as e:
            print(e)
            return 10000000000
    
    
    # inst_feats = {str(arg): [idx] for idx, arg in enumerate(args)}
    error = ""

    fitness = evaluateAll()

    fitness = -1 * fitness #we optimize (not minimize)

    feedback = f"The heuristic {algorithm_name_long} got an average fitness of {fitness:0.2f} (closer to zero is better)."

    complete_log = {}
    return feedback, fitness, error, complete_log


role_prompt = "You are a highly skilled computer scientist your task it to design novel and efficient heuristics in Python."
task_prompt = """
Task: Given an edge distance matrix and a local optimal route, please help me design a strategy to update the distance matrix to avoid being trapped in the local optimum with the final goal of finding a tour with minimized distance (TSP problem).
You should create an algorithm for me to update the edge distance matrix.
Provide the Python code for the new strategy. The code is a Python class that should contain two functions an "__init__()" function containing any hyper-parameters that can be optimmized, and a 
function called 'update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used)' that takes three inputs, and outputs the 'updated_edge_distance', 
where 'local_opt_tour' includes the local optimal tour of IDs, 'edge_distance' and 'edge_n_used' are matrixes, 'edge_n_used' includes the number of each edge used during permutation. 
All are Numpy arrays. 
The novel function should be sufficiently complex in order to achieve better performance. It is important to ensure self-consistency.

An example heuristic to show the structure is as follows.
```python
import numpy as np

class Sample:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
        # code here
        return updated_edge_distance
```

Give an excellent and novel heuristic to solve this task and also give it a name. Give the response in the format:
# Name: <name>
# Code: <code>
"""

feedback_prompt = (
    f"Either refine or redesign to improve the solution (and give it a distinct name). Give the response in the format:\n"
    f"# Name: <name>\n"
    f"# Code: <code>\n"
)

for experiment_i in [1,2,3]:
    es = LLaMEA(
        evaluate,
        budget=500,
        role_prompt=role_prompt,
        task_prompt=task_prompt,
        feedback_prompt=feedback_prompt,
        api_key=api_key,
        experiment_name=experiment_name,
        model=ai_model,
        elitism=True,
        HPO=False,
    )
    print(es.run())

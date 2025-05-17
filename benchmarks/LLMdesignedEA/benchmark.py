import os
import numpy as np
from ioh import get_problem, wrap_problem, logger
import ioh
import re
import sys
sys.path.append(".")
# from llamea.individual import Individual
from llamea import llm
from llamea import LLaMEA
from benchmarks.LLMdesignedEA.GNBG.GNBG_instances import load_problem
from misc import (
    aoc_logger,
    correct_aoc,
    OverBudgetException,
    budget_logger,
    ThresholdReachedException,
)


def calculate_objective(intance, dim):
    # print(dim, intance)
    gnbg = load_problem(intance, 0)
    # print(gnbg.OptimumPosition.flatten().tolist())
    return gnbg.OptimumPosition.flatten().tolist(), 0.0

if __name__ == "__main__":
    runs = 31
    best_alg_path = "exp-05-16_185247-LLaMEA-gpt-4o-2024-08-06-competition/code/try-97-AdaptiveDifferentialEvolution.py"
    best_alg_name = "AdaptiveDifferentialEvolution"
    best_alg = open(best_alg_path, "r")
    best_alg = best_alg.read()
    exec(best_alg, globals())
    l1 = logger.Analyzer(
        folder_name="LLaMEA_best",
        algorithm_name="AdaptiveDifferentialEvolution",
        store_positions=True,
        root="./",
    )
    for fid in range(1, 25):
        gnbg = load_problem(fid)
        dim = gnbg.Dimension
        budget = gnbg.MaxEvals
        wrap_problem(
            lambda x: gnbg.fitness(x) - gnbg.OptimumValue,
            f"gnbg{fid}",
            ioh.ProblemClass.REAL,
            lb=-100,
            ub=100,
            dimension=dim,
            instance=fid,
            calculate_objective=calculate_objective,
        )
        problem = get_problem(
            f"gnbg{fid}",
            dimension=dim,
            instance=fid,
            problem_class=ioh.ProblemClass.REAL,
        )
        problem.attach_logger(l1)
        algorithm = globals()[best_alg_name](budget=budget, dim=dim)
        for run in range(runs):
            algorithm(problem)
            with open(f"results/LLaMEA_best/f_{fid}_value.txt", "w") as f:
                f.write(str(gnbg.BestFoundResult - gnbg.OptimumValue))
            with open(f"results/LLaMEA_best/f_{fid}_params.txt", "w") as f:
                content = ""
                for i in range(len(gnbg.BestFoundPosition)):
                    x = gnbg.BestFoundPosition[i]
                    content += str(x[0])
                    if i != len(gnbg.BestFoundPosition) - 1:
                        content += ","
                f.write(content)
            problem.reset()
    l1.close()

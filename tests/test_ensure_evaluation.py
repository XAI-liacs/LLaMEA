import math
from random import random
from llamea.llamea import LLaMEA
from llamea.llm import Dummy_LLM
from llamea.multi_objective_fitness import Fitness
from llamea.solution import Solution

def f(solution: Solution, logger):
    solution.fitness = random()
    return solution

def f_multi(solution: Solution, logger):
    solution.fitness = Fitness({
        'Distance' : random(),
        'Fuel': random()
    })

def test_ensure_evaluation_resolve_single_objective_maximisation():
    llamea = LLaMEA(f, Dummy_LLM(), budget=10, minimization=False, experiment_name='test_ensure_single_max')
    solution = Solution()
    assert math.isnan(solution.fitness)
    solution = llamea._ensure_fitness_evaluates([solution])[0]
    assert solution.fitness == float('-inf')

def test_ensure_evaluation_resolve_single_objective_minimisation():
    llamea = LLaMEA(f, Dummy_LLM(), budget=10, minimization=True, experiment_name='test_ensure_single_min')
    solution = Solution()
    assert math.isnan(solution.fitness)
    solution = llamea._ensure_fitness_evaluates([solution])[0]
    assert solution.fitness == float('inf')

def test_ensure_evaluation_resolve_multi_objective_maximisation():
    llamea = LLaMEA(f_multi, Dummy_LLM(), budget=10, minimization=False, experiment_name='test_ensure_multi_max', multi_objective=True, multi_objective_keys=['Distance', 'Fuel'])
    solution = Solution()
    assert math.isnan(solution.fitness)
    solution = llamea._ensure_fitness_evaluates([solution])[0]
    assert solution.fitness == Fitness({
        "Distance" : float('-inf'),
        "Fuel": float('-inf')
    })

def test_ensure_evaluation_resolve_multi_objective_minimisation():
    llamea = LLaMEA(f_multi, Dummy_LLM(), budget=10, minimization=True, experiment_name='test_ensure_multi_min', multi_objective=True, multi_objective_keys=['Distance', 'Fuel'])
    solution = Solution()
    assert math.isnan(solution.fitness)
    solution = llamea._ensure_fitness_evaluates([solution])[0]
    assert solution.fitness == Fitness({
        "Distance" : float('inf'),
        "Fuel": float('inf')
    })

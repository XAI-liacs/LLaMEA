import numpy as np
import random
import pytest

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from llamea.pareto_archive import ParetoArchive
from llamea.solution import Solution

def test_pareto_archive_fails_on_single_objective_solutions():

    def evaluate(solution: Solution) -> Solution:
        fitness = random.random()
        solution.fitness = fitness
        return solution
    
    solution = Solution()
    solution = evaluate(solution)
    archieve = ParetoArchive(minimisation=True)

    with pytest.raises(AssertionError):
        archieve.add_solution(solution)
    
def test_dominates_stub_works_as_expected():
    
    solution1 = Solution()
    solution2 = Solution()
    solution3 = Solution()
    solution1.fitness = {
        'f(1)': 10,
        'f(2)': 30
    }
    solution2.fitness = {
        'f(1)': 20,
        'f(2)': 40
    }
    solution3.fitness = {
        'f(1)': 8,
        'f(2)': 32
    }
    archieve = ParetoArchive(minimisation=True)
    assert archieve._dominates(solution1, solution2) == True
    assert archieve._dominates(solution2, solution1) == False
    assert archieve._dominates(solution1, solution3) == False

def test_pareto_saves_first_front():

    def evaluate(solution: Solution) -> Solution:
        fitness = {}
        for i in range(1, 3):
            fitness[f'f({i})'] = random.random()
        solution.fitness = fitness
        return solution
    
    solutions = [Solution() for _ in range(30)]
    solutions = [evaluate(solution) for solution in solutions]
    archieve = ParetoArchive(minimisation=True)
    for solution in solutions:
        dominated = archieve.add_solution(solution)
        print(len(archieve.archive), solution.get_fitness_vector(), dominated)
    
    all_solution_fitness = np.asarray([solution.get_fitness_vector() for solution in solutions], dtype=float)
    nds = NonDominatedSorting()
    pf_index = nds.do(all_solution_fitness)[0]
    pf = [solutions[index] for index in pf_index]

    archieved_front = archieve.get_best()
    for front_solution in archieved_front:
        assert front_solution in pf
    
    assert len(archieved_front) == len(pf)


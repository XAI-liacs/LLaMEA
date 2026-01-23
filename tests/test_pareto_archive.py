import numpy as np
import random
import pytest

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from llamea.multi_objective_fitness import Fitness
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
        archieve.add_solutions([solution])

def test_pareto_saves_first_front():

    def evaluate(solution: Solution) -> Solution:
        fitness = Fitness()
        for i in range(1, 3):
            fitness[f'f({i})'] = random.random()
        solution.fitness = fitness
        return solution
    
    solutions = [Solution() for _ in range(30)]
    solutions = [evaluate(solution) for solution in solutions]
    archieve = ParetoArchive(minimisation=True)
    archieve.add_solutions(solutions)
    print(f"Archieve size: {len(archieve.archive)}")
    for solution in archieve.archive:
        print(solution.fitness)
    
    all_solution_fitness = np.asarray([solution.get_fitness_vector() for solution in solutions], dtype=float)
    nds = NonDominatedSorting()
    pf_index = nds.do(all_solution_fitness)[0]
    pf = [solutions[index] for index in pf_index]

    archieved_front = archieve.get_best()
    for front_solution in archieved_front:
        assert front_solution in pf
    
    assert len(archieved_front) == len(pf)

def test_pareto_does_not_copy_solution_across_iterations():
    def evaluate(solution: Solution) -> Solution:
        fitness = Fitness()
        for i in range(1, 3):
            fitness[f'f({i})'] = random.random()
        solution.fitness = fitness
        return solution
    
    solutions = [Solution() for _ in range(40)]
    solutions = [evaluate(solution) for solution in solutions]
    archieve = ParetoArchive(minimisation=True)
    archieve.add_solutions(solutions)
    archieve.add_solutions(solutions[:])
    print(f"Archieve size: {len(archieve.archive)}")
    for solution in archieve.archive:
        print(solution.fitness)
    
    all_solution_fitness = np.asarray([solution.get_fitness_vector() for solution in solutions], dtype=float)
    nds = NonDominatedSorting()
    pf_index = nds.do(all_solution_fitness)[0]
    pf = [solutions[index] for index in pf_index]

    archieved_front = set(archieve.get_best())
    for front_solution in archieved_front:
        assert front_solution in pf
    
    assert len(archieved_front) == len(pf)
    
    for soln1 in archieved_front:
        for soln2 in archieved_front - set([soln1]):
            assert soln1.id != soln2.id


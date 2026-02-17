import numpy as np
import random
import pytest

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from llamea.multi_objective_fitness import Fitness
from llamea.pareto_archive import ParetoArchive
from llamea.solution import Solution

def test_pareto_archive_handles_invalid_solutions():
    
    solutions = []
    solution = Solution()
    solution.set_scores(Fitness({'obj1': float('nan'), 'obj2': 2.0}), "Invalid fitness")
    solutions.append(solution)
    solution = Solution()
    solution.set_scores(Fitness({'obj1': 1.0, 'obj2': float('inf')}), "Invalid fitness")
    solutions.append(solution)
    solution = Solution()
    solution.set_scores(float('nan'), feedback="Invalid fitness type")
    solutions.append(solution)
    solution = Solution()
    solution.set_scores(float('inf'), feedback="Invalid fitness type")
    solutions.append(solution)
    solution = Solution()
    solution.set_scores(Fitness({'obj1': 1.0, 'obj2': 2.0}), "Valid fitness")
    solutions.append(solution)
    archieve = ParetoArchive(minimisation=True)
    print(solutions)
    archieve.add_solutions(solutions)
    assert len(archieve.archive) == 1

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

def test_pareto_does_not_copy_solution_add_same_solutions():
    recurring_solution = Solution()
    fitness = Fitness({"Distance": 10, "Fuel": 12})

    recurring_solution.set_scores(fitness, f"Got fitness {fitness}, try minimising further.")
    solutions = []
    for i in range(100):
        if i % 2 == 0:
            solutions.append(recurring_solution)
        else:
            solution = Solution()
            fitness = Fitness({"Distance": random.randint(11, 30), "Fuel": random.randint(13, 30)})
            solution.set_scores(fitness, f"Got fitness {fitness}, try minimising further.")

    pareto_archive = ParetoArchive(minimisation=True)
    for i in range(0, 100, 5):
        new_solutions = solutions[i : i + 5]
        pareto_archive.add_solutions(new_solutions)
    assert len([individual for individual in pareto_archive.get_best() if individual.id == recurring_solution.id]) == 1




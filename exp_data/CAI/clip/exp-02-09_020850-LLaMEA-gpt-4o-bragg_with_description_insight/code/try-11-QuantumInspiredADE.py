import numpy as np
from scipy.optimize import minimize

class QuantumInspiredADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.base_mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.local_search_prob = 0.2
        self.best_solution = None
        self.wave_amplitude = 0.1
        self.global_best = None

    def quantum_wave_function(self, solution, lb, ub):
        perturbation = self.wave_amplitude * np.sin(2 * np.pi * np.random.rand(self.dim))
        return np.clip(solution + perturbation, lb, ub)

    def initialize_population(self, lb, ub):
        population = lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)
        return population

    def adapt_mutation_factor(self, generation):
        return self.base_mutation_factor + 0.1 * np.sin(2 * np.pi * generation / 10)

    def mutate(self, population, idx, generation):
        indices = np.random.choice(range(self.pop_size), 3, replace=False)
        while idx in indices:
            indices = np.random.choice(range(self.pop_size), 3, replace=False)
        a, b, c = population[indices]
        mutation_factor = self.adapt_mutation_factor(generation)
        mutant = a + mutation_factor * (b - c)
        return np.clip(mutant, 0, 1)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, target, trial, func):
        return trial if func(trial) < func(target) else target

    def local_refinement(self, solution, func, lb, ub):
        res = minimize(func, solution, bounds=[(lb[i], ub[i]) for i in range(self.dim)])
        return res.x

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        evaluations = 0
        generation = 0

        while evaluations < self.budget:
            new_population = []
            for idx in range(self.pop_size):
                target = population[idx]
                mutant = self.mutate(population, idx, generation)
                trial = self.crossover(target, mutant)
                trial = lb + (ub - lb) * trial
                quantum_trial = self.quantum_wave_function(trial, lb, ub)
                selected = self.select(target, quantum_trial, func)
                if np.random.rand() < self.local_search_prob:
                    selected = self.local_refinement(selected, func, lb, ub)
                new_population.append(selected)
                evaluations += 1
                if evaluations >= self.budget:
                    break
            population = np.array(new_population)
            generation += 1

        best_idx = np.argmin([func(ind) for ind in population])
        best_solution = population[best_idx]

        if self.best_solution is None or func(best_solution) < func(self.best_solution):
            self.best_solution = best_solution

        result = minimize(func, self.best_solution, bounds=[(lb[i], ub[i]) for i in range(self.dim)])
        return result.x
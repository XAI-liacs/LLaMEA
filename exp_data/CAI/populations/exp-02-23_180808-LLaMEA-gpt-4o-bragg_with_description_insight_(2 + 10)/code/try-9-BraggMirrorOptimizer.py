import numpy as np
from scipy.optimize import minimize
from copy import deepcopy

class BraggMirrorOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_probability = 0.7
        self.best_solution = None
        self.best_fitness = float('-inf')

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                donor_vector = self.mutation(population, i, evaluations)
                trial_vector = self.crossover(population[i], donor_vector)

                # Evaluate trial vector
                trial_fitness = func(trial_vector)
                evaluations += 1

                # Select based on fitness
                if trial_fitness > func(population[i]):
                    population[i] = trial_vector
                    if trial_fitness > self.best_fitness:
                        self.best_solution = deepcopy(trial_vector)
                        self.best_fitness = trial_fitness

            # Local search with periodicity encouragement
            self.local_search(func, lb, ub)

        return self.best_solution

    def initialize_population(self, lb, ub):
        return [np.random.uniform(lb, ub, self.dim) for _ in range(self.population_size)]

    def mutation(self, population, target_idx, evaluations):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        adaptive_factor = self.mutation_factor * (1 + 0.5 * np.sin(2 * np.pi * evaluations / self.budget))
        return population[a] + adaptive_factor * (population[b] - population[c])

    def crossover(self, target, donor):
        crossover = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        return np.where(crossover, donor, target)

    def local_search(self, func, lb, ub):
        # Encourage periodicity by using a periodic perturbation
        for _ in range(2):  # Two iterations for periodicity injection
            if self.best_solution is not None:
                periodic_solution = self.best_solution * (1 + 0.1 * np.sin(2 * np.pi * np.arange(self.dim) / self.dim))
                result = minimize(func, periodic_solution, bounds=list(zip(lb, ub)), method='L-BFGS-B')
                if result.fun > self.best_fitness:
                    self.best_solution = result.x
                    self.best_fitness = result.fun
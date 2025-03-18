import numpy as np
from scipy.optimize import minimize

class HybridOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 20
        F = 0.8
        CR = 0.85
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (population_size, self.dim))
        opposition_population = bounds[0] + bounds[1] - population
        population = np.vstack((population, opposition_population))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = len(population)

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break
                indices = [idx for idx in range(2 * population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                F_adaptive = F * np.random.uniform(0.7, 1.3)  # Adjusted adaptive F range
                mutant_vector = np.clip(population[a] + F_adaptive * (population[b] - population[c]), bounds[0], bounds[1])
                trial_vector = np.array([
                    mutant_vector[j] if np.random.rand() < CR else population[i][j]
                    for j in range(self.dim)
                ])
                trial_vector = self._enforce_periodicity(trial_vector)
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            result = minimize(func, best_solution, bounds=list(zip(bounds[0], bounds[1])), method='L-BFGS-B')
            if result.fun < fitness[best_idx]:
                population[best_idx] = result.x
                fitness[best_idx] = result.fun
                evaluations += result.nfev

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def _enforce_periodicity(self, vector):
        period = 2
        periodic_vector = vector.copy()
        for i in range(0, self.dim, period):
            block_average = np.mean(vector[i:i+period])
            periodic_vector[i:i+period] = block_average
        return periodic_vector
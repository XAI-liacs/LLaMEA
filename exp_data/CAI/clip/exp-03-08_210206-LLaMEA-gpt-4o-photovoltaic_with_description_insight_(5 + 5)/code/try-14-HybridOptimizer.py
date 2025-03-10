import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Population size for DE
        self.F = 0.85  # DE mutation factor adjusted for better exploration
        self.CR = 0.9  # DE crossover probability
        self.population = np.random.rand(self.population_size, dim)
        self.best_solution = None
        self.best_score = float('inf')

    def differential_evolution(self, func, bounds):
        # Initialize population within bounds
        for i in range(self.population_size):
            self.population[i] = bounds['lb'] + self.population[i] * (bounds['ub'] - bounds['lb'])

        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation step
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, bounds['lb'], bounds['ub'])

                # Crossover step
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection step
                trial_score = func(trial)
                evaluations += 1
                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_solution = trial

                if trial_score < func(self.population[i]):
                    self.population[i] = trial

                if evaluations >= self.budget:
                    break

    def local_search(self, func, x0, bounds):
        result = minimize(func, x0, method='SLSQP', bounds=list(zip(bounds['lb'], bounds['ub'])))
        return result.x if result.success else x0

    def __call__(self, func):
        bounds = {'lb': func.bounds.lb, 'ub': func.bounds.ub}

        # Step 1: Global Optimization with DE
        self.differential_evolution(func, bounds)

        # Step 2: Local Search for Refinement
        if self.best_solution is not None:
            self.best_solution = self.local_search(func, self.best_solution, bounds)

        return self.best_solution
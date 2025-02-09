import numpy as np
from scipy.optimize import minimize

class HybridOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def quasi_oppositional_initialization(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        opp_population = lb + ub - population
        return np.vstack((population, opp_population))

    def differential_evolution(self, population, bounds, func):
        new_population = np.copy(population)
        for i in range(len(population)):
            idxs = [idx for idx in range(len(population)) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.F * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            if func(trial) < func(population[i]):
                new_population[i] = trial
        return new_population

    def local_optimization(self, x0, func, bounds):
        result = minimize(func, x0, bounds=bounds, method='L-BFGS-B')
        return result.x if result.success else x0

    def __call__(self, func):
        bounds = func.bounds
        population = self.quasi_oppositional_initialization(bounds)
        evaluations = 0

        while evaluations < self.budget:
            population = self.differential_evolution(population, bounds, func)
            evaluations += len(population)
            if evaluations < self.budget:
                for i in range(len(population)):
                    population[i] = self.local_optimization(population[i], func, bounds)
                    evaluations += 1
                    if evaluations >= self.budget:
                        break

        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]